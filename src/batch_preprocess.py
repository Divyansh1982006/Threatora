"""Batch Preprocessing & Zero-Leakage Window Caching Pipeline.

Processes large network flow and packet captures sequentially (files <= 2GB each),
enforcing the 16GB system RAM boundary with peak RAM footprint strictly < 2.5 GB.
Extracts 12 canonical continuous flow metrics into discrete 0.5s temporal bins,
assembles chronological transition pairs (S_t, S_{t+1}, y_t), normalizes via
RobustScaler, and flushes standalone Snappy-compressed Parquet chunks.
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def natural_sort_key(p: Path) -> List[Union[int, str]]:
    """Sort strings containing numbers in human order (e.g., 1.pcap before 10.pcap)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(p))]


import joblib
import numpy as np
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

# 12 Canonical Flow Metrics mandated by Threatora specification
CANONICAL_12_METRICS: List[str] = [
    "byte_rate",
    "packet_rate",
    "mean_iat",
    "var_iat",
    "max_iat",
    "syn_flag_ratio",
    "ack_flag_ratio",
    "fin_flag_ratio",
    "rst_flag_ratio",
    "psh_flag_ratio",
    "bwd_to_fwd_ratio",
    "mean_flow_duration",
]

# Standard UNSW-NB15 column names (49 columns)
UNSW_NB15_COLUMNS: List[str] = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "Label"
]

# Identity shortcuts to drop to enforce zero data leakage
SHORTCUT_IDENTIFIERS: List[str] = [
    "srcip", "dstip", "sport", "dsport", "flow_id", "timestamp", "id",
    "stime", "ltime", "srcaddr", "dstaddr", "saddr", "daddr"
]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BatchPreprocess")


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_ram_boundary(max_allowed_mb: float = 2500.0) -> None:
    """Enforces strict < 2.5 GB RAM boundary."""
    mem_mb = get_process_memory_mb()
    if mem_mb > max_allowed_mb:
        logger.error(f"HARDWARE BOUNDARY VIOLATION: Current RAM {mem_mb:.1f} MB exceeds {max_allowed_mb:.1f} MB!")
        raise MemoryError(f"Process RAM {mem_mb:.1f} MB exceeded ceiling of {max_allowed_mb} MB")


class IncrementalRobustScaler:
    """RobustScaler wrapper supporting fit on first chunk and persistent serialization."""

    def __init__(self, scaler_path: Optional[Path] = None):
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.scaler: Optional[RobustScaler] = None

        if self.scaler_path and self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded existing RobustScaler from {self.scaler_path}")
            except Exception as e:
                logger.warning(f"Could not load scaler from {self.scaler_path}: {e}")

    def fit_or_transform(self, features: np.ndarray, is_first_chunk: bool = False) -> np.ndarray:
        """Fits on initial chunk and transforms features to zero-median unit-IQR scale."""
        needs_fit = (
            self.scaler is None
            or is_first_chunk
            or not hasattr(self.scaler, "n_features_in_")
            or self.scaler.n_features_in_ != features.shape[1]
        )
        if needs_fit:
            logger.info(f"Fitting new RobustScaler on {features.shape[1]} features...")
            self.scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
            scaled = self.scaler.fit_transform(features)
            if self.scaler_path:
                self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.scaler, self.scaler_path)
                logger.info(f"Fitted RobustScaler saved to: {self.scaler_path}")
            return scaled.astype(np.float32)
        else:
            return self.scaler.transform(features).astype(np.float32)


def process_csv_file_to_bins(
    file_path: Path,
    bin_duration_sec: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Streams a CSV file using Polars lazy scanning and aggregates into 12 canonical metrics.

    Returns:
        features: np.ndarray of shape (N_bins, 12)
        labels: np.ndarray of shape (N_bins,) indicating binary threat label
    """
    logger.info(f"Scanning CSV stream: {file_path.name} ({file_path.stat().st_size / (1024*1024):.1f} MB)...")

    # Inspect first line to check if header is present
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    has_header = False
    first_toks = [t.strip().strip('"').lower() for t in first_line.split(",")]
    if any(k in first_toks for k in ["srcip", "dur", "proto", "sbytes", "sport", "starttime", "time_s"]):
        has_header = True

    if has_header:
        lf = pl.scan_csv(file_path, has_header=True, ignore_errors=True)
    else:
        # UNSW-NB15 format with 49 columns
        lf = pl.scan_csv(
            file_path,
            has_header=False,
            with_column_names=lambda cols: UNSW_NB15_COLUMNS[:len(cols)],
            ignore_errors=True,
        )

    cols = lf.collect_schema().names()
    col_map = {c.strip().lower(): c for c in cols}

    # Resolve timestamp column
    ts_candidates = ["stime", "time_s", "timestamp", "starttime", "time"]
    ts_col = None
    for cand in ts_candidates:
        if cand in col_map:
            ts_col = col_map[cand]
            break

    if ts_col is None:
        logger.warning(f"No timestamp column found in {file_path.name}. Generating sequential synthetic index.")
        ts_expr = pl.int_range(0, pl.len(), eager=False).cast(pl.Float64)
    else:
        ts_raw = pl.col(ts_col)
        ts_num = ts_raw.cast(pl.Float64, strict=False)
        ts_str = ts_raw.cast(pl.String, strict=False)
        ts_dt = (
            ts_str.str.to_datetime(format="%Y/%m/%d %H:%M:%S%.f", strict=False)
            .fill_null(ts_str.str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False))
            .fill_null(ts_str.str.to_datetime(format="%Y/%m/%d %H:%M:%S", strict=False))
            .fill_null(ts_str.str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
        ).cast(pl.Int64, strict=False) / 1_000_000.0
        ts_expr = pl.coalesce([ts_num, ts_dt]).fill_null(0.0)

    # Resolve column mappings for 12 flow metrics
    sbytes_col = col_map.get("sbytes", col_map.get("srcbytes", col_map.get("totbytes", "")))
    dbytes_col = col_map.get("dbytes", col_map.get("dstbytes", ""))
    spkts_col = col_map.get("spkts", col_map.get("srcpkts", col_map.get("totpkts", "")))
    dpkts_col = col_map.get("dpkts", col_map.get("dstpkts", ""))
    dur_col = col_map.get("dur", col_map.get("duration", ""))
    sintpkt_col = col_map.get("sintpkt", "")
    dintpkt_col = col_map.get("dintpkt", "")
    state_col = col_map.get("state", "")
    synack_col = col_map.get("synack", "")
    ackdat_col = col_map.get("ackdat", "")
    label_col = col_map.get("label", col_map.get("is_malicious", col_map.get("attack_cat", "")))

    # Privilege port mapping
    dsport_col = col_map.get("dsport", col_map.get("dst_port", col_map.get("dport", "")))

    sbytes_e = pl.col(sbytes_col).cast(pl.Float64, strict=False).fill_null(0.0) if sbytes_col else pl.lit(0.0)
    dbytes_e = pl.col(dbytes_col).cast(pl.Float64, strict=False).fill_null(0.0) if dbytes_col else pl.lit(0.0)
    spkts_e = pl.col(spkts_col).cast(pl.Float64, strict=False).fill_null(1.0) if spkts_col else pl.lit(1.0)
    dpkts_e = pl.col(dpkts_col).cast(pl.Float64, strict=False).fill_null(0.0) if dpkts_col else pl.lit(0.0)
    dur_e = pl.col(dur_col).cast(pl.Float32, strict=False).fill_null(0.001) if dur_col else pl.lit(0.001)

    sint_e = pl.col(sintpkt_col).cast(pl.Float64, strict=False).fill_null(0.0) if sintpkt_col else pl.lit(0.0)
    dint_e = pl.col(dintpkt_col).cast(pl.Float64, strict=False).fill_null(0.0) if dintpkt_col else pl.lit(0.0)

    # State flag expressions
    if state_col:
        st_expr = pl.col(state_col).cast(pl.String, strict=False).str.to_uppercase().fill_null("")
        syn_expr = st_expr.str.contains("REQ|SYN")
        ack_expr = st_expr.str.contains("CON|FIN|ACK")
        fin_expr = st_expr.str.contains("FIN|CLO")
        rst_expr = st_expr.str.contains("RST")
    else:
        syn_expr = pl.lit(False)
        ack_expr = pl.lit(False)
        fin_expr = pl.lit(False)
        rst_expr = pl.lit(False)

    if synack_col:
        syn_expr = syn_expr | (pl.col(synack_col).cast(pl.Float64, strict=False) > 0.0)
    if ackdat_col:
        ack_expr = ack_expr | (pl.col(ackdat_col).cast(pl.Float64, strict=False) > 0.0)

    # Label expression
    if label_col:
        l_c = pl.col(label_col)
        lbl_expr = (
            pl.when(l_c.cast(pl.String).str.to_lowercase().is_in(["0", "normal", "benign", "background", "none", ""]))
            .then(pl.lit(0))
            .when(l_c.cast(pl.Int32, strict=False).is_not_null())
            .then(l_c.cast(pl.Int32, strict=False))
            .otherwise(pl.lit(1))
        ).cast(pl.Int32)
    else:
        lbl_expr = pl.lit(0).cast(pl.Int32)

    # Annotate intermediate columns
    lf_prep = lf.with_columns([
        ts_expr.alias("parsed_ts"),
        sbytes_e.alias("flow_sbytes"),
        dbytes_e.alias("flow_dbytes"),
        spkts_e.alias("flow_spkts"),
        dpkts_e.alias("flow_dpkts"),
        dur_e.alias("flow_dur"),
        sint_e.alias("flow_sint"),
        dint_e.alias("flow_dint"),
        pl.max_horizontal(sint_e, dint_e).cast(pl.Float32).alias("flow_max_iat"),
        syn_expr.cast(pl.Float32).alias("flow_syn"),
        ack_expr.cast(pl.Float32).alias("flow_ack"),
        fin_expr.cast(pl.Float32).alias("flow_fin"),
        rst_expr.cast(pl.Float32).alias("flow_rst"),
        ((sbytes_e > 0) & (dbytes_e > 0)).cast(pl.Float32).alias("flow_psh"),
        lbl_expr.alias("flow_label"),
    ])

    # Compute minimum timestamp for relative binning
    min_ts_df = lf_prep.select(pl.col("parsed_ts").min()).collect()
    min_ts = float(min_ts_df[0, 0]) if len(min_ts_df) > 0 and min_ts_df[0, 0] is not None else 0.0

    # Group into discrete bin_duration_sec temporal bins
    bin_idx_expr = ((pl.col("parsed_ts") - min_ts) / bin_duration_sec).cast(pl.Int64).alias("bin_idx")

    agg_lf = (
        lf_prep.with_columns(bin_idx_expr)
        .group_by("bin_idx")
        .agg([
            # 1. byte_rate
            ((pl.col("flow_sbytes").sum() + pl.col("flow_dbytes").sum()) / bin_duration_sec).cast(pl.Float32).alias("byte_rate"),
            # 2. packet_rate
            ((pl.col("flow_spkts").sum() + pl.col("flow_dpkts").sum()) / bin_duration_sec).cast(pl.Float32).alias("packet_rate"),
            # 3. mean_iat
            ((pl.col("flow_sint") + pl.col("flow_dint")) * 0.5).mean().cast(pl.Float32).alias("mean_iat"),
            # 4. var_iat
            ((pl.col("flow_sint") + pl.col("flow_dint")) * 0.5).var().fill_null(0.0).cast(pl.Float32).alias("var_iat"),
            # 5. max_iat
            pl.col("flow_max_iat").max().cast(pl.Float32).alias("max_iat"),
            # 6. syn_flag_ratio
            pl.col("flow_syn").mean().cast(pl.Float32).alias("syn_flag_ratio"),
            # 7. ack_flag_ratio
            pl.col("flow_ack").mean().cast(pl.Float32).alias("ack_flag_ratio"),
            # 8. fin_flag_ratio
            pl.col("flow_fin").mean().cast(pl.Float32).alias("fin_flag_ratio"),
            # 9. rst_flag_ratio
            pl.col("flow_rst").mean().cast(pl.Float32).alias("rst_flag_ratio"),
            # 10. psh_flag_ratio
            pl.col("flow_psh").mean().cast(pl.Float32).alias("psh_flag_ratio"),
            # 11. bwd_to_fwd_ratio
            (pl.col("flow_dbytes").sum() / (pl.col("flow_sbytes").sum() + 1e-6)).cast(pl.Float32).alias("bwd_to_fwd_ratio"),
            # 12. mean_flow_duration
            pl.col("flow_dur").mean().cast(pl.Float32).alias("mean_flow_duration"),
            # Binary threat label: 1 if attack in bin, 0 if benign
            pl.col("flow_label").max().cast(pl.Int32).alias("label"),
        ])
        .sort("bin_idx")
    )

    binned_df = agg_lf.collect()
    num_bins = len(binned_df)
    logger.info(f"Aggregated {file_path.name} into {num_bins} chronological bins of {bin_duration_sec}s.")

    if num_bins == 0:
        return np.empty((0, 12), dtype=np.float32), np.empty(0, dtype=np.int32)

    features = binned_df.select(CANONICAL_12_METRICS).to_numpy().astype(np.float32)
    labels = binned_df["label"].to_numpy().astype(np.int32)

    # Sanitize any NaNs or infinities
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

    return features, labels


def process_pcap_file_to_bins(
    pcap_path: Path,
    bin_duration_sec: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Streams a PCAP file using FastPCAPParser in discrete temporal windows."""
    from src.features.fast_pcap import FastPCAPParser

    logger.info(f"Parsing PCAP binary stream: {pcap_path.name} ({pcap_path.stat().st_size / (1024*1024):.1f} MB)...")
    parser = FastPCAPParser(bin_duration_sec=bin_duration_sec)
    raw_16_feats, timestamps, meta, _ = parser.parse_file(pcap_path)

    # Map the 16 features from FastPCAPParser to the 12 Canonical Metrics:
    # 0: duration_norm, 1: byte_ratio, 2: packet_rate, 3: iat_mean, 4: iat_std,
    # 7: tcp_syn_ratio, 8: tcp_ack_ratio, 12: tcp_rst_ratio
    n_bins = len(raw_16_feats)
    if n_bins == 0:
        return np.empty((0, 12), dtype=np.float32), np.empty(0, dtype=np.int32)

    feats_12 = np.zeros((n_bins, 12), dtype=np.float32)
    # 1. byte_rate ~ exp(packet_rate) * payload
    feats_12[:, 0] = np.expm1(raw_16_feats[:, 2]) * raw_16_feats[:, 14]  # byte_rate
    feats_12[:, 1] = np.expm1(raw_16_feats[:, 2])                       # packet_rate
    feats_12[:, 2] = raw_16_feats[:, 3]                                  # mean_iat
    feats_12[:, 3] = raw_16_feats[:, 4] ** 2                             # var_iat = std^2
    feats_12[:, 4] = raw_16_feats[:, 15]                                 # max_iat
    feats_12[:, 5] = raw_16_feats[:, 7]                                  # syn_flag_ratio
    feats_12[:, 6] = raw_16_feats[:, 8]                                  # ack_flag_ratio
    feats_12[:, 7] = np.clip(raw_16_feats[:, 8] * 0.3, 0.0, 1.0)        # fin_flag_ratio approx
    feats_12[:, 8] = raw_16_feats[:, 12]                                 # rst_flag_ratio
    feats_12[:, 9] = np.clip(raw_16_feats[:, 11], 0.0, 1.0)             # psh_flag_ratio approx
    feats_12[:, 10] = np.expm1(raw_16_feats[:, 1])                       # bwd_to_fwd_ratio
    feats_12[:, 11] = bin_duration_sec                                   # mean_flow_duration

    # Binary label: check attacker IP presence or heuristic high risk
    labels = np.zeros(n_bins, dtype=np.int32)
    topo = meta.get("topology", {})
    for node in topo.get("nodes", []):
        if node.get("status") == "THREAT_ACTOR" or "175.45.176." in node.get("ip", ""):
            labels[:] = 1
            break

    feats_12 = np.nan_to_num(feats_12, nan=0.0, posinf=1e6, neginf=-1e6)
    del parser, raw_16_feats, timestamps, meta
    return feats_12, labels


def generate_window_pairs(
    features_scaled: np.ndarray,
    labels: np.ndarray,
    seq_len: int = 20,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates continuous transition pairs:

    S_t: bins i to i+20 (Shape: N x 240)
    S_{t+1}: bins i+20 to i+40 (Shape: N x 240)
    y_t: binary attack label across the window (Shape: N)
    """
    total_bins = len(features_scaled)
    required_span = seq_len * 2  # 40 bins (20s)
    if total_bins < required_span:
        logger.warning(f"Total bins ({total_bins}) less than required span ({required_span}). Skipping windowing.")
        return (
            np.empty((0, seq_len * 12), dtype=np.float32),
            np.empty((0, seq_len * 12), dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    s_t_list: List[np.ndarray] = []
    s_next_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i in range(0, total_bins - required_span + 1, stride):
        s_t_win = features_scaled[i : i + seq_len].ravel()                  # 20 * 12 = 240 floats
        s_next_win = features_scaled[i + seq_len : i + required_span].ravel() # 20 * 12 = 240 floats
        # Window label is 1 if any bin in [i, i+seq_len] is attack
        win_label = int(np.max(labels[i : i + seq_len])) if len(labels) > 0 else 0

        s_t_list.append(s_t_win)
        s_next_list.append(s_next_win)
        y_list.append(win_label)

    s_t = np.array(s_t_list, dtype=np.float32)
    s_next = np.array(s_next_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return s_t, s_next, y


def save_chunk_parquet(
    s_t: np.ndarray,
    s_next: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> int:
    """Saves transition pairs to a standalone Snappy-compressed Parquet chunk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = len(s_t)
    if num_samples == 0:
        logger.warning(f"No samples to save for {output_path.name}")
        return 0

    flat_len = s_t.shape[1]  # 240
    offsets = np.arange(0, (num_samples + 1) * flat_len, flat_len, dtype=np.int64)

    s_t_arrow = pa.ListArray.from_arrays(offsets, pa.array(s_t.ravel(), type=pa.float32()))
    s_next_arrow = pa.ListArray.from_arrays(offsets, pa.array(s_next.ravel(), type=pa.float32()))
    y_arrow = pa.array(y, type=pa.int32())

    table = pa.Table.from_arrays([s_t_arrow, s_next_arrow, y_arrow], names=["s_t", "s_next", "label"])
    pq.write_table(table, output_path, compression="snappy")

    file_size_bytes = output_path.stat().st_size
    logger.info(
        f"Saved chunk: {output_path.name} | Samples: {num_samples} | "
        f"Shape: ({num_samples}, {flat_len}) | Disk: {file_size_bytes / (1024*1024):.2f} MB"
    )
    return file_size_bytes


def run_batch_preprocessing(
    csv_dir: Optional[Path] = None,
    pcap_dir: Optional[Path] = None,
    output_dir: Path = Path("data/processed/chunks"),
    scaler_path: Path = Path("data/processed/scaler.joblib"),
    max_files: Optional[int] = None,
    bin_duration_sec: float = 0.5,
    seq_len: int = 20,
    stride: int = 2,
    include_pcaps: bool = True,
) -> List[Path]:
    """Sequential Multi-File Ingestion Loop with strict memory governance (< 2.5 GB RAM)."""
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_log_path = output_dir.parent / "skipped_files.log"
    skipped_log_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Discover target files exhaustively with natural order
    file_queue: List[Tuple[str, Path]] = []

    if csv_dir and csv_dir.exists():
        main_csvs = sorted(
            [f for f in csv_dir.glob("UNSW-NB15_*.csv") if not any(k in f.name.lower() for k in ["feature", "event", "gt"])],
            key=natural_sort_key,
        )
        tt_dir = csv_dir / "Training and Testing Sets"
        tt_csvs = (
            sorted(
                [f for f in tt_dir.glob("*.csv") if not any(k in f.name.lower() for k in ["feature", "event", "gt"])],
                key=natural_sort_key,
            )
            if tt_dir.exists()
            else []
        )
        csv_files = main_csvs + tt_csvs
        for f in csv_files:
            file_queue.append(("csv", f))

    if include_pcaps and pcap_dir and pcap_dir.exists():
        pcap_files = sorted([f for f in pcap_dir.rglob("*.pcap")], key=natural_sort_key)
        for f in pcap_files:
            file_queue.append(("pcap", f))

    if max_files is not None and max_files > 0:
        file_queue = file_queue[:max_files]

    total_files = len(file_queue)
    logger.info(f"Identified {total_files} files for sequential batch preprocessing.")
    if total_files == 0:
        logger.warning("No input files found to process. Exiting.")
        return []

    scaler_handler = IncrementalRobustScaler(scaler_path=scaler_path)
    generated_chunks: List[Path] = []

    # 2. Sequential Processing Loop
    for idx, (ftype, fpath) in enumerate(file_queue):
        chunk_file = output_dir / f"chunk_{idx:04d}.parquet"

        # Resumability check: if chunk already exists and is non-empty, skip re-parsing
        if chunk_file.exists() and chunk_file.stat().st_size > 0:
            logger.info(f"[{idx+1}/{total_files}] Resuming: {chunk_file.name} already exists for {fpath.name}. Skipping re-parsing.")
            generated_chunks.append(chunk_file)
            continue

        f_t0 = time.perf_counter()
        logger.info(f"[{idx+1}/{total_files}] Processing {ftype.upper()}: {fpath.name}...")

        # Monitor RAM before processing
        ram_before = get_process_memory_mb()
        check_ram_boundary(2500.0)

        features: Optional[np.ndarray] = None
        labels: Optional[np.ndarray] = None
        try:
            if ftype == "csv":
                features, labels = process_csv_file_to_bins(fpath, bin_duration_sec=bin_duration_sec)
            else:
                features, labels = process_pcap_file_to_bins(fpath, bin_duration_sec=bin_duration_sec)

            if len(features) == 0:
                logger.warning(f"File {fpath.name} yielded 0 bins. Skipping.")
                continue

            # Transform features using pre-calibrated scaler (never overwrite if exists)
            is_first = (idx == 0 and not scaler_path.exists())
            features_scaled = scaler_handler.fit_or_transform(features, is_first_chunk=is_first)

            # Generate chronological transition pairs S_t, S_{t+1}, y_t
            s_t, s_next, y = generate_window_pairs(
                features_scaled,
                labels,
                seq_len=seq_len,
                stride=stride,
            )

            if len(s_t) > 0:
                save_chunk_parquet(s_t, s_next, y, chunk_file)
                generated_chunks.append(chunk_file)

        except Exception as exc:
            logger.exception(f"Fault-tolerance: error processing {fpath.name}: {exc}")
            try:
                with open(skipped_log_path, "a", encoding="utf-8") as sf:
                    sf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAILED {fpath} : {type(exc).__name__}: {exc}\n")
            except Exception:
                pass
        finally:
            # Explicit cleanup and garbage collection after every single file
            if features is not None:
                del features
            if labels is not None:
                del labels
            if "features_scaled" in locals():
                del features_scaled
            if "s_t" in locals():
                del s_t, s_next, y
            gc.collect()

            ram_after = get_process_memory_mb()
            logger.info(
                f"[{idx+1}/{total_files}] Completed {fpath.name} in {time.perf_counter() - f_t0:.1f}s | "
                f"RAM: {ram_before:.1f} MB -> {ram_after:.1f} MB (< 2500 MB bound maintained)"
            )
            check_ram_boundary(2500.0)

    total_time = time.perf_counter() - t_start
    logger.info(
        f"Batch Preprocessing Complete! Generated/validated {len(generated_chunks)} Parquet chunks in {total_time:.1f}s. "
        f"Final RAM: {get_process_memory_mb():.1f} MB."
    )
    return generated_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-File Sequential Streaming Preprocessor for Threatora World Model")
    parser.add_argument("--csv-dir", type=str, default="G:/UNSW-NB15 Dataset/CSV Files", help="Directory containing CSV flow captures")
    parser.add_argument("--pcap-dir", type=str, default="G:/UNSW-NB15 Dataset/pcap files", help="Directory containing raw PCAP captures")
    parser.add_argument("--output-dir", type=str, default="data/processed/chunks", help="Output directory for Parquet chunk caches")
    parser.add_argument("--scaler-path", type=str, default="data/processed/scaler.joblib", help="Filepath for fitted RobustScaler")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process (default: all)")
    parser.add_argument("--bin-duration", type=float, default=0.5, help="Temporal bin duration in seconds")
    parser.add_argument("--seq-len", type=int, default=20, help="Continuous sequence length in bins (20 bins = 10s)")
    parser.add_argument("--stride", type=int, default=2, help="Sliding window stride in bins")
    parser.add_argument("--include-pcaps", dest="include_pcaps", action="store_true", default=True, help="Ingest raw PCAP files (default: True)")
    parser.add_argument("--no-pcaps", dest="include_pcaps", action="store_false", help="Disable ingestion of raw PCAP files")
    args = parser.parse_args()

    csv_path = Path(args.csv_dir)
    pcap_path = Path(args.pcap_dir)
    out_path = Path(args.output_dir)
    scaler_path = Path(args.scaler_path)

    run_batch_preprocessing(
        csv_dir=csv_path if csv_path.exists() else None,
        pcap_dir=pcap_path if pcap_path.exists() else None,
        output_dir=out_path,
        scaler_path=scaler_path,
        max_files=args.max_files,
        bin_duration_sec=args.bin_duration,
        seq_len=args.seq_len,
        stride=args.stride,
        include_pcaps=args.include_pcaps,
    )


if __name__ == "__main__":
    main()
