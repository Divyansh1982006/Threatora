"""High-Performance Ingestion Pipeline for 5-6 GB PCAPs + All Argus Flow Logs.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Executes:
  1. Targeted 5-6 GB representative PCAP ingestion across Day 1 (5, 7, 9.pcap) and Day 2 (2.pcap)
     via FastPCAPParser binary streaming in 8MB chunks.
  2. Complete Argus flow log ingestion across all 45 CSVs for both capture dates.
  3. Zero-leakage sanitization (scrubs all IP addresses, ports, timestamps, flow IDs; computes is_privileged_port).
  4. 16-slot canonical continuous feature mapping.
  5. Feature scaling via existing pre-fitted RobustScaler (data/processed/scaler.joblib) without refitting.
  6. Rolling paired state transition window construction (S_t, S_{t+1}, Y_t in R^5, M_t in {0..5}).
  7. Anti-forgetting Experience Replay Buffer: samples 30% stratified windows from
     data/processed/train_windows_full.parquet and combines them into data/processed/train_windows_finetune.parquet.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

from .adapters.dataset_adapter import CanonicalFeatureExtractor, CANONICAL_SLOTS
from .features.fast_pcap import FastPCAPParser

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("IngestArgusPCAP")

ARGUS_COLUMN_NAMES: List[str] = [
    "stime", "proto", "srcip", "sport", "dstip", "dsport", "totpkts", "state", "ltime",
    "spkts", "dpkts", "sbytes", "dbytes", "dur", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sjit", "djit", "sintpkt", "dintpkt", "swin", "stcpb", "dtcpb",
    "dwin", "tcprtt", "synack", "ackdat", "smeansz", "dmeansz"
]


def resolve_representative_pcap_paths(repo_root: Path) -> List[Path]:
    """Resolves ~5.03 GB representative raw PCAP slices across both capture dates."""
    day1_dir = repo_root / "data" / "UNSW-NB15 Dataset" / "pcap files" / "22-1-15"
    day2_dir = repo_root / "data" / "UNSW-NB15 Dataset" / "pcap files" / "17-2-15"

    selected: List[Path] = []
    # Day 1: 5.pcap (~1.0 GB), 7.pcap (~1.03 GB), 9.pcap (~1.0 GB) -> ~3.03 GB
    for name in ["5.pcap", "7.pcap", "9.pcap"]:
        p = day1_dir / name
        if p.exists():
            selected.append(p)
        else:
            logger.warning(f"Day 1 PCAP slice not found: {p}")

    # Day 2: 2.pcap (~2.0 GB)
    for name in ["2.pcap"]:
        p = day2_dir / name
        if p.exists():
            selected.append(p)
        else:
            logger.warning(f"Day 2 PCAP slice not found: {p}")

    total_bytes = sum(p.stat().st_size for p in selected)
    logger.info(
        f"Selected {len(selected)} representative PCAP slices ({total_bytes / (1024**3):.2f} GB total volume): "
        f"{[p.name for p in selected]}"
    )
    return selected


def resolve_all_argus_csv_paths(repo_root: Path) -> Tuple[List[Path], List[Path]]:
    """Resolves all available Argus flow CSVs across both capture dates."""
    day1_argus_dir = repo_root / "data" / "UNSW-NB15 Dataset" / "Argus Files" / "Argus Files 22-1-2015" / "CSV of Argus"
    day2_argus_dir = repo_root / "data" / "UNSW-NB15 Dataset" / "Argus Files" / "Argus Files 17-2-2015" / "CSV of Argus"

    day1_files = sorted(day1_argus_dir.glob("*.csv")) if day1_argus_dir.exists() else []
    day2_files = sorted(day2_argus_dir.glob("*.csv")) if day2_argus_dir.exists() else []

    logger.info(f"Resolved {len(day1_files)} Argus CSVs for Day 1 and {len(day2_files)} Argus CSVs for Day 2.")
    return day1_files, day2_files


def parse_argus_csv_file(
    file_path: Path,
    extractor: CanonicalFeatureExtractor,
) -> Optional[pl.DataFrame]:
    """Parses an individual Argus CSV file into canonical slots with zero data leakage."""
    schema_overrides = {
        "column_1": pl.Float64,
        "column_2": pl.String,
        "column_3": pl.String,
        "column_4": pl.String,
        "column_5": pl.String,
        "column_6": pl.String,
        "column_7": pl.Float64,
        "column_8": pl.String,
        "column_9": pl.Float64,
        "column_10": pl.Float64,
        "column_11": pl.Float64,
        "column_12": pl.Float64,
        "column_13": pl.Float64,
        "column_14": pl.Float64,
        "column_15": pl.Float64,
        "column_16": pl.Float64,
    }

    try:
        df = pl.read_csv(
            file_path,
            has_header=False,
            schema_overrides=schema_overrides,
            ignore_errors=True,
        )
        if len(df) == 0:
            return None

        # Rename columns to standard UNSW-NB15 names
        num_cols = len(df.columns)
        rename_dict = {f"column_{i+1}": name for i, name in enumerate(ARGUS_COLUMN_NAMES[:num_cols])}
        df = df.rename(rename_dict)

        # Attacker IP subnet in UNSW-NB15: 175.45.176.0/24
        # Label = 1 if source or destination IP starts with 175.45.176.
        src_is_atk = pl.col("srcip").cast(pl.String).str.starts_with("175.45.176.")
        dst_is_atk = pl.col("dstip").cast(pl.String).str.starts_with("175.45.176.")
        is_attack_expr = (src_is_atk | dst_is_atk)

        df = df.with_columns([
            pl.when(is_attack_expr).then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int32).alias("label"),
            pl.when(is_attack_expr).then(pl.lit(2)).otherwise(pl.lit(0)).cast(pl.Int32).alias("mitre_stage"),
        ])

        # Adapt to canonical schema and strip shortcut identifiers
        sanitized_lf, _ = extractor.sanitize_and_extract(df.lazy(), dataset="unsw_nb15")
        return sanitized_lf.collect()

    except Exception as exc:
        logger.warning(f"Failed to parse Argus file {file_path.name}: {exc}")
        return None


def aggregate_argus_into_temporal_bins(
    argus_files: List[Path],
    extractor: CanonicalFeatureExtractor,
    bin_duration_sec: float = 0.5,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    """Aggregates flows from multiple Argus CSVs into unified 0.5s temporal bins."""
    logger.info(f"Ingesting and aggregating {len(argus_files)} Argus CSVs into {bin_duration_sec}s bins...")
    t0 = time.perf_counter()

    bin_agg: Dict[int, List[np.ndarray]] = {}
    bin_labels: Dict[int, int] = {}
    bin_mitre: Dict[int, int] = {}
    total_flows = 0

    for i, f_path in enumerate(argus_files):
        df = parse_argus_csv_file(f_path, extractor)
        if df is None or len(df) == 0:
            continue

        total_flows += len(df)
        b_ids = (df["timestamp"] / bin_duration_sec).floor().cast(pl.Int64).to_numpy()
        feats = df.select(CANONICAL_SLOTS).to_numpy().astype(np.float32)
        lbls = df["label"].to_numpy().astype(np.int32)
        mitres = df["mitre_stage"].to_numpy().astype(np.int32)

        for j in range(len(b_ids)):
            b_id = int(b_ids[j])
            if b_id not in bin_agg:
                bin_agg[b_id] = []
                bin_labels[b_id] = 0
                bin_mitre[b_id] = 0

            bin_agg[b_id].append(feats[j])
            if lbls[j] == 1:
                bin_labels[b_id] = 1
            if mitres[j] > bin_mitre[b_id]:
                bin_mitre[b_id] = int(mitres[j])

    # Compute mean continuous vector per bin
    flow_features: Dict[int, np.ndarray] = {}
    for b_id, feat_list in bin_agg.items():
        flow_features[b_id] = np.mean(feat_list, axis=0).astype(np.float32)

    elapsed = time.perf_counter() - t0
    num_attacks = sum(bin_labels.values())
    logger.info(
        f"Argus flow aggregation complete in {elapsed:.2f}s: {total_flows:,} flows -> {len(flow_features)} bins "
        f"(Attacks: {num_attacks}, Benign: {len(flow_features) - num_attacks})"
    )
    return flow_features, bin_labels, bin_mitre


def fuse_argus_and_pcap_modalities(
    flow_features: Dict[int, np.ndarray],
    flow_labels: Dict[int, int],
    flow_mitre: Dict[int, int],
    packet_features: Dict[int, np.ndarray],
    bin_duration_sec: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synchronizes flow and packet modalities into uniform continuous epoch timeline."""
    all_bin_ids = sorted(set(flow_features.keys()).union(packet_features.keys()))
    n_bins = len(all_bin_ids)
    if n_bins == 0:
        return np.empty((0, 16), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.float64)

    features = np.zeros((n_bins, 16), dtype=np.float32)
    labels = np.zeros(n_bins, dtype=np.int32)
    mitre = np.zeros(n_bins, dtype=np.int32)
    timestamps = np.zeros(n_bins, dtype=np.float64)

    num_fused = 0
    num_flow_only = 0
    num_pcap_only = 0

    for i, b_id in enumerate(all_bin_ids):
        timestamps[i] = b_id * bin_duration_sec
        has_flow = b_id in flow_features
        has_packet = b_id in packet_features

        if has_flow and has_packet:
            features[i] = 0.5 * flow_features[b_id] + 0.5 * packet_features[b_id]
            labels[i] = flow_labels[b_id]
            mitre[i] = flow_mitre.get(b_id, 0)
            num_fused += 1
        elif has_flow:
            features[i] = flow_features[b_id]
            labels[i] = flow_labels[b_id]
            mitre[i] = flow_mitre.get(b_id, 0)
            num_flow_only += 1
        else:
            features[i] = packet_features[b_id]
            labels[i] = 0
            mitre[i] = 0
            num_pcap_only += 1

    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=0.0)
    logger.info(
        f"Fusion complete: {n_bins} total bins "
        f"(Fused: {num_fused}, Flow-Only: {num_flow_only}, PCAP-Only: {num_pcap_only})"
    )
    return features, labels, mitre, timestamps


def generate_paired_windows(
    scaled_features: np.ndarray,
    labels: np.ndarray,
    mitre_stages: np.ndarray,
    sequence_length: int = 20,
    stride: int = 1,
    horizon_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates paired rolling window tensors (S_t, S_{t+1}, y, y_horizon, mitre_stage)."""
    w_len = sequence_length
    total_span = 2 * w_len
    num_bins, num_feats = scaled_features.shape

    if num_bins < total_span:
        return (
            np.empty((0, w_len, num_feats), dtype=np.float32),
            np.empty((0, w_len, num_feats), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, horizon_k), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    num_windows = (num_bins - total_span) // stride + 1
    sw = np.lib.stride_tricks.sliding_window_view(scaled_features, window_shape=(w_len, num_feats))
    sw = sw[:, 0, :, :]

    indices = np.arange(0, num_windows * stride, stride)
    s_t_windows = sw[indices].astype(np.float32)
    s_next_windows = sw[indices + w_len].astype(np.float32)

    sw_labels = np.lib.stride_tricks.sliding_window_view(labels, window_shape=w_len)
    y_windows = np.any(sw_labels[indices] == 1, axis=1).astype(np.int32)

    y_horizon = np.zeros((num_windows, horizon_k), dtype=np.float32)
    for k in range(horizon_k):
        target_idx = np.clip(indices + k, 0, len(sw_labels) - 1)
        y_horizon[:, k] = np.any(sw_labels[target_idx] == 1, axis=1).astype(np.float32)

    sw_mitre = np.lib.stride_tricks.sliding_window_view(mitre_stages, window_shape=w_len)
    mitre_windows = np.max(sw_mitre[indices], axis=1).astype(np.int32)

    return s_t_windows, s_next_windows, y_windows, y_horizon, mitre_windows


def build_experience_replay_buffer(
    train_full_path: Path,
    sample_ratio: float = 0.30,
    seed: int = 42,
) -> pl.DataFrame:
    """Samples a stratified Experience Replay Buffer from previously learned training dataset."""
    logger.info(f"Loading previous training dataset from {train_full_path} for Experience Replay Buffer...")
    df = pl.read_parquet(train_full_path)
    total_prev = len(df)

    # Stratified sampling on label
    df_benign = df.filter(pl.col("label") == 0).sample(fraction=sample_ratio, seed=seed)
    df_attack = df.filter(pl.col("label") == 1).sample(fraction=sample_ratio, seed=seed)
    replay_df = pl.concat([df_benign, df_attack])

    logger.info(
        f"Experience Replay Buffer created: {len(replay_df):,} windows sampled ({sample_ratio*100:.0f}% of {total_prev:,}) "
        f"(Attacks: {(replay_df['label'] == 1).sum():,}, Benign: {(replay_df['label'] == 0).sum():,})"
    )
    return replay_df


def save_windows_to_parquet(
    s_t: np.ndarray,
    s_next: np.ndarray,
    labels: np.ndarray,
    y_horizon: np.ndarray,
    mitre_stages: np.ndarray,
    output_path: Path,
) -> pl.DataFrame:
    """Converts numpy window arrays to a structured Polars DataFrame."""
    num_windows = len(s_t)
    elements_per_window = int(np.prod(s_t.shape[1:]))

    offsets_states = np.arange(0, (num_windows + 1) * elements_per_window, elements_per_window, dtype=np.int64)
    s_t_arrow = pa.ListArray.from_arrays(offsets_states, pa.array(s_t.ravel(), type=pa.float32()))
    s_next_arrow = pa.ListArray.from_arrays(offsets_states, pa.array(s_next.ravel(), type=pa.float32()))

    offsets_horizon = np.arange(0, (num_windows + 1) * 5, 5, dtype=np.int64)
    y_horizon_arrow = pa.ListArray.from_arrays(offsets_horizon, pa.array(y_horizon.ravel(), type=pa.float32()))

    label_arrow = pa.array(labels, type=pa.int32())
    mitre_arrow = pa.array(mitre_stages, type=pa.int32())

    table = pa.Table.from_arrays(
        [s_t_arrow, s_next_arrow, label_arrow, y_horizon_arrow, mitre_arrow],
        names=["s_t", "s_next", "label", "y_horizon", "mitre_stage"]
    )
    return pl.from_arrow(table)


def run_targeted_ingestion_and_replay_pipeline(
    repo_root: Optional[Path] = None,
    bin_duration_sec: float = 0.5,
    sequence_length: int = 20,
    stride: int = 1,
    replay_ratio: float = 0.30,
) -> Dict[str, Any]:
    """End-to-end execution of targeted PCAP + Argus ingestion and Experience Replay integration."""
    t_start = time.time()
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = processed_dir / "scaler.joblib"
    train_full_path = processed_dir / "train_windows_full.parquet"
    finetune_out_path = processed_dir / "train_windows_finetune.parquet"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Existing RobustScaler not found at: {scaler_path}")
    if not train_full_path.exists():
        raise FileNotFoundError(f"Existing training parquet not found at: {train_full_path}")

    logger.info("=" * 75)
    logger.info("Targeted 5-6 GB PCAP + Argus Flow Logs Ingestion with Experience Replay Buffer")
    logger.info("=" * 75)

    # 1. Load existing RobustScaler strictly for transformation (DO NOT refit)
    logger.info(f"Loading existing pre-fitted RobustScaler from {scaler_path} (refitting disabled)...")
    scaler: RobustScaler = joblib.load(scaler_path)

    extractor = CanonicalFeatureExtractor()

    # 2. Ingest representative ~5.03 GB PCAPs (Day 1: 5, 7, 9.pcap; Day 2: 2.pcap)
    pcap_paths = resolve_representative_pcap_paths(repo_root)
    pcap_parser = FastPCAPParser(bin_duration_sec=bin_duration_sec, chunk_size=8 * 1024 * 1024)
    logger.info(f"Streaming {len(pcap_paths)} PCAP slices via FastPCAPParser in 8MB chunks...")
    packet_bins, pcap_summary = pcap_parser.stream_slices_to_epoch_bins(
        pcap_paths=pcap_paths,
        max_packets_per_slice=2_000_000,
        chunk_size=8 * 1024 * 1024,
    )
    logger.info(
        f"PCAP Ingestion Complete: {pcap_summary['total_packets']:,} packets "
        f"({pcap_summary['total_bytes_mb']:.1f} MB, {pcap_summary['throughput_mb_s']:.1f} MB/s) -> {len(packet_bins)} bins."
    )

    # 3. Ingest all 45 Argus flow CSV files for Day 1 and Day 2
    day1_csvs, day2_csvs = resolve_all_argus_csv_paths(repo_root)
    all_argus_csvs = day1_csvs + day2_csvs
    flow_bins, flow_labels, flow_mitre = aggregate_argus_into_temporal_bins(
        argus_files=all_argus_csvs,
        extractor=extractor,
        bin_duration_sec=bin_duration_sec,
    )

    # 4. Modality fusion along continuous Unix epoch timeline
    features_full, labels_full, mitre_full, timestamps_full = fuse_argus_and_pcap_modalities(
        flow_features=flow_bins,
        flow_labels=flow_labels,
        flow_mitre=flow_mitre,
        packet_features=packet_bins,
        bin_duration_sec=bin_duration_sec,
    )

    # 5. Session-bounded paired window generation using pre-fitted scaler
    day1_mask = timestamps_full < 1.423e9
    day2_mask = timestamps_full >= 1.423e9

    new_windows_list: List[pl.DataFrame] = []

    for day_name, mask in [("Day 1 (22-01-2015)", day1_mask), ("Day 2 (17-02-2015)", day2_mask)]:
        if not np.any(mask):
            continue
        sub_feats = features_full[mask]
        sub_lbls = labels_full[mask]
        sub_mitre = mitre_full[mask]

        # Transform features using pre-fitted scaler
        sub_scaled = scaler.transform(sub_feats).astype(np.float32)

        s_t, s_nxt, y, y_h, m = generate_paired_windows(
            scaled_features=sub_scaled,
            labels=sub_lbls,
            mitre_stages=sub_mitre,
            sequence_length=sequence_length,
            stride=stride,
            horizon_k=5,
        )
        if len(s_t) > 0:
            df_day = save_windows_to_parquet(s_t, s_nxt, y, y_h, m, output_path=finetune_out_path)
            new_windows_list.append(df_day)
            logger.info(f"Generated {len(s_t):,} new windows for {day_name} (Attacks: {np.sum(y):,})")

    if new_windows_list:
        new_windows_df = pl.concat(new_windows_list)
    else:
        new_windows_df = pl.DataFrame()

    logger.info(f"Total newly generated Argus+PCAP windows: {len(new_windows_df):,}")

    # 6. Experience Replay Buffer Integration
    replay_df = build_experience_replay_buffer(
        train_full_path=train_full_path,
        sample_ratio=replay_ratio,
        seed=42,
    )

    # 7. Concatenate and shuffle/interleave combined fine-tuning dataset
    if len(new_windows_df) > 0:
        combined_df = pl.concat([new_windows_df, replay_df])
    else:
        combined_df = replay_df

    combined_df = combined_df.sample(fraction=1.0, shuffle=True, seed=42)

    # Save to data/processed/train_windows_finetune.parquet
    combined_df.write_parquet(finetune_out_path, compression="snappy")
    file_size_mb = finetune_out_path.stat().st_size / (1024 * 1024)

    elapsed = time.time() - t_start
    num_attacks = int((combined_df["label"] == 1).sum())
    num_benign = int((combined_df["label"] == 0).sum())

    logger.info("=" * 75)
    logger.info("EXPERIENCE REPLAY & TARGETED INGESTION COMPLETE")
    logger.info(f"Total Combined Windows for Fine-Tuning: {len(combined_df):,}")
    logger.info(f"  - Newly Ingested Argus+PCAP Windows: {len(new_windows_df):,}")
    logger.info(f"  - Experience Replay Buffer Windows:   {len(replay_df):,}")
    logger.info(f"  - Class Balance: Attacks: {num_attacks:,} ({num_attacks/len(combined_df)*100:.1f}%), Benign: {num_benign:,} ({num_benign/len(combined_df)*100:.1f}%)")
    logger.info(f"Saved Fine-Tuning Parquet to: {finetune_out_path} ({file_size_mb:.2f} MB)")
    logger.info(f"Total Execution Time: {elapsed:.2f}s")
    logger.info("=" * 75)

    return {
        "finetune_parquet": str(finetune_out_path),
        "total_windows": len(combined_df),
        "new_windows": len(new_windows_df),
        "replay_windows": len(replay_df),
        "attacks": num_attacks,
        "benign": num_benign,
        "size_mb": file_size_mb,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest 5-6 GB PCAPs + Argus Flow Logs with Experience Replay")
    parser.add_argument("--bin-duration", type=float, default=0.5, help="Bin duration (seconds)")
    parser.add_argument("--replay-ratio", type=float, default=0.30, help="Replay buffer sampling ratio (default: 0.30)")
    args = parser.parse_args()

    run_targeted_ingestion_and_replay_pipeline(
        bin_duration_sec=args.bin_duration,
        replay_ratio=args.replay_ratio,
    )


if __name__ == "__main__":
    main()
