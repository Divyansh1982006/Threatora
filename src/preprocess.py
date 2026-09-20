"""High-Performance, RAM-Safe Preprocessing & Temporal Windowing Pipeline.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
First step of the predictive "Network World Model":
  1. Ingests UNSW-NB15 flow data (22-1-2015 capture slice) with Polars lazy evaluation (RAM-safe).
  2. Eliminates data leakage by stripping shortcut identifiers ('srcip', 'dstip', 'sport', 'dsport',
     'flow_id', 'timestamp') and converting destination ports to `is_privileged_port`.
  3. Computes 16 canonical flow metrics aggregated across 0.5-second temporal bins.
  4. Chronologically partitions bins (first 80% Train, last 20% Val/Test).
  5. Fits RobustScaler strictly on the training partition, saving to data/processed/scaler.joblib.
  6. Generates chronologically paired state transitions (S_t, S_{t+1}) of shape (20, 16) -> 320 floats,
     multi-horizon ground-truth label vectors Y_t in R^5, and MITRE ATT&CK stage targets (0..5).
  7. Outputs compressed Parquet files (train_windows.parquet, val_windows.parquet) with snappy compression.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

from .adapters import CanonicalFeatureExtractor, CANONICAL_SLOTS

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PreprocessWorldModel")

# Standard 49 UNSW-NB15 column names (from NUSW-NB15_features.csv)
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

# Shortcut identifier columns that must be strictly stripped to prevent data leakage
SHORTCUT_IDENTIFIER_COLS = [
    "srcip", "dstip", "sport", "dsport", "flow_id", "timestamp", "id"
]

# 16 canonical flow metrics for the World Model state vector
TARGET_FEATURE_NAMES: List[str] = list(CANONICAL_SLOTS)


def resolve_default_raw_path() -> Path:
    """Locates the default UNSW-NB15 22-1-2015 capture slice CSV in the project."""
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "data" / "UNSW-NB15 Dataset" / "CSV Files" / "UNSW-NB15_1.csv"
    if candidate.exists():
        return candidate
    alt_candidates = list((repo_root / "data").glob("**/UNSW-NB15_1.csv"))
    if alt_candidates:
        return alt_candidates[0]
    return candidate


def parse_port_value(val: object) -> int:
    """Safely parses port values which may be int, float, string, or hex (e.g. '0xc0a8')."""
    if val is None:
        return 0
    val_str = str(val).strip()
    if not val_str or val_str == "-":
        return 0
    if val_str.startswith(("0x", "0X")):
        try:
            return int(val_str, 16)
        except ValueError:
            return 0
    try:
        return int(float(val_str))
    except ValueError:
        return 0


def map_attack_cat_to_mitre(cat_str: object) -> int:
    """Maps UNSW-NB15 attack_cat strings to 6-stage MITRE ATT&CK taxonomy.

    0: Benign (Normal)
    1: Reconnaissance (Fuzzers, Reconnaissance, Analysis)
    2: Initial Access (Exploits, Shellcode)
    3: Lateral Movement (Generic)
    4: Command & Control (Backdoor, Backdoors)
    5: Exfiltration / Impact (DoS, Worms)
    """
    if cat_str is None:
        return 0
    c = str(cat_str).strip().lower()
    if c in ("normal", "0", "benign", "", "none", "-"):
        return 0
    if any(k in c for k in ("fuzz", "recon", "analy")):
        return 1
    if any(k in c for k in ("exploit", "shell")):
        return 2
    if "generic" in c:
        return 3
    if "backdoor" in c:
        return 4
    if any(k in c for k in ("dos", "worm")):
        return 5
    return 2


class UNSWPreprocessor:
    """RAM-safe streaming preprocessor and paired temporal window generator."""

    def __init__(
        self,
        bin_duration_sec: float = 0.5,
        sequence_length: int = 20,
        stride: int = 1,
        train_ratio: float = 0.8,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.bin_duration_sec = bin_duration_sec
        self.sequence_length = sequence_length
        self.stride = stride
        self.train_ratio = train_ratio
        self.features = list(TARGET_FEATURE_NAMES)
        self.feature_dim = len(self.features)
        assert self.feature_dim == 16, f"Expected exactly 16 features, got {self.feature_dim}"

        repo_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else repo_root / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path = self.output_dir / "scaler.joblib"

        self.extractor = CanonicalFeatureExtractor(
            dataset="unsw_nb15",
            bin_duration_sec=self.bin_duration_sec,
            sequence_length=self.sequence_length,
            stride=self.stride,
            scaler_path=self.scaler_path,
        )

    def scan_unsw_csv(self, file_path: Union[str, Path]) -> pl.LazyFrame:
        """Lazily scans a UNSW-NB15 CSV file using Polars, auto-detecting headers and schemas."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {file_path}")

        logger.info(f"Scanning raw CSV lazily: {file_path} ({file_path.stat().st_size / (1024*1024):.2f} MB)")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
        has_header = False
        if any(h in first_line.lower() for h in ["sport", "dsport", "dur", "sbytes", "sttl"]):
            has_header = True

        schema_overrides = {
            "sport": pl.String,
            "dsport": pl.String,
            "column_2": pl.String,
            "column_4": pl.String,
        }

        if has_header:
            lf = pl.scan_csv(
                file_path,
                has_header=True,
                schema_overrides=schema_overrides,
                ignore_errors=True,
            )
            cols = lf.collect_schema().names()
            rename_map = {c: c.strip().lower() for c in cols}
            lf = lf.rename(rename_map)
        else:
            lf = pl.scan_csv(
                file_path,
                has_header=False,
                schema_overrides=schema_overrides,
                ignore_errors=True,
            )
            col_count = len(lf.collect_schema().names())
            col_names = [name.strip().lower() for name in UNSW_NB15_COLUMNS[:col_count]]
            lf = lf.rename({f"column_{i+1}": name for i, name in enumerate(col_names)})

        return lf

    def clean_and_sanitize(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Strips shortcut identifier columns and projects to canonical flow features with zero data leakage.

        Returns sanitized LazyFrame with canonical slots + timestamp + label + mitre_stage
        """
        sanitized_lf, _ = self.extractor.sanitize_and_extract(lf, dataset="unsw_nb15")
        if "mitre_stage" not in sanitized_lf.collect_schema().names():
            sanitized_lf = sanitized_lf.with_columns(pl.lit(0).cast(pl.Int32).alias("mitre_stage"))
        logger.info("Applied zero-leakage filtering: stripped shortcut identifier columns via CanonicalFeatureExtractor.")
        return sanitized_lf

    def aggregate_temporal_bins(
        self, cleaned_lf: pl.LazyFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Aggregates flow events into continuous temporal bins of duration bin_duration_sec.

        Returns (features, binary_labels, mitre_stages, min_t, max_t)
        """
        time_bounds = cleaned_lf.select([
            pl.col("timestamp").min().alias("min_t"),
            pl.col("timestamp").max().alias("max_t"),
        ]).collect()

        min_t = float(time_bounds["min_t"][0]) if len(time_bounds) > 0 and time_bounds["min_t"][0] is not None else 0.0
        max_t = float(time_bounds["max_t"][0]) if len(time_bounds) > 0 and time_bounds["max_t"][0] is not None else 0.0

        total_time_span = max(max_t - min_t, 0.0)
        bin_dt = self.bin_duration_sec
        expected_bins = max(int(np.floor(total_time_span / bin_dt)) + 1, 1)

        agg_exprs = [
            pl.col("duration_norm").mean().fill_null(0.0).alias("duration_norm"),
            pl.col("byte_ratio").mean().fill_null(0.0).alias("byte_ratio"),
            pl.col("packet_rate").mean().fill_null(0.0).alias("packet_rate"),
            pl.col("iat_mean").mean().fill_null(0.0).alias("iat_mean"),
            pl.col("iat_std").mean().fill_null(0.0).alias("iat_std"),
            pl.col("ttl_mean").mean().fill_null(64.0).alias("ttl_mean"),
            pl.col("ttl_variance").mean().fill_null(0.0).alias("ttl_variance"),
            pl.col("tcp_syn_ratio").mean().fill_null(0.0).alias("tcp_syn_ratio"),
            pl.col("tcp_ack_ratio").mean().fill_null(0.0).alias("tcp_ack_ratio"),
            pl.col("tcp_window_norm").mean().fill_null(0.0).alias("tcp_window_norm"),
            pl.col("is_privileged_port").max().fill_null(0.0).alias("is_privileged_port"),
            pl.col("payload_entropy").mean().fill_null(0.0).alias("payload_entropy"),
            pl.col("tcp_rst_ratio").mean().fill_null(0.0).alias("tcp_rst_ratio"),
            pl.col("fwd_bwd_packet_ratio").mean().fill_null(0.0).alias("fwd_bwd_packet_ratio"),
            pl.col("payload_bytes_mean").mean().fill_null(0.0).alias("payload_bytes_mean"),
            pl.col("iat_max_norm").mean().fill_null(0.0).alias("iat_max_norm"),
            pl.col("label").max().alias("label"),
            pl.col("mitre_stage").max().alias("mitre_stage"),
        ]

        binned_df = (
            cleaned_lf
            .with_columns([
                ((pl.col("timestamp") - min_t) / bin_dt).floor().cast(pl.Int64).alias("bin_id")
            ])
            .group_by("bin_id")
            .agg(agg_exprs)
            .sort("bin_id")
            .collect()
        )

        features_full = np.zeros((expected_bins, 16), dtype=np.float32)
        labels_full = np.zeros(expected_bins, dtype=np.int32)
        mitre_full = np.zeros(expected_bins, dtype=np.int32)

        if len(binned_df) > 0:
            bin_ids = binned_df["bin_id"].to_numpy()
            valid_mask = (bin_ids >= 0) & (bin_ids < expected_bins)
            valid_bin_ids = bin_ids[valid_mask]

            feat_matrix = binned_df.select(self.features).to_numpy().astype(np.float32)
            label_vec = binned_df["label"].to_numpy().astype(np.int32)
            mitre_vec = binned_df["mitre_stage"].to_numpy().astype(np.int32)

            features_full[valid_bin_ids] = feat_matrix[valid_mask]
            labels_full[valid_bin_ids] = label_vec[valid_mask]
            mitre_full[valid_bin_ids] = mitre_vec[valid_mask]

        features_full = np.nan_to_num(features_full, nan=0.0, posinf=1e6, neginf=0.0)
        return features_full, labels_full, mitre_full, min_t, max_t

    def chronological_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mitre_stages: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Chronologically partitions continuous bins into Train (first 80%) and Val/Test (last 20%)."""
        total_bins = len(features)
        split_idx = int(np.floor(self.train_ratio * total_bins))

        train_feats = features[:split_idx]
        train_labels = labels[:split_idx]
        train_mitre = mitre_stages[:split_idx]

        val_feats = features[split_idx:]
        val_labels = labels[split_idx:]
        val_mitre = mitre_stages[split_idx:]

        logger.info(
            f"Chronological split (Train {self.train_ratio*100:.0f}% / Val {(1-self.train_ratio)*100:.0f}%): "
            f"Train bins: {len(train_feats)} (attacks: {np.sum(train_labels)}), "
            f"Val bins: {len(val_feats)} (attacks: {np.sum(val_labels)})"
        )
        return train_feats, train_labels, train_mitre, val_feats, val_labels, val_mitre

    def fit_and_scale(
        self,
        train_feats: np.ndarray,
        val_feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, RobustScaler]:
        """Fits RobustScaler strictly on the training partition and scales both partitions."""
        logger.info("Fitting RobustScaler strictly on training partition via CanonicalFeatureExtractor...")
        scaler = self.extractor.fit_scaler(train_feats)
        train_scaled = self.extractor.scale_features(train_feats)
        val_scaled = self.extractor.scale_features(val_feats)
        return train_scaled, val_scaled, scaler

    def generate_paired_windows(
        self,
        scaled_features: np.ndarray,
        labels: np.ndarray,
        mitre_stages: Optional[np.ndarray] = None,
        horizon_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized generation of chronologically paired state transitions, 5-step future horizon targets, and MITRE stages."""
        w_len = self.sequence_length
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

        num_windows = (num_bins - total_span) // self.stride + 1
        sw = np.lib.stride_tricks.sliding_window_view(scaled_features, window_shape=(w_len, num_feats))
        sw = sw[:, 0, :, :]

        indices = np.arange(0, num_windows * self.stride, self.stride)
        s_t_windows = sw[indices].astype(np.float32)
        s_next_windows = sw[indices + w_len].astype(np.float32)

        sw_labels = np.lib.stride_tricks.sliding_window_view(labels, window_shape=w_len)
        y_windows = np.any(sw_labels[indices] == 1, axis=1).astype(np.int32)

        # Multi-Horizon Ground Truth Target Vectors: Y_t = [y_t, y_{t+1}, y_{t+2}, y_{t+3}, y_{t+4}] in R^5
        y_horizon = np.zeros((num_windows, horizon_k), dtype=np.float32)
        for k in range(horizon_k):
            target_idx = np.clip(indices + k, 0, len(sw_labels) - 1)
            y_horizon[:, k] = np.any(sw_labels[target_idx] == 1, axis=1).astype(np.float32)

        # MITRE ATT&CK stages per window
        if mitre_stages is not None:
            sw_mitre = np.lib.stride_tricks.sliding_window_view(mitre_stages, window_shape=w_len)
            mitre_windows = np.max(sw_mitre[indices], axis=1).astype(np.int32)
        else:
            mitre_windows = y_windows * 2  # default attack to stage 2 (Initial Access)

        return s_t_windows, s_next_windows, y_windows, y_horizon, mitre_windows

    def save_parquet(
        self,
        s_t: np.ndarray,
        s_next: np.ndarray,
        labels: np.ndarray,
        y_horizon: np.ndarray,
        mitre_stages: np.ndarray,
        output_file: Path,
    ) -> int:
        """Saves paired 3D window dataset (B, 20, 16) to Snappy-compressed Parquet.

        Schema:
          - "s_t": List of floats (contiguous 20x16 = 320 elements per window)
          - "s_next": List of floats (contiguous 20x16 = 320 elements per window)
          - "label": Integer (0 or 1)
          - "y_horizon": List of floats (5 future horizon targets in R^5)
          - "mitre_stage": Integer (0..5)
        """
        num_windows = len(s_t)
        if num_windows == 0:
            logger.warning(f"No windows to save to {output_file}")
            return 0

        logger.info(f"Exporting {num_windows} windows to Parquet with Snappy compression: {output_file}...")

        elements_per_window = int(np.prod(s_t.shape[1:]))  # 20 * 16 = 320
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

        pq.write_table(table, output_file, compression="snappy")
        file_size_bytes = output_file.stat().st_size
        logger.info(f"Successfully wrote {output_file.name} ({file_size_bytes / (1024*1024):.2f} MB)")
        return file_size_bytes


def run_pipeline(
    input_csv: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    bin_duration_sec: float = 0.5,
    sequence_length: int = 20,
    stride: int = 1,
    train_ratio: float = 0.8,
) -> Dict[str, object]:
    """End-to-end execution of the high-performance RAM-safe preprocessing pipeline."""
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Starting Network World Model Preprocessing & Windowing Pipeline (16 Slots)")
    logger.info(f"Bin Duration: {bin_duration_sec}s | Sequence Length: {sequence_length} bins | Stride: {stride}")
    logger.info("=" * 70)

    # 1. Resolve paths
    raw_path = Path(input_csv) if input_csv else resolve_default_raw_path()
    preprocessor = UNSWPreprocessor(
        bin_duration_sec=bin_duration_sec,
        sequence_length=sequence_length,
        stride=stride,
        train_ratio=train_ratio,
        output_dir=output_dir,
    )

    # 2. Lazy scanning & sanitization
    lazy_df = preprocessor.scan_unsw_csv(raw_path)
    cleaned_df = preprocessor.clean_and_sanitize(lazy_df)

    # 3. Continuous temporal bin aggregation
    features_full, labels_full, mitre_full, min_t, max_t = preprocessor.aggregate_temporal_bins(cleaned_df)

    # 4. Chronological split (first 80% Train, last 20% Val/Test)
    train_feats, train_labels, train_mitre, val_feats, val_labels, val_mitre = preprocessor.chronological_split(
        features_full, labels_full, mitre_full
    )

    # 5. Fit RobustScaler strictly on train partition & scale features
    train_scaled, val_scaled, scaler = preprocessor.fit_and_scale(train_feats, val_feats)

    # 6. Generate paired state transition windows + 5-horizon targets + MITRE stages
    s_t_train, s_next_train, y_train, y_h_train, m_train = preprocessor.generate_paired_windows(
        train_scaled, train_labels, train_mitre, horizon_k=5
    )
    s_t_val, s_next_val, y_val, y_h_val, m_val = preprocessor.generate_paired_windows(
        val_scaled, val_labels, val_mitre, horizon_k=5
    )

    # 7. Export compressed Parquet files
    train_parquet_path = preprocessor.output_dir / "train_windows.parquet"
    val_parquet_path = preprocessor.output_dir / "val_windows.parquet"

    train_bytes = preprocessor.save_parquet(s_t_train, s_next_train, y_train, y_h_train, m_train, train_parquet_path)
    val_bytes = preprocessor.save_parquet(s_t_val, s_next_val, y_val, y_h_val, m_val, val_parquet_path)

    elapsed = time.time() - t_start
    total_mb = (train_bytes + val_bytes) / (1024 * 1024)

    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE EXECUTION SUMMARY")
    logger.info(f"Total Execution Time: {elapsed:.2f}s")
    logger.info(f"Timeline Span: {max_t - min_t:.1f}s ({len(features_full)} continuous 0.5s bins)")
    logger.info(f"Train Windows: {len(s_t_train)} (malicious: {np.sum(y_train)}, benign: {len(y_train) - np.sum(y_train)})")
    logger.info(f"Val Windows:   {len(s_t_val)} (malicious: {np.sum(y_val)}, benign: {len(y_val) - np.sum(y_val)})")
    logger.info(f"Tensor Shapes: S_t: {s_t_train.shape} | S_next: {s_next_train.shape} | y_horizon: {y_h_train.shape} | mitre: {m_train.shape}")
    logger.info(f"Scaler Checkpoint: {preprocessor.scaler_path} ({preprocessor.scaler_path.stat().st_size} bytes)")
    logger.info(f"Train Parquet:    {train_parquet_path} ({train_bytes / (1024*1024):.2f} MB)")
    logger.info(f"Val Parquet:      {val_parquet_path} ({val_bytes / (1024*1024):.2f} MB)")
    logger.info(f"Total Parquet Disk Footprint: {total_mb:.2f} MB")
    logger.info("=" * 70)

    return {
        "train_windows": len(s_t_train),
        "val_windows": len(s_t_val),
        "s_t_shape": s_t_train.shape,
        "s_next_shape": s_next_train.shape,
        "train_malicious": int(np.sum(y_train)),
        "train_benign": int(len(y_train) - np.sum(y_train)),
        "train_parquet_size_mb": train_bytes / (1024 * 1024),
        "val_parquet_size_mb": val_bytes / (1024 * 1024),
        "scaler_path": str(preprocessor.scaler_path),
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance, RAM-Safe Preprocessing for Network World Model (UNSW-NB15, 16 Slots)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input UNSW-NB15 CSV slice (default: auto-detect UNSW-NB15_1.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for processed artifacts (default: data/processed)",
    )
    parser.add_argument(
        "--bin-duration",
        type=float,
        default=0.5,
        help="Temporal bin duration in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Number of temporal bins per window (default: 20 -> 10 seconds)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride (default: 1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio (default: 0.8)",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            input_csv=args.input,
            output_dir=args.output_dir,
            bin_duration_sec=args.bin_duration,
            sequence_length=args.sequence_length,
            stride=args.stride,
            train_ratio=args.train_ratio,
        )
    except Exception as exc:
        logger.exception(f"Pipeline execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
