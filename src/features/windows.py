"""Windowing and Aggregation Pipeline for NetForecast.

Groups network flows and packet telemetry into 60-second (Host, Window) state cells (S_t).
Combines 39 flow-level + 23 packet-level features into the 62-dimensional state vector.
Provides FeatureScaler with log1p and z-score normalization serializable to scaler.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd

from ..config import (
    WINDOW_SECONDS, SEQUENCE_LENGTH, MIN_FLOWS_PER_CELL,
    FLOW_FEATURE_COLS, PACKET_FEATURE_COLS, ALL_FEATURE_COLS,
    CHECKPOINT_DIR
)
from .flow import extract_flow_window_features
from .packet import extract_packet_window_features
from ..mitre import label_flow_stage, determine_dominant_stage


LOG1P_COLUMNS = [
    "n_flows", "n_unique_dst", "n_unique_dport", "n_unique_sport",
    "tot_bytes", "tot_pkts", "src_bytes", "dst_bytes",
    "bytes_per_sec", "pkts_per_sec", "avg_pkt_size", "bytes_per_flow", "pkts_per_flow",
    "dur_mean", "dur_std", "dur_max", "flow_iat_mean", "flow_iat_std", "dns_query_count",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_iat_mean", "pkt_iat_std", "pkt_iat_max",
    "tcp_win_mean", "tcp_win_std", "zero_win_count", "retrans_count"
]


class FeatureScaler:
    """Robust feature scaler applying log1p to heavy-tailed counters and z-score normalization."""

    def __init__(self):
        self.mean: np.ndarray = np.zeros(len(ALL_FEATURE_COLS), dtype=np.float32)
        self.std: np.ndarray = np.ones(len(ALL_FEATURE_COLS), dtype=np.float32)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray):
        """Fits mean and std over 2D array of shape (N, 62)."""
        X_trans = self._apply_log1p(X)
        self.mean = np.mean(X_trans, axis=0).astype(np.float32)
        std = np.std(X_trans, axis=0).astype(np.float32)
        self.std = np.where(std < 1e-4, 1.0, std)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms 2D (N, 62) or 3D (B, T, 62) array."""
        orig_shape = X.shape
        if X.ndim == 3:
            b, t, d = orig_shape
            flat_x = X.reshape(-1, d)
            norm_x = self.transform(flat_x)
            return norm_x.reshape(b, t, d)

        X_trans = self._apply_log1p(X)
        if not self.is_fitted:
            return X_trans.astype(np.float32)
        return ((X_trans - self.mean) / self.std).astype(np.float32)

    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        X_copy = np.array(X, copy=True, dtype=np.float32)
        for idx, col in enumerate(ALL_FEATURE_COLS):
            if col in LOG1P_COLUMNS:
                X_copy[:, idx] = np.log1p(np.maximum(X_copy[:, idx], 0.0))
        return X_copy

    def save(self, file_path: Optional[Path | str] = None):
        target = Path(file_path) if file_path else (CHECKPOINT_DIR / "scaler.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_columns": ALL_FEATURE_COLS,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "is_fitted": self.is_fitted
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Optional[Path | str] = None) -> "FeatureScaler":
        target = Path(file_path) if file_path else (CHECKPOINT_DIR / "scaler.json")
        scaler = cls()
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            scaler.mean = np.array(data["mean"], dtype=np.float32)
            scaler.std = np.array(data["std"], dtype=np.float32)
            scaler.is_fitted = data.get("is_fitted", True)
        return scaler


def build_host_windows_from_flows(
    flows_df: pd.DataFrame,
    packets_by_window: Optional[Dict[Tuple[str, int], List[Dict[str, Any]]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int]]]:
    """Aggregates flows dataframe into 60s host-window feature matrix, labels, and stages.

    Returns:
        X: (num_cells, 62) feature matrix
        y_infilt: (num_cells,) binary infiltration labels
        y_stage: (num_cells,) MITRE ATT&CK stages (0..5)
        meta: list of (host_ip, window_index) tuples
    """
    if flows_df.empty:
        return np.empty((0, len(ALL_FEATURE_COLS))), np.empty((0,)), np.empty((0,)), []

    df = flows_df.copy()

    # Resolve timestamps to epoch seconds
    ts_col = None
    for cand in ("StartTime", "timestamp", "Timestamp", "start_time"):
        if cand in df.columns:
            ts_col = cand
            break

    if ts_col is not None:
        try:
            df["epoch"] = pd.to_datetime(df[ts_col], errors="coerce").astype(int) / 1e9
        except Exception:
            df["epoch"] = pd.to_numeric(df[ts_col], errors="coerce").fillna(0)
    else:
        # Synthetic incremental epoch
        df["epoch"] = np.arange(len(df)) * 0.5

    df["window_idx"] = (df["epoch"] // WINDOW_SECONDS).astype(int)

    # Resolve Host IP (Source)
    host_col = None
    for cand in ("saddr", "Src IP", "src_ip", "Source IP"):
        if cand in df.columns:
            host_col = cand
            break
    if host_col is None:
        df["host_ip"] = "192.168.1.100"
    else:
        df["host_ip"] = df[host_col].astype(str)

    # Group by (host_ip, window_idx)
    groups = df.groupby(["host_ip", "window_idx"])
    x_rows = []
    y_infilts = []
    y_stages = []
    meta = []

    for (host, w_idx), group in groups:
        if len(group) < MIN_FLOWS_PER_CELL:
            continue

        # 39 Flow features
        flow_feats = extract_flow_window_features(group, window_duration=WINDOW_SECONDS)

        # 23 Packet features
        pkts = []
        if packets_by_window and (host, w_idx) in packets_by_window:
            pkts = packets_by_window[(host, w_idx)]
        pkt_feats = extract_packet_window_features(pkts)

        # Combine into 62-length vector in exact canonical order
        combined_row = [flow_feats[c] for c in FLOW_FEATURE_COLS] + [pkt_feats[c] for c in PACKET_FEATURE_COLS]
        x_rows.append(combined_row)

        # Infiltration & Stage Labels
        stage_counts = Counter()
        for _, flow in group.iterrows():
            stg = label_flow_stage(flow.to_dict())
            stage_counts[stg] += 1

        dom_stage = determine_dominant_stage(stage_counts)
        is_attack = 1.0 if dom_stage > 0 else 0.0

        y_infilts.append(is_attack)
        y_stages.append(dom_stage)
        meta.append((host, int(w_idx)))

    if not x_rows:
        return np.empty((0, len(ALL_FEATURE_COLS))), np.empty((0,)), np.empty((0,)), []

    X = np.array(x_rows, dtype=np.float32)
    y_infilt = np.array(y_infilts, dtype=np.float32)
    y_stage = np.array(y_stages, dtype=np.int64)

    return X, y_infilt, y_stage, meta
