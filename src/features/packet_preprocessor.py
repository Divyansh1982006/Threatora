"""Packet-Level Preprocessor for Threatora World Model (20 Features).

Parses PCAP / packet telemetry, groups packets into 5-second state windows,
computes 20 statistical features, standardizes using scaler.pkl, and windows into sequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib

from ..model.packet_world_model import PACKET_FEATURE_NAMES
from .packet import parse_pcap_file


DEFAULT_PACKET_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "packet"
WINDOW_SECONDS = 5.0


class PacketPreprocessor:
    """Preprocessor for Network Packet Telemetry & PCAP Files (20 Features)."""

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        sequence_length: int = 10,
        window_seconds: float = WINDOW_SECONDS,
    ):
        self.sequence_length = sequence_length
        self.window_seconds = window_seconds
        self.feature_names = list(PACKET_FEATURE_NAMES)

        base_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_PACKET_ARTIFACT_DIR
        self.scaler_file = base_dir / "scaler.pkl"

        # Fallback to source location if not copied yet
        if not self.scaler_file.exists():
            alt_scaler = Path(r"D:\world_model_lstm\models\scaler.pkl")
            if alt_scaler.exists():
                self.scaler_file = alt_scaler

        self.scaler = None
        self._load_scaler()

    def _load_scaler(self):
        """Loads fitted StandardScaler."""
        if self.scaler_file.exists():
            try:
                self.scaler = joblib.load(self.scaler_file)
            except Exception as e:
                print(f"[!] Warning: Could not load packet scaler from {self.scaler_file}: {e}")

    def build_states_from_packets(self, packets: List[Dict[str, Any]]) -> pd.DataFrame:
        """Groups raw packet dictionaries into 5-second state windows with 20 features."""
        if not packets:
            return pd.DataFrame(columns=self.feature_names)

        df = pd.DataFrame(packets)
        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df), dtype=np.float64) * 0.01

        # Fill missing attributes
        for col in ["ttl", "window_size", "length", "payload_size", "src_port", "dst_port", "iat"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in ["src_ip", "dst_ip"]:
            if col not in df.columns:
                df[col] = "unknown"
            else:
                df[col] = df[col].astype(str).fillna("unknown")

        payload_col = "payload_size" if "payload_size" in df.columns else "length"
        start_time = float(df["timestamp"].min())

        df["window"] = ((df["timestamp"] - start_time) // self.window_seconds).astype(int)
        df["flow_key"] = (
            df["src_ip"] + ":" + df["src_port"].astype(str) + "->" +
            df["dst_ip"] + ":" + df["dst_port"].astype(str)
        )

        # Flag extraction if tcp_flags string or integer present
        if "tcp_flags" in df.columns:
            flags_str = df["tcp_flags"].astype(str).str.upper()
            df["syn_count"] = flags_str.str.contains("S").astype(int)
            df["ack_count"] = flags_str.str.contains("A").astype(int)
            df["fin_count"] = flags_str.str.contains("F").astype(int)
            df["rst_count"] = flags_str.str.contains("R").astype(int)
            df["psh_count"] = flags_str.str.contains("P").astype(int)
            df["urg_count"] = flags_str.str.contains("U").astype(int)
        else:
            for f in ["syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count"]:
                df[f] = 0

        grouped = df.groupby("window", sort=True)
        state_df = grouped.agg(
            packet_count=("window", "size"),
            byte_count=(payload_col, "sum"),
            flow_count=("flow_key", "nunique"),
            unique_src_ips=("src_ip", "nunique"),
            unique_dst_ips=("dst_ip", "nunique"),
            syn_count=("syn_count", "sum"),
            ack_count=("ack_count", "sum"),
            fin_count=("fin_count", "sum"),
            rst_count=("rst_count", "sum"),
            psh_count=("psh_count", "sum"),
            urg_count=("urg_count", "sum"),
            ttl_mean=("ttl", "mean"),
            ttl_std=("ttl", "std"),
            payload_mean=(payload_col, "mean"),
            payload_std=(payload_col, "std"),
            window_mean=("window_size", "mean"),
            window_std=("window_size", "std"),
            iat_mean=("iat", "mean"),
            iat_std=("iat", "std"),
        ).reset_index(drop=True)

        state_df["unique_ports"] = (
            grouped["src_port"].nunique().to_numpy() + grouped["dst_port"].nunique().to_numpy()
        )

        for col in self.feature_names:
            if col not in state_df.columns:
                state_df[col] = 0.0

        state_df = state_df[self.feature_names].fillna(0.0)
        return state_df

    def process_dataframe_states(self, state_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Scales and windows a 20-feature DataFrame into sequences of shape (N_windows, 10, 20)."""
        if len(state_df) == 0:
            return np.empty((0, self.sequence_length, len(self.feature_names)), dtype=np.float32), state_df

        # Reorder columns
        for col in self.feature_names:
            if col not in state_df.columns:
                state_df[col] = 0.0

        arr = state_df[self.feature_names].to_numpy(dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = 0.0

        # Scale features
        if self.scaler is not None:
            try:
                arr_scaled = self.scaler.transform(arr).astype(np.float32)
            except Exception:
                arr_scaled = arr.astype(np.float32)
        else:
            arr_scaled = arr.astype(np.float32)

        # Build sequences of length 10
        n_rows, n_feats = arr_scaled.shape
        seq_len = self.sequence_length

        if n_rows < seq_len:
            pad_len = seq_len - n_rows
            pad_head = np.repeat(arr_scaled[:1], pad_len, axis=0)
            seq = np.vstack([pad_head, arr_scaled])
            return np.expand_dims(seq, axis=0).astype(np.float32), state_df

        windows = []
        for s in range(0, n_rows - seq_len + 1):
            windows.append(arr_scaled[s : s + seq_len])

        return np.stack(windows).astype(np.float32), state_df

    def process_pcap(self, pcap_path: Union[str, Path]) -> Tuple[np.ndarray, pd.DataFrame]:
        """Extracts packets from PCAP, builds 20-feature state windows, scales, and creates sequences."""
        pkts = parse_pcap_file(str(pcap_path))
        state_df = self.build_states_from_packets(pkts)
        sequences, state_df = self.process_dataframe_states(state_df)
        return sequences, state_df

    def process_state_csv(self, csv_path: Union[str, Path]) -> Tuple[np.ndarray, pd.DataFrame]:
        """Ingests pre-computed state CSV and builds sequences."""
        df = pd.read_csv(csv_path)
        return self.process_dataframe_states(df)
