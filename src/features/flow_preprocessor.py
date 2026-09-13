"""Flow-Level Telemetry Preprocessor for NetForecast World Model (12 Features).

Extracts, cleans, clips, standardizes, and windows CSV flow telemetry into sequences
for consumption by the FlowLSTMWorldModel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from ..model.flow_world_model import FLOW_FEATURE_NAMES


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "flow"


# Flexible column aliases for mapping generic flow CSVs to the 12 model features
COLUMN_ALIASES: Dict[str, List[str]] = {
    "flow_duration": ["flow_duration", "Flow Duration", "dur", "Duration_sec", "duration"],
    "Duration": ["Duration", "duration", "dur", "flow_duration"],
    "Rate": ["Rate", "rate", "flow_rate", "pkts_per_sec", "bytes_per_sec"],
    "Srate": ["Srate", "srate", "src_rate", "src_pkts_per_sec", "Rate"],
    "Drate": ["Drate", "drate", "dst_rate", "dst_pkts_per_sec"],
    "fin_flag_number": ["fin_flag_number", "fin_flag", "FIN", "flags_fin", "frac_fin"],
    "syn_flag_number": ["syn_flag_number", "syn_flag", "SYN", "flags_syn", "frac_syn_only"],
    "rst_flag_number": ["rst_flag_number", "rst_flag", "RST", "flags_rst", "frac_rst"],
    "ack_flag_number": ["ack_flag_number", "ack_flag", "ACK", "flags_ack", "frac_ack"],
    "Tot size": ["Tot size", "tot_size", "tot_bytes", "TotLen Fwd Pkts", "tot_len", "bytes"],
    "IAT": ["IAT", "iat", "flow_iat", "flow_iat_mean", "iat_mean", "inter_arrival_time"],
    "Number": ["Number", "number", "tot_pkts", "Tot Fwd Pkts", "pkts", "n_flows"],
}


class FlowPreprocessor:
    """Robust Preprocessor for CICIoT2023 12-Feature Flow Datasets and Telemetry."""

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        window_size: int = 20,
        stride: int = 5,
    ):
        self.window_size = window_size
        self.stride = stride
        self.feature_names = list(FLOW_FEATURE_NAMES)

        base_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_ARTIFACT_DIR

        self.stats_file = base_dir / "feature_statistics.json"
        self.scaler_file = base_dir / "scaler.pkl"

        self.medians: Dict[str, float] = {}
        self.clips: Dict[str, Dict[str, float]] = {}
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.scaler = None

        self._load_statistics()

    def _load_statistics(self):
        """Loads fitted medians, clipping bounds, and scaler parameters."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                self.medians = stats.get("medians", {})
                self.clips = stats.get("clip_stats", {})
                if "scaler_mean" in stats and "scaler_scale" in stats:
                    self.scaler_mean = np.array(stats["scaler_mean"], dtype=np.float64)
                    self.scaler_scale = np.array(stats["scaler_scale"], dtype=np.float64)
            except Exception as e:
                print(f"[!] Warning: Failed to read {self.stats_file}: {e}")

        if self.scaler_file.exists():
            try:
                self.scaler = joblib.load(self.scaler_file)
            except Exception as e:
                print(f"[!] Warning: Failed to load scaler.pkl: {e}")

        # Fallback defaults if stats are not present
        if not self.medians:
            self.medians = {f: 0.0 for f in self.feature_names}
            self.medians["Duration"] = 64.0
            self.medians["Number"] = 9.5
            self.medians["Tot size"] = 54.0
            self.medians["IAT"] = 8.3e7

        if not self.clips:
            self.clips = {f: {"low": -1e9, "high": 1e9} for f in self.feature_names}

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolves column aliases to standard 12 feature names."""
        df_cols_lower = {str(c).strip().lower(): c for c in df.columns}
        extracted: Dict[str, pd.Series] = {}

        for target_col in self.feature_names:
            found = False
            # Check exact name
            if target_col in df.columns:
                extracted[target_col] = pd.to_numeric(df[target_col], errors="coerce")
                found = True
            else:
                # Check aliases
                aliases = COLUMN_ALIASES.get(target_col, [target_col])
                for alias in aliases:
                    alias_lower = alias.lower()
                    if alias_lower in df_cols_lower:
                        orig_col = df_cols_lower[alias_lower]
                        extracted[target_col] = pd.to_numeric(df[orig_col], errors="coerce")
                        found = True
                        break

            # If still not found, fill with median
            if not found:
                med_val = self.medians.get(target_col, 0.0)
                extracted[target_col] = pd.Series([med_val] * len(df), index=df.index, dtype=np.float64)

        return pd.DataFrame(extracted, index=df.index)

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Preprocesses tabular DataFrame into scaled 2D array of shape (N, 12)."""
        if len(df) == 0:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)

        mapped_df = self._map_columns(df)
        arr = mapped_df[self.feature_names].to_numpy(dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = np.nan

        # Median imputation and clipping per feature
        for j, f in enumerate(self.feature_names):
            med = self.medians.get(f, 0.0)
            c_low = self.clips.get(f, {}).get("low", -1e9)
            c_high = self.clips.get(f, {}).get("high", 1e9)

            arr[:, j] = np.nan_to_num(arr[:, j], nan=med, posinf=c_high, neginf=c_low)
            arr[:, j] = np.clip(arr[:, j], c_low, c_high)

        # Standardization (z-score)
        if self.scaler is not None:
            arr_scaled = self.scaler.transform(arr).astype(np.float32)
        elif self.scaler_mean is not None and self.scaler_scale is not None:
            arr_scaled = ((arr - self.scaler_mean) / np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)).astype(np.float32)
        else:
            arr_scaled = arr.astype(np.float32)

        return arr_scaled

    def build_sequences(
        self,
        scaled_array: np.ndarray,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """Windows 2D array (N, 12) into temporal sequences (Num_windows, Window_size, 12).

        Args:
            scaled_array: Scaled array of shape (N, 12)
            stride: Step size between window starts (defaults to self.stride)

        Returns:
            3D numpy array of shape (Num_windows, Window_size, 12)
        """
        n_rows, n_feats = scaled_array.shape
        w_size = self.window_size
        s_step = stride or self.stride

        if n_rows == 0:
            return np.empty((0, w_size, n_feats), dtype=np.float32)

        if n_rows < w_size:
            # Pad by repeating the initial row
            pad_len = w_size - n_rows
            pad_head = np.repeat(scaled_array[:1], pad_len, axis=0)
            seq = np.vstack([pad_head, scaled_array])
            return np.expand_dims(seq, axis=0).astype(np.float32)

        starts = range(0, n_rows - w_size + 1, s_step)
        windows = [scaled_array[s : s + w_size] for s in starts]

        # Always include the trailing window if not captured by stride
        last_start = n_rows - w_size
        if starts and list(starts)[-1] != last_start:
            windows.append(scaled_array[last_start : n_rows])

        return np.stack(windows).astype(np.float32)

    def process_csv(
        self,
        csv_path_or_df: Union[str, Path, pd.DataFrame],
        stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """End-to-end processing from CSV file/DataFrame to 3D sequences.

        Returns:
            Tuple of (sequences (Num_windows, 20, 12), raw_dataframe)
        """
        if isinstance(csv_path_or_df, (str, Path)):
            df = pd.read_csv(csv_path_or_df, low_memory=False)
        else:
            df = csv_path_or_df.copy()

        scaled_2d = self.preprocess_dataframe(df)
        sequences = self.build_sequences(scaled_2d, stride=stride)
        return sequences, df
