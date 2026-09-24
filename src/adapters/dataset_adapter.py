"""Dynamic Canonical Feature Adapter for Heterogeneous Network Flow Telemetry.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Preprocesses, adapts, and standardizes heterogeneous network flow schemas
(UNSW-NB15, CSE-CIC-IDS2018, and CTU-13) into 12 canonical continuous slots:
  1. duration_norm
  2. byte_ratio
  3. packet_rate
  4. iat_mean
  5. iat_std
  6. ttl_mean
  7. ttl_variance
  8. tcp_syn_ratio
  9. tcp_ack_ratio
  10. tcp_window_norm
  11. is_privileged_port
  12. payload_entropy

Enforces zero data leakage by stripping all shortcut identifiers
(srcip, dstip, sport, dsport, timestamps, flow_ids).
Yields normalized rolling time windows of fixed shape (B, 20, 12).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import joblib
import numpy as np
import polars as pl
import torch
import yaml
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger("CanonicalFeatureAdapter")

# 16 Canonical continuous feature slots mandated by Threatora World Model specification
CANONICAL_SLOTS: List[str] = [
    "duration_norm",
    "byte_ratio",
    "packet_rate",
    "iat_mean",
    "iat_std",
    "ttl_mean",
    "ttl_variance",
    "tcp_syn_ratio",
    "tcp_ack_ratio",
    "tcp_window_norm",
    "is_privileged_port",
    "payload_entropy",
    "tcp_rst_ratio",
    "fwd_bwd_packet_ratio",
    "payload_bytes_mean",
    "iat_max_norm",
]

# Shortcut identity columns that must be strictly stripped to prevent data leakage
SHORTCUT_IDENTIFIER_COLS: List[str] = [
    "srcip", "dstip", "src_ip", "dst_ip", "source_ip", "destination_ip",
    "saddr", "daddr", "srcaddr", "dstaddr",
    "sport", "dsport", "src_port", "dst_port", "source_port", "destination_port",
    "dport", "sport",
    "flow_id", "flow id", "id", "session_id", "uid",
    "timestamp", "flow timestamp", "stime", "ltime", "starttime", "lasttime", "time",
]


def load_schema_registry(registry_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads schema_registry.yaml defining canonical slot mappings across datasets."""
    if registry_path is None:
        registry_path = Path(__file__).resolve().parent / "schema_registry.yaml"
    registry_path = Path(registry_path)

    if not registry_path.exists():
        logger.warning(f"Schema registry file not found: {registry_path}, using built-in defaults.")
        return {}

    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_port_value(val: object) -> int:
    """Safely parses port values from int, float, string, or hexadecimal representations."""
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


class CanonicalFeatureExtractor:
    """Dynamic Adapter and Preprocessor extracting 16 canonical continuous feature slots.

    Features:
      - Polars lazy inspection of columns to auto-detect dataset schema or accept explicit tag.
      - Strict zero data leakage enforcement (strips IP, port, and timestamp identifiers).
      - Continuous temporal binning and RobustScaler normalization.
      - Yields fixed rolling temporal windows of shape (B, 20, 16).
    """

    def __init__(
        self,
        dataset: Optional[str] = None,
        bin_duration_sec: float = 0.5,
        sequence_length: int = 20,
        stride: int = 1,
        scaler: Optional[RobustScaler] = None,
        scaler_path: Optional[Union[str, Path]] = None,
    ):
        self.dataset_hint = dataset.lower().strip() if dataset else None
        self.bin_duration_sec = bin_duration_sec
        self.sequence_length = sequence_length
        self.stride = stride
        self.registry = load_schema_registry()
        self.slots = list(CANONICAL_SLOTS)
        assert len(self.slots) == 16, f"Expected 16 canonical slots, got {len(self.slots)}"

        self.scaler = scaler
        self.scaler_path = Path(scaler_path) if scaler_path else None
        if self.scaler is None and self.scaler_path and self.scaler_path.exists():
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Loaded fitted scaler from: {self.scaler_path}")
            except Exception as exc:
                logger.warning(f"Failed to load scaler from {self.scaler_path}: {exc}")

    def detect_dataset(self, columns: List[str]) -> str:
        """Inspects column names and determines whether schema is UNSW-NB15, CIC-IDS2018, or CTU-13."""
        if self.dataset_hint:
            hint = self.dataset_hint.replace("-", "_")
            if "unsw" in hint:
                return "unsw_nb15"
            if "cic" in hint or "ids2018" in hint:
                return "cic_ids2018"
            if "ctu" in hint or "binetflow" in hint:
                return "ctu_13"

        cols_lower = [c.strip().lower() for c in columns]
        cols_set = set(cols_lower)

        # 0. Check Tabular Telemetry signatures (pre-windowed CSV/TXT telemetry)
        tabular_sigs = {
            "byte_rate", "packet_rate", "mean_iat", "var_iat", "max_iat",
            "syn_flag_ratio", "ack_flag_ratio", "fin_flag_ratio", "rst_flag_ratio",
            "psh_flag_ratio", "bwd_to_fwd_ratio", "mean_flow_duration", "is_privileged_port",
            "window_id", "time_s"
        }
        if len(cols_set.intersection(tabular_sigs)) >= 3:
            return "tabular_telemetry"

        # 1. Check CIC-IDS2018 signatures
        cic_sigs = {"flow duration", "tot fwd pkts", "tot bwd pkts", "flow byts/s", "init fwd win byts"}
        if len(cols_set.intersection(cic_sigs)) >= 2:
            return "cic_ids2018"

        # 2. Check CTU-13 signatures
        ctu_sigs = {"srcaddr", "dstaddr", "totbytes", "totpkts", "srcbytes"}
        if len(cols_set.intersection(ctu_sigs)) >= 2 or "binetflow" in "".join(cols_lower):
            return "ctu_13"

        # 3. Check UNSW-NB15 signatures
        unsw_sigs = {"sbytes", "dbytes", "sttl", "dttl", "swin", "dwin", "sintpkt", "tcprtt"}
        if len(cols_set.intersection(unsw_sigs)) >= 2:
            return "unsw_nb15"

        # Fallback to UNSW-NB15 convention if standard 49 columns or partial match
        return "unsw_nb15"

    def _find_column(self, col_names: List[str], candidates: List[str]) -> Optional[str]:
        """Finds first matching column name from candidates list (case-insensitive)."""
        col_map = {c.lower(): c for c in col_names}
        for cand in candidates:
            if cand.lower() in col_map:
                return col_map[cand.lower()]
        return None

    def sanitize_and_extract(
        self,
        lf: pl.LazyFrame,
        dataset: Optional[str] = None,
    ) -> Tuple[pl.LazyFrame, str]:
        """Lazy Polars transformation projecting raw columns to 12 canonical continuous slots.

        Strictly eliminates shortcut identifier columns to enforce zero data leakage.
        Returns:
            Tuple of (sanitized_lazyframe, detected_dataset_name)
        """
        cols = lf.collect_schema().names()
        ds = dataset if dataset else self.detect_dataset(cols)
        col_lower_map = {c.strip().lower(): c for c in cols}

        # 0. Multicast & Local Discovery Protocol Filtering (224.0.0.0/4 and discovery ports)
        dst_ip_candidates = ["dstip", "dst_ip", "destination_ip", "dstaddr", "daddr"]
        dst_ip_col = self._find_column(cols, dst_ip_candidates)
        if dst_ip_col is not None:
            ip_str = pl.col(dst_ip_col).cast(pl.String).str.strip_chars()
            multicast_mask = (
                ip_str.str.starts_with("224.") | ip_str.str.starts_with("225.") |
                ip_str.str.starts_with("226.") | ip_str.str.starts_with("227.") |
                ip_str.str.starts_with("228.") | ip_str.str.starts_with("229.") |
                ip_str.str.starts_with("230.") | ip_str.str.starts_with("231.") |
                ip_str.str.starts_with("232.") | ip_str.str.starts_with("233.") |
                ip_str.str.starts_with("234.") | ip_str.str.starts_with("235.") |
                ip_str.str.starts_with("236.") | ip_str.str.starts_with("237.") |
                ip_str.str.starts_with("238.") | ip_str.str.starts_with("239.") |
                (ip_str == "255.255.255.255")
            )
            lf = lf.filter(~multicast_mask)

        ts_candidates = ["time_s", "time", "timestamp", "stime", "ltime", "starttime", "lasttime", "flow timestamp", "window_id"]
        ts_col = self._find_column(cols, ts_candidates)
        if ts_col is not None:
            raw_ts = pl.col(ts_col)
            ts_float = raw_ts.cast(pl.Float64, strict=False)
            ts_str = raw_ts.cast(pl.String, strict=False)
            ts_dt = (
                ts_str.str.to_datetime(format="%Y/%m/%d %H:%M:%S%.f", strict=False)
                .fill_null(ts_str.str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False))
                .fill_null(ts_str.str.to_datetime(format="%Y/%m/%d %H:%M:%S", strict=False))
                .fill_null(ts_str.str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
            ).cast(pl.Int64, strict=False) / 1_000_000.0
            ts_expr = pl.coalesce([ts_float, ts_dt]).fill_null(0.0).alias("timestamp")
        else:
            ts_expr = pl.int_range(0, pl.len(), eager=False).cast(pl.Float64).alias("timestamp")

        # 2. Resolve label column
        label_candidates = ["label", "is_malicious", "attack_cat"]
        lbl_col = self._find_column(cols, label_candidates)
        if lbl_col is not None:
            # Map string labels like 'Normal' or '0' to int 0, attacks to 1
            label_col_expr = pl.col(lbl_col)
            label_expr = (
                pl.when(label_col_expr.cast(pl.String).str.to_lowercase().is_in(["0", "normal", "benign", "background", "none", ""]))
                .then(pl.lit(0))
                .when(label_col_expr.cast(pl.Int32, strict=False).is_not_null())
                .then(label_col_expr.cast(pl.Int32, strict=False))
                .otherwise(pl.lit(1))
            ).cast(pl.Int32).alias("label")
        else:
            label_expr = pl.lit(0).cast(pl.Int32).alias("label")

        # 3. Resolve destination port for is_privileged_port computation
        if "is_privileged_port" in col_lower_map:
            is_privileged_expr = pl.col(col_lower_map["is_privileged_port"]).cast(pl.Float32, strict=False).fill_null(0.0).alias("is_privileged_port")
        else:
            dst_port_candidates = ["dsport", "dst_port", "destination port", "dst port", "dport", "daddr"]
            dport_col = self._find_column(cols, dst_port_candidates)
            if dport_col is not None:
                port_str = pl.col(dport_col).cast(pl.String).str.strip_chars()
                port_parsed = (
                    pl.when(port_str.str.starts_with("0x") | port_str.str.starts_with("0X"))
                    .then(pl.lit(0))
                    .otherwise(port_str.cast(pl.Int64, strict=False).fill_null(0))
                )
                is_privileged_expr = (
                    pl.when((port_parsed > 0) & (port_parsed < 1024))
                    .then(pl.lit(1.0))
                    .otherwise(pl.lit(0.0))
                ).cast(pl.Float32).alias("is_privileged_port")
            else:
                is_privileged_expr = pl.lit(0.0).cast(pl.Float32).alias("is_privileged_port")

        # 4. Canonical Slot Expressions per Dataset
        exprs: List[pl.Expr] = [ts_expr, label_expr, is_privileged_expr]

        if ds == "cic_ids2018":
            exprs.extend(self._build_cic_expressions(col_lower_map))
        elif ds == "ctu_13":
            exprs.extend(self._build_ctu_expressions(col_lower_map))
        elif ds == "tabular_telemetry":
            exprs.extend(self._build_tabular_expressions(col_lower_map))
        else:
            exprs.extend(self._build_unsw_expressions(col_lower_map))

        # 5. Check for mitre_stage, technique_id, or attack category
        if "mitre_stage" in col_lower_map:
            stage_raw = pl.col(col_lower_map["mitre_stage"])
            stage_str = stage_raw.cast(pl.String).str.to_lowercase()
            exprs.append(
                pl.when(stage_raw.cast(pl.Int32, strict=False).is_not_null())
                .then(stage_raw.cast(pl.Int32, strict=False))
                .when(stage_str.is_in(["normal", "0", "benign", "background", ""]))
                .then(pl.lit(0))
                .when(stage_str.str.contains("fuzz|recon|analy"))
                .then(pl.lit(1))
                .when(stage_str.str.contains("exploit|shell"))
                .then(pl.lit(2))
                .when(stage_str.str.contains("generic"))
                .then(pl.lit(3))
                .when(stage_str.str.contains("backdoor"))
                .then(pl.lit(4))
                .when(stage_str.str.contains("dos|worm"))
                .then(pl.lit(5))
                .otherwise(pl.lit(0))
                .cast(pl.Int32)
                .alias("mitre_stage")
            )
        else:
            cat_col = self._find_column(cols, ["attack_cat", "attack_category"])
            if cat_col is not None:
                cat_expr = pl.col(cat_col).cast(pl.String).fill_null("Normal")
                mitre_expr = (
                    pl.when(cat_expr.str.to_lowercase().is_in(["normal", "0", "benign", "background", ""]))
                    .then(pl.lit(0))
                    .when(cat_expr.str.to_lowercase().str.contains("fuzz|recon|analy"))
                    .then(pl.lit(1))
                    .when(cat_expr.str.to_lowercase().str.contains("exploit|shell"))
                    .then(pl.lit(2))
                    .when(cat_expr.str.to_lowercase().str.contains("generic"))
                    .then(pl.lit(3))
                    .when(cat_expr.str.to_lowercase().str.contains("backdoor"))
                    .then(pl.lit(4))
                    .when(cat_expr.str.to_lowercase().str.contains("dos|worm"))
                    .then(pl.lit(5))
                    .otherwise(pl.lit(2))
                ).cast(pl.Int32).alias("mitre_stage")
                exprs.append(mitre_expr)

        if "technique_id" in col_lower_map:
            exprs.append(pl.col(col_lower_map["technique_id"]).cast(pl.String).fill_null("None").alias("technique_id"))
        if "window_id" in col_lower_map:
            exprs.append(pl.col(col_lower_map["window_id"]).cast(pl.Int64, strict=False).alias("window_id"))

        # Select canonical columns + timestamp + label (+ mitre_stage / technique_id if available)
        transformed_lf = lf.select(exprs)
        logger.info(f"Adapted dataset '{ds}' to canonical schema with zero data leakage.")
        return transformed_lf, ds

    def _build_tabular_expressions(self, col_map: Dict[str, str]) -> List[pl.Expr]:
        """Builds canonical expressions for tabular window telemetry CSV/TXT directly."""
        dur_c = col_map.get("mean_flow_duration", col_map.get("dur", col_map.get("duration")))
        dur_expr = pl.col(dur_c).cast(pl.Float32, strict=False).fill_null(1.0) if dur_c else pl.lit(1.0).cast(pl.Float32)

        bwd_fwd_c = col_map.get("bwd_to_fwd_ratio", col_map.get("byte_ratio"))
        byte_ratio_expr = pl.col(bwd_fwd_c).cast(pl.Float32, strict=False).fill_null(1.0) if bwd_fwd_c else pl.lit(1.0).cast(pl.Float32)

        pr_c = col_map.get("packet_rate")
        pr_expr = pl.col(pr_c).cast(pl.Float32, strict=False).fill_null(10.0) if pr_c else pl.lit(10.0).cast(pl.Float32)

        iat_m_c = col_map.get("mean_iat", col_map.get("iat_mean"))
        iat_m_expr = pl.col(iat_m_c).cast(pl.Float32, strict=False).fill_null(0.05) if iat_m_c else pl.lit(0.05).cast(pl.Float32)

        var_iat_c = col_map.get("var_iat", col_map.get("iat_std"))
        if var_iat_c:
            iat_std_expr = pl.col(var_iat_c).cast(pl.Float32, strict=False).clip(0.0, None).sqrt().fill_null(0.01)
        else:
            iat_std_expr = pl.lit(0.01).cast(pl.Float32)

        ttl_mean_expr = pl.lit(64.0).cast(pl.Float32)
        ttl_var_expr = pl.lit(0.0).cast(pl.Float32)

        syn_c = col_map.get("syn_flag_ratio", col_map.get("tcp_syn_ratio"))
        syn_expr = pl.col(syn_c).cast(pl.Float32, strict=False).fill_null(0.05) if syn_c else pl.lit(0.05).cast(pl.Float32)

        ack_c = col_map.get("ack_flag_ratio", col_map.get("tcp_ack_ratio"))
        ack_expr = pl.col(ack_c).cast(pl.Float32, strict=False).fill_null(0.90) if ack_c else pl.lit(0.90).cast(pl.Float32)

        win_norm_expr = pl.lit(1.0).cast(pl.Float32)

        psh_c = col_map.get("psh_flag_ratio", col_map.get("payload_entropy"))
        entropy_expr = (pl.col(psh_c).cast(pl.Float32, strict=False).fill_null(0.2) * 4.0) if psh_c else pl.lit(0.5).cast(pl.Float32)

        rst_c = col_map.get("rst_flag_ratio", col_map.get("tcp_rst_ratio"))
        rst_expr = pl.col(rst_c).cast(pl.Float32, strict=False).fill_null(0.0) if rst_c else pl.lit(0.0).cast(pl.Float32)

        fwd_bwd_expr = byte_ratio_expr.alias("fwd_bwd_packet_ratio")

        br_c = col_map.get("byte_rate")
        if br_c and pr_c:
            pld_mean_expr = (pl.col(br_c).cast(pl.Float32, strict=False) / pl.max_horizontal([pl.col(pr_c).cast(pl.Float32, strict=False), pl.lit(1.0)])).cast(pl.Float32)
        else:
            pld_mean_expr = pl.lit(200.0).cast(pl.Float32)

        max_iat_c = col_map.get("max_iat", col_map.get("iat_max_norm"))
        iat_max_expr = pl.col(max_iat_c).cast(pl.Float32, strict=False).log1p().fill_null(0.1) if max_iat_c else pl.lit(0.1).cast(pl.Float32)

        return [
            dur_expr.alias("duration_norm"),
            byte_ratio_expr.alias("byte_ratio"),
            pr_expr.alias("packet_rate"),
            iat_m_expr.alias("iat_mean"),
            iat_std_expr.alias("iat_std"),
            ttl_mean_expr.alias("ttl_mean"),
            ttl_var_expr.alias("ttl_variance"),
            syn_expr.alias("tcp_syn_ratio"),
            ack_expr.alias("tcp_ack_ratio"),
            win_norm_expr.alias("tcp_window_norm"),
            entropy_expr.alias("payload_entropy"),
            rst_expr.alias("tcp_rst_ratio"),
            fwd_bwd_expr.alias("fwd_bwd_packet_ratio"),
            pld_mean_expr.alias("payload_bytes_mean"),
            iat_max_expr.alias("iat_max_norm"),
        ]

    def _build_unsw_expressions(self, col_map: Dict[str, str]) -> List[pl.Expr]:
        """Builds canonical expressions for UNSW-NB15 schema."""
        dur_c = col_map.get("dur", col_map.get("duration"))
        dur_expr = pl.col(dur_c).cast(pl.Float64, strict=False).fill_null(0.0) if dur_c else pl.lit(0.0)

        sbytes_c = col_map.get("sbytes", col_map.get("src_bytes"))
        dbytes_c = col_map.get("dbytes", col_map.get("dst_bytes"))
        sbytes_expr = pl.col(sbytes_c).cast(pl.Float64, strict=False).fill_null(0.0) if sbytes_c else pl.lit(0.0)
        dbytes_expr = pl.col(dbytes_c).cast(pl.Float64, strict=False).fill_null(0.0) if dbytes_c else pl.lit(0.0)

        spkts_c = col_map.get("spkts", col_map.get("src_pkts"))
        dpkts_c = col_map.get("dpkts", col_map.get("dst_pkts"))
        spkts_expr = pl.col(spkts_c).cast(pl.Float64, strict=False).fill_null(0.0) if spkts_c else pl.lit(0.0)
        dpkts_expr = pl.col(dpkts_c).cast(pl.Float64, strict=False).fill_null(0.0) if dpkts_c else pl.lit(0.0)

        sintpkt_c = col_map.get("sintpkt", col_map.get("sinpkt"))
        dintpkt_c = col_map.get("dintpkt", col_map.get("dinpkt"))
        sintpkt_expr = pl.col(sintpkt_c).cast(pl.Float64, strict=False).fill_null(0.0) if sintpkt_c else pl.lit(0.0)
        dintpkt_expr = pl.col(dintpkt_c).cast(pl.Float64, strict=False).fill_null(0.0) if dintpkt_c else pl.lit(0.0)

        sjit_c = col_map.get("sjit")
        djit_c = col_map.get("djit")
        sjit_expr = pl.col(sjit_c).cast(pl.Float64, strict=False).fill_null(0.0) if sjit_c else pl.lit(0.0)
        djit_expr = pl.col(djit_c).cast(pl.Float64, strict=False).fill_null(0.0) if djit_c else pl.lit(0.0)

        sttl_c = col_map.get("sttl", col_map.get("src_ttl"))
        dttl_c = col_map.get("dttl", col_map.get("dst_ttl"))
        sttl_expr = pl.col(sttl_c).cast(pl.Float64, strict=False).fill_null(64.0) if sttl_c else pl.lit(64.0)
        dttl_expr = pl.col(dttl_c).cast(pl.Float64, strict=False).fill_null(64.0) if dttl_c else pl.lit(64.0)

        swin_c = col_map.get("swin")
        dwin_c = col_map.get("dwin")
        swin_expr = pl.col(swin_c).cast(pl.Float64, strict=False).fill_null(0.0) if swin_c else pl.lit(0.0)
        dwin_expr = pl.col(dwin_c).cast(pl.Float64, strict=False).fill_null(0.0) if dwin_c else pl.lit(0.0)

        state_c = col_map.get("state")
        synack_c = col_map.get("synack")
        ackdat_c = col_map.get("ackdat")
        synack_expr = pl.col(synack_c).cast(pl.Float64, strict=False).fill_null(0.0) if synack_c else pl.lit(0.0)
        ackdat_expr = pl.col(ackdat_c).cast(pl.Float64, strict=False).fill_null(0.0) if ackdat_c else pl.lit(0.0)

        if state_c:
            state_str = pl.col(state_c).cast(pl.String).str.to_uppercase().fill_null("")
            syn_expr = ((state_str.is_in(["CON", "REQ", "ACC", "FIN"])) | (synack_expr > 0)).cast(pl.Float32)
            ack_expr = ((state_str.is_in(["CON", "FIN", "ACC", "CLO"])) | (ackdat_expr > 0)).cast(pl.Float32)
            # TCP Flag Ratio Safeguard: For established bidirectional sessions (CON), clamp SYN ratio to baseline (< 0.15)
            syn_expr = pl.when((ack_expr > 0) & (state_str == "CON")).then(pl.lit(0.08)).otherwise(syn_expr).cast(pl.Float32)
        else:
            syn_expr = pl.lit(0.0).cast(pl.Float32)
            ack_expr = pl.lit(0.0).cast(pl.Float32)

        res_bdy_c = col_map.get("res_bdy_len")
        res_expr = pl.col(res_bdy_c).cast(pl.Float64, strict=False).fill_null(0.0) if res_bdy_c else pl.lit(0.0)

        # Slot 13: tcp_rst_ratio (Detects evasive scan termination and aborted connections)
        if state_c:
            rst_expr = state_str.str.contains("RST").cast(pl.Float32)
        else:
            rst_expr = pl.lit(0.0).cast(pl.Float32)

        # Slot 14: fwd_bwd_packet_ratio (Bidirectional packet count imbalance)
        fwd_bwd_pkt_expr = (spkts_expr / (dpkts_expr + 1e-6)).clip(0.0, 1000.0).cast(pl.Float32)

        # Slot 15: payload_bytes_mean (Mean payload size per packet)
        pld_mean_expr = ((sbytes_expr + dbytes_expr) / (spkts_expr + dpkts_expr + 1e-6)).clip(0.0, 65535.0).cast(pl.Float32)

        # Slot 16: iat_max_norm (Burstiness and C2 jitter regularity)
        iat_max_expr = pl.max_horizontal([sintpkt_expr, dintpkt_expr]).fill_null(0.0).log1p().cast(pl.Float32)

        return [
            dur_expr.log1p().cast(pl.Float32).alias("duration_norm"),
            (dbytes_expr / (sbytes_expr + 1e-6)).log1p().cast(pl.Float32).alias("byte_ratio"),
            ((spkts_expr + dpkts_expr) / (dur_expr + 1e-6)).log1p().cast(pl.Float32).alias("packet_rate"),
            ((sintpkt_expr + dintpkt_expr) / 2.0).cast(pl.Float32).alias("iat_mean"),
            ((sjit_expr + djit_expr) / 2.0).cast(pl.Float32).alias("iat_std"),
            ((sttl_expr + dttl_expr) / 2.0).cast(pl.Float32).alias("ttl_mean"),
            (((sttl_expr - dttl_expr) ** 2) / 4.0).cast(pl.Float32).alias("ttl_variance"),
            syn_expr.alias("tcp_syn_ratio"),
            ack_expr.alias("tcp_ack_ratio"),
            ((swin_expr + dwin_expr) / (2.0 * 65535.0)).cast(pl.Float32).alias("tcp_window_norm"),
            (res_expr / (res_expr + 1500.0)).cast(pl.Float32).alias("payload_entropy"),
            rst_expr.alias("tcp_rst_ratio"),
            fwd_bwd_pkt_expr.alias("fwd_bwd_packet_ratio"),
            pld_mean_expr.alias("payload_bytes_mean"),
            iat_max_expr.alias("iat_max_norm"),
        ]

    def _build_cic_expressions(self, col_map: Dict[str, str]) -> List[pl.Expr]:
        """Builds canonical expressions for CSE-CIC-IDS2018 schema."""
        dur_c = col_map.get("flow duration")
        dur_expr = pl.col(dur_c).cast(pl.Float64, strict=False).fill_null(0.0) if dur_c else pl.lit(0.0)
        dur_sec = dur_expr / 1e6

        fwd_b_c = col_map.get("totlen fwd pkts", col_map.get("total length of fwd packets"))
        bwd_b_c = col_map.get("totlen bwd pkts", col_map.get("total length of bwd packets"))
        fwd_b = pl.col(fwd_b_c).cast(pl.Float64, strict=False).fill_null(0.0) if fwd_b_c else pl.lit(0.0)
        bwd_b = pl.col(bwd_b_c).cast(pl.Float64, strict=False).fill_null(0.0) if bwd_b_c else pl.lit(0.0)

        fwd_p_c = col_map.get("tot fwd pkts", col_map.get("total fwd packets"))
        bwd_p_c = col_map.get("tot bwd pkts", col_map.get("total backward packets"))
        fwd_p = pl.col(fwd_p_c).cast(pl.Float64, strict=False).fill_null(0.0) if fwd_p_c else pl.lit(0.0)
        bwd_p = pl.col(bwd_p_c).cast(pl.Float64, strict=False).fill_null(0.0) if bwd_p_c else pl.lit(0.0)

        rate_c = col_map.get("flow pkts/s", col_map.get("flow packets/s"))
        rate_expr = pl.col(rate_c).cast(pl.Float64, strict=False).fill_null(0.0) if rate_c else ((fwd_p + bwd_p) / (dur_sec + 1e-6))

        iat_m_c = col_map.get("flow iat mean")
        iat_m_expr = pl.col(iat_m_c).cast(pl.Float64, strict=False).fill_null(0.0) / 1e6 if iat_m_c else pl.lit(0.0)

        iat_s_c = col_map.get("flow iat std")
        iat_s_expr = pl.col(iat_s_c).cast(pl.Float64, strict=False).fill_null(0.0) / 1e6 if iat_s_c else pl.lit(0.0)

        syn_c = col_map.get("syn flag cnt")
        syn_expr = pl.col(syn_c).cast(pl.Float32, strict=False).fill_null(0.0) if syn_c else pl.lit(0.0)

        ack_c = col_map.get("ack flag cnt")
        ack_expr = pl.col(ack_c).cast(pl.Float32, strict=False).fill_null(0.0) if ack_c else pl.lit(0.0)

        fwin_c = col_map.get("init fwd win byts", col_map.get("init_win_bytes_forward"))
        bwin_c = col_map.get("init bwd win byts", col_map.get("init_win_bytes_backward"))
        fwin = pl.col(fwin_c).cast(pl.Float64, strict=False).fill_null(0.0) if fwin_c else pl.lit(0.0)
        bwin = pl.col(bwin_c).cast(pl.Float64, strict=False).fill_null(0.0) if bwin_c else pl.lit(0.0)

        std_c = col_map.get("pkt len std", col_map.get("packet length std"))
        mean_c = col_map.get("pkt len mean", col_map.get("packet length mean"))
        std_expr = pl.col(std_c).cast(pl.Float64, strict=False).fill_null(0.0) if std_c else pl.lit(0.0)
        mean_expr = pl.col(mean_c).cast(pl.Float64, strict=False).fill_null(0.0) if mean_c else pl.lit(1.0)
        entropy_expr = (std_expr / (mean_expr + 1e-6)).clip(0.0, 8.0)

        rst_c = col_map.get("rst flag cnt", col_map.get("rst flag count"))
        rst_expr = pl.col(rst_c).cast(pl.Float32, strict=False).fill_null(0.0) if rst_c else pl.lit(0.0).cast(pl.Float32)
        fwd_bwd_pkt_expr = (fwd_p / (bwd_p + 1e-6)).clip(0.0, 1000.0).cast(pl.Float32)
        pld_mean_expr = ((fwd_b + bwd_b) / (fwd_p + bwd_p + 1e-6)).clip(0.0, 65535.0).cast(pl.Float32)
        iat_max_c = col_map.get("flow iat max")
        iat_max_expr = (pl.col(iat_max_c).cast(pl.Float64, strict=False).fill_null(0.0) / 1e6).log1p().cast(pl.Float32) if iat_max_c else iat_m_expr.log1p().cast(pl.Float32)

        return [
            dur_sec.log1p().cast(pl.Float32).alias("duration_norm"),
            (bwd_b / (fwd_b + 1e-6)).log1p().cast(pl.Float32).alias("byte_ratio"),
            rate_expr.log1p().cast(pl.Float32).alias("packet_rate"),
            iat_m_expr.cast(pl.Float32).alias("iat_mean"),
            iat_s_expr.cast(pl.Float32).alias("iat_std"),
            pl.lit(64.0).cast(pl.Float32).alias("ttl_mean"),
            pl.lit(0.0).cast(pl.Float32).alias("ttl_variance"),
            syn_expr.alias("tcp_syn_ratio"),
            ack_expr.alias("tcp_ack_ratio"),
            ((fwin + bwin) / (2.0 * 65535.0)).cast(pl.Float32).alias("tcp_window_norm"),
            entropy_expr.cast(pl.Float32).alias("payload_entropy"),
            rst_expr.alias("tcp_rst_ratio"),
            fwd_bwd_pkt_expr.alias("fwd_bwd_packet_ratio"),
            pld_mean_expr.alias("payload_bytes_mean"),
            iat_max_expr.alias("iat_max_norm"),
        ]

    def _build_ctu_expressions(self, col_map: Dict[str, str]) -> List[pl.Expr]:
        """Builds canonical expressions for CTU-13 schema."""
        dur_c = col_map.get("dur")
        dur_expr = pl.col(dur_c).cast(pl.Float64, strict=False).fill_null(0.0) if dur_c else pl.lit(0.0)

        tot_b_c = col_map.get("totbytes")
        src_b_c = col_map.get("srcbytes")
        tot_b = pl.col(tot_b_c).cast(pl.Float64, strict=False).fill_null(0.0) if tot_b_c else pl.lit(0.0)
        src_b = pl.col(src_b_c).cast(pl.Float64, strict=False).fill_null(0.0) if src_b_c else pl.lit(0.0)
        dst_b = (tot_b - src_b).clip(0.0, None)

        tot_p_c = col_map.get("totpkts")
        tot_p = pl.col(tot_p_c).cast(pl.Float64, strict=False).fill_null(0.0) if tot_p_c else pl.lit(0.0)

        state_c = col_map.get("state", col_map.get("flags"))
        if state_c:
            state_str = pl.col(state_c).cast(pl.String).str.to_uppercase().fill_null("")
            syn_expr = state_str.str.contains("S").cast(pl.Float32)
            ack_expr = (state_str.str.contains("A") | state_str.str.contains("CON")).cast(pl.Float32)
            rst_expr = state_str.str.contains("R").cast(pl.Float32)
        else:
            syn_expr = pl.lit(0.0).cast(pl.Float32)
            ack_expr = pl.lit(0.0).cast(pl.Float32)
            rst_expr = pl.lit(0.0).cast(pl.Float32)

        iat_expr = dur_expr / (tot_p.clip(1.0, None))
        fwd_bwd_pkt_expr = pl.lit(1.0).cast(pl.Float32)
        pld_mean_expr = (tot_b / (tot_p + 1e-6)).clip(0.0, 65535.0).cast(pl.Float32)
        iat_max_expr = iat_expr.log1p().cast(pl.Float32)

        return [
            dur_expr.log1p().cast(pl.Float32).alias("duration_norm"),
            (dst_b / (src_b + 1e-6)).log1p().cast(pl.Float32).alias("byte_ratio"),
            (tot_p / (dur_expr + 1e-6)).log1p().cast(pl.Float32).alias("packet_rate"),
            iat_expr.cast(pl.Float32).alias("iat_mean"),
            pl.lit(0.0).cast(pl.Float32).alias("iat_std"),
            pl.lit(64.0).cast(pl.Float32).alias("ttl_mean"),
            pl.lit(0.0).cast(pl.Float32).alias("ttl_variance"),
            syn_expr.alias("tcp_syn_ratio"),
            ack_expr.alias("tcp_ack_ratio"),
            pl.lit(0.5).cast(pl.Float32).alias("tcp_window_norm"),
            pl.lit(0.0).cast(pl.Float32).alias("payload_entropy"),
            rst_expr.alias("tcp_rst_ratio"),
            fwd_bwd_pkt_expr.alias("fwd_bwd_packet_ratio"),
            pld_mean_expr.alias("payload_bytes_mean"),
            iat_max_expr.alias("iat_max_norm"),
        ]

    def aggregate_temporal_bins(
        self,
        sanitized_lf: pl.LazyFrame,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Aggregates flow events into continuous temporal bins of duration bin_duration_sec.

        Returns:
            Tuple of:
              - features: np.ndarray of shape (N_bins, 16)
              - labels: np.ndarray of shape (N_bins,)
              - min_t: float
              - max_t: float
        """
        # Step 1: Query global time bounds
        time_bounds = sanitized_lf.select([
            pl.col("timestamp").min().alias("min_t"),
            pl.col("timestamp").max().alias("max_t"),
        ]).collect()

        min_t = float(time_bounds["min_t"][0]) if len(time_bounds) > 0 and time_bounds["min_t"][0] is not None else 0.0
        max_t = float(time_bounds["max_t"][0]) if len(time_bounds) > 0 and time_bounds["max_t"][0] is not None else 0.0

        total_time_span = max(max_t - min_t, 0.0)
        bin_dt = self.bin_duration_sec
        expected_bins = max(int(np.floor(total_time_span / bin_dt)) + 1, 1)

        # Step 2: Binning and aggregation
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
        ]

        binned_df = (
            sanitized_lf
            .with_columns([
                ((pl.col("timestamp") - min_t) / bin_dt).floor().cast(pl.Int64).alias("bin_id")
            ])
            .group_by("bin_id")
            .agg(agg_exprs)
            .sort("bin_id")
            .collect()
        )

        # Step 3: Populate continuous timeline array
        features_full = np.zeros((expected_bins, len(self.slots)), dtype=np.float32)
        labels_full = np.zeros(expected_bins, dtype=np.int32)

        if len(binned_df) > 0:
            bin_ids = binned_df["bin_id"].to_numpy()
            valid_mask = (bin_ids >= 0) & (bin_ids < expected_bins)
            valid_bin_ids = bin_ids[valid_mask]

            feat_matrix = binned_df.select(self.slots).to_numpy().astype(np.float32)
            label_vec = binned_df["label"].to_numpy().astype(np.int32)

            features_full[valid_bin_ids] = feat_matrix[valid_mask]
            labels_full[valid_bin_ids] = label_vec[valid_mask]

        features_full = np.nan_to_num(features_full, nan=0.0, posinf=1e6, neginf=0.0)
        return features_full, labels_full, min_t, max_t

    def fit_scaler(self, features: np.ndarray) -> RobustScaler:
        """Fits RobustScaler strictly on train partition."""
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
        self.scaler.fit(features)
        if self.scaler_path:
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Saved fitted scaler to: {self.scaler_path}")
        return self.scaler

    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scales features using fitted scaler with robust fallback and clipping."""
        if self.scaler is None:
            self.fit_scaler(features)

        expected_feats = getattr(self.scaler, "n_features_in_", features.shape[1])
        if features.shape[1] < expected_feats:
            pad_width = expected_feats - features.shape[1]
            features_padded = np.pad(features, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
            scaled = self.scaler.transform(features_padded).astype(np.float32)
            return np.clip(scaled[:, :features.shape[1]], -4.0, 4.0)
        elif features.shape[1] > expected_feats:
            scaled = self.scaler.transform(features[:, :expected_feats]).astype(np.float32)
            rem = features[:, expected_feats:]
            return np.clip(np.hstack([scaled, rem]), -4.0, 4.0)

        scaled = self.scaler.transform(features).astype(np.float32)
        return np.clip(scaled, -4.0, 4.0)

    def generate_windows(
        self,
        scaled_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Vectorized extraction of rolling temporal windows (N, 20, 12).

        Returns:
            Tuple of:
              - s_t: np.ndarray of shape (num_windows, 20, 12)
              - s_next: np.ndarray of shape (num_windows, 20, 12) if span >= 40, else None
              - y: np.ndarray of shape (num_windows,) if labels provided, else None
        """
        w_len = self.sequence_length  # 20
        num_bins, num_feats = scaled_features.shape

        if num_bins < w_len:
            # Pad if fewer than 20 bins
            pad_len = w_len - num_bins
            scaled_features = np.pad(scaled_features, ((0, pad_len), (0, 0)), mode="edge")
            if labels is not None:
                labels = np.pad(labels, (0, pad_len), mode="constant", constant_values=0)
            num_bins = w_len

        sw = np.lib.stride_tricks.sliding_window_view(scaled_features, window_shape=(w_len, num_feats))
        sw = sw[:, 0, :, :]  # (num_windows, 20, 12)

        indices = np.arange(0, len(sw), self.stride)
        s_t = sw[indices]

        # S_{t+1} next state paired windows
        total_span = 2 * w_len
        s_next = None
        if num_bins >= total_span:
            num_paired = (num_bins - total_span) // self.stride + 1
            paired_indices = np.arange(0, num_paired * self.stride, self.stride)
            s_next = sw[paired_indices + w_len]

        # Labels
        y = None
        if labels is not None:
            sw_lbl = np.lib.stride_tricks.sliding_window_view(labels, window_shape=w_len)
            y = np.any(sw_lbl[indices] == 1, axis=1).astype(np.int32)

        return s_t, s_next, y

    def generate_paired_windows(
        self,
        scaled_features: np.ndarray,
        labels: np.ndarray,
        return_horizon: bool = False,
        horizon_k: int = 5,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Vectorized generation of chronologically paired state transitions (S_t, S_{t+1}) and threat labels.

        Returns:
            If return_horizon is False:
              Tuple of (s_t, s_next, y)
            If return_horizon is True:
              Tuple of (s_t, s_next, y, y_horizon) where y_horizon has shape (num_windows, horizon_k)
        """
        w_len = self.sequence_length
        total_span = 2 * w_len
        num_bins, num_feats = scaled_features.shape

        if num_bins < total_span:
            empty_feats = np.empty((0, w_len, num_feats), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int32)
            if return_horizon:
                return empty_feats, empty_feats, empty_y, np.empty((0, horizon_k), dtype=np.float32)
            return empty_feats, empty_feats, empty_y

        num_windows = (num_bins - total_span) // self.stride + 1
        sw = np.lib.stride_tricks.sliding_window_view(scaled_features, window_shape=(w_len, num_feats))
        sw = sw[:, 0, :, :]

        indices = np.arange(0, num_windows * self.stride, self.stride)
        s_t_windows = sw[indices].astype(np.float32)
        s_next_windows = sw[indices + w_len].astype(np.float32)

        sw_labels = np.lib.stride_tricks.sliding_window_view(labels, window_shape=w_len)
        y_windows = np.any(sw_labels[indices] == 1, axis=1).astype(np.int32)

        if return_horizon:
            y_horizon = np.zeros((num_windows, horizon_k), dtype=np.float32)
            for k in range(horizon_k):
                target_idx = np.clip(indices + k, 0, len(sw_labels) - 1)
                y_horizon[:, k] = np.any(sw_labels[target_idx] == 1, axis=1).astype(np.float32)
            return s_t_windows, s_next_windows, y_windows, y_horizon

        return s_t_windows, s_next_windows, y_windows

    def yield_batches(
        self,
        windows: Union[np.ndarray, torch.Tensor],
        batch_size: int = 64,
        as_torch: bool = True,
    ) -> Iterator[Union[torch.Tensor, np.ndarray]]:
        """Yields fixed batches of shape (B, 20, num_features).

        Args:
            windows: Array of shape (N, 20, num_features) or (N, 20 * num_features)
            batch_size: Mini-batch size B
            as_torch: If True, yields torch.Tensor of shape torch.Size([B, 20, num_features])
        """
        feat_dim = len(self.slots)
        if isinstance(windows, torch.Tensor):
            if windows.ndim == 3:
                feat_dim = windows.shape[-1]
            arr = windows.view(-1, self.sequence_length, feat_dim)
        else:
            w_arr = np.asarray(windows, dtype=np.float32)
            if w_arr.ndim == 3:
                feat_dim = w_arr.shape[-1]
            arr = w_arr.reshape(-1, self.sequence_length, feat_dim)

        n = len(arr)
        for i in range(0, n, batch_size):
            batch = arr[i : i + batch_size]
            if as_torch:
                yield torch.as_tensor(batch, dtype=torch.float32)
            else:
                yield batch

    def process_dataframe(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        dataset: Optional[str] = None,
        batch_size: Optional[int] = 64,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """End-to-end processing from DataFrame to torch.Tensor of shape (B, 20, num_features)."""
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        sanitized_lf, _ = self.sanitize_and_extract(lf, dataset=dataset)
        features, labels, _, _ = self.aggregate_temporal_bins(sanitized_lf)
        scaled = self.scale_features(features)
        s_t, _, y = self.generate_windows(scaled, labels)

        tensors = torch.from_numpy(s_t).float()
        return tensors
