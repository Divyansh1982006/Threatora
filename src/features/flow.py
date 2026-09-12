"""Flow-level Feature Extraction Pipeline for NetForecast (39 Features).

Extracts connection statistics, expanded TCP flag distributions, volume counters,
directional ratios, entropy metrics, and beacon regularity from network flows.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def shannon_entropy(values: List[Any]) -> float:
    """Computes Shannon entropy of categorical distributions (base 2)."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return float(round(ent, 4))


def compute_beacon_regularity(timestamps: np.ndarray) -> float:
    """Computes coefficient of variation (CV) of inter-arrival times as beacon score.
    Periodic automated C2 heartbeats yield low variance (CV close to 0 -> high regularity close to 1).
    """
    if len(timestamps) < 3:
        return 0.0
    sorted_ts = np.sort(timestamps)
    iats = np.diff(sorted_ts)
    mean_iat = np.mean(iats)
    if mean_iat <= 1e-4:
        return 0.0
    std_iat = np.std(iats)
    cv = std_iat / mean_iat
    # Map into [0, 1] regularity score where 1 = perfect periodic beaconing
    regularity = float(1.0 / (1.0 + cv))
    return round(regularity, 4)


def extract_flow_window_features(flows_df: pd.DataFrame, window_duration: float = 60.0) -> Dict[str, float]:
    """Computes the 39 flow-level features for a single host within a 60-second window."""
    n_flows = len(flows_df)
    if n_flows == 0:
        return {col: 0.0 for col in [
            "n_flows", "n_unique_dst", "n_unique_dport", "n_unique_sport",
            "tot_bytes", "tot_pkts", "src_bytes", "dst_bytes",
            "bytes_per_sec", "pkts_per_sec", "avg_pkt_size", "bytes_per_flow", "pkts_per_flow", "egress_ratio",
            "frac_inbound", "frac_outbound", "frac_established", "frac_reset", "frac_interrupted",
            "frac_syn_only", "frac_syn_ack", "frac_fin", "frac_rst", "frac_psh", "frac_urg", "frac_ack",
            "dur_mean", "dur_std", "dur_max", "flow_iat_mean", "flow_iat_std", "beacon_regularity",
            "frac_tcp", "frac_udp", "frac_icmp", "dns_query_count",
            "dst_ip_entropy", "dport_entropy", "sport_entropy"
        ]}

    # Addresses & Ports
    daddr_col = next((c for c in ("daddr", "Dst IP") if c in flows_df.columns), None)
    dport_col = next((c for c in ("dport", "Dst Port") if c in flows_df.columns), None)
    sport_col = next((c for c in ("sport", "Src Port") if c in flows_df.columns), None)

    daddrs = flows_df[daddr_col].astype(str).tolist() if daddr_col else ["unknown"] * n_flows
    dports = flows_df[dport_col].astype(int).tolist() if dport_col else [0] * n_flows
    sports = flows_df[sport_col].astype(int).tolist() if sport_col else [0] * n_flows

    # Bytes & Packets
    tot_bytes_col = next((c for c in ("tot_bytes", "TotLen Fwd Pkts") if c in flows_df.columns), None)
    tot_pkts_col = next((c for c in ("tot_pkts", "Tot Fwd Pkts") if c in flows_df.columns), None)
    src_bytes_col = "src_bytes" if "src_bytes" in flows_df.columns else None

    tot_bytes = float(flows_df[tot_bytes_col].sum()) if tot_bytes_col else 0.0
    tot_pkts = float(flows_df[tot_pkts_col].sum()) if tot_pkts_col else 0.0
    src_bytes = float(flows_df[src_bytes_col].sum()) if src_bytes_col else tot_bytes * 0.5
    dst_bytes = max(tot_bytes - src_bytes, 0.0)

    # Durations & Timestamps
    dur_col = next((c for c in ("dur", "Flow Duration") if c in flows_df.columns), None)
    durs = flows_df[dur_col].astype(float).values if dur_col else np.zeros(n_flows)
    dur_mean = float(np.mean(durs))
    dur_std = float(np.std(durs))
    dur_max = float(np.max(durs))

    ts_col_flow = next((c for c in ("StartTime", "timestamp") if c in flows_df.columns), None)
    if ts_col_flow is not None and len(flows_df) > 1:
        try:
            ts_num = pd.to_numeric(flows_df[ts_col_flow], errors="coerce").fillna(0).values
            iats = np.diff(np.sort(ts_num))
            iat_mean = float(np.mean(iats))
            iat_std = float(np.std(iats))
            beacon = compute_beacon_regularity(ts_num)
        except Exception:
            iat_mean, iat_std, beacon = 0.0, 0.0, 0.0
    else:
        iat_mean, iat_std, beacon = 0.0, 0.0, 0.0

    # Rates
    dur_total = max(window_duration, 1.0)
    bytes_per_sec = tot_bytes / dur_total
    pkts_per_sec = tot_pkts / dur_total
    avg_pkt_size = tot_bytes / max(tot_pkts, 1.0)
    bytes_per_flow = tot_bytes / max(n_flows, 1.0)
    pkts_per_flow = tot_pkts / max(n_flows, 1.0)
    egress_ratio = float(src_bytes / max(tot_bytes, 1.0))

    # Direction & States (Argus state mapping)
    state_col = next((c for c in ("state", "State") if c in flows_df.columns), None)
    states = flows_df[state_col].astype(str).str.upper() if state_col else pd.Series([""] * n_flows)
    frac_established = float((states.str.contains("CON|EST", regex=True)).sum() / n_flows)
    frac_reset = float((states.str.contains("RST", regex=False)).sum() / n_flows)
    frac_interrupted = float((states.str.contains("INT|URH", regex=True)).sum() / n_flows)

    dir_col = next((c for c in ("dir", "Dir") if c in flows_df.columns), None)
    dirs = flows_df[dir_col].astype(str) if dir_col else pd.Series(["->"] * n_flows)
    frac_outbound = float((dirs == "->").sum() / n_flows)
    frac_inbound = float((dirs == "<-").sum() / n_flows)

    # TCP Flags expanded
    flags_col_name = "flags" if "flags" in flows_df.columns else None
    flags_col = flows_df[flags_col_name].astype(str).str.upper() if flags_col_name else pd.Series([""] * n_flows)
    frac_syn_only = float((flags_col.str.contains("S", regex=False) & ~flags_col.str.contains("A", regex=False)).sum() / n_flows)
    frac_syn_ack = float((flags_col.str.contains("S", regex=False) & flags_col.str.contains("A", regex=False)).sum() / n_flows)
    frac_fin = float(flags_col.str.contains("F", regex=False).sum() / n_flows)
    frac_rst = float(flags_col.str.contains("R", regex=False).sum() / n_flows)
    frac_psh = float(flags_col.str.contains("P", regex=False).sum() / n_flows)
    frac_urg = float(flags_col.str.contains("U", regex=False).sum() / n_flows)
    frac_ack = float(flags_col.str.contains("A", regex=False).sum() / n_flows)

    # Protocols
    proto_col = next((c for c in ("proto", "Protocol") if c in flows_df.columns), None)
    protos = flows_df[proto_col].astype(str).str.lower() if proto_col else pd.Series(["tcp"] * n_flows)
    frac_tcp = float((protos == "tcp").sum() / n_flows)
    frac_udp = float((protos == "udp").sum() / n_flows)
    frac_icmp = float((protos == "icmp").sum() / n_flows)
    dns_count = float(((protos.reset_index(drop=True) == "udp") & (pd.Series(dports).reset_index(drop=True) == 53)).sum())

    return {
        "n_flows": float(n_flows),
        "n_unique_dst": float(len(set(daddrs))),
        "n_unique_dport": float(len(set(dports))),
        "n_unique_sport": float(len(set(sports))),
        "tot_bytes": tot_bytes,
        "tot_pkts": tot_pkts,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "bytes_per_sec": bytes_per_sec,
        "pkts_per_sec": pkts_per_sec,
        "avg_pkt_size": avg_pkt_size,
        "bytes_per_flow": bytes_per_flow,
        "pkts_per_flow": pkts_per_flow,
        "egress_ratio": egress_ratio,
        "frac_inbound": frac_inbound,
        "frac_outbound": frac_outbound,
        "frac_established": frac_established,
        "frac_reset": frac_reset,
        "frac_interrupted": frac_interrupted,
        "frac_syn_only": frac_syn_only,
        "frac_syn_ack": frac_syn_ack,
        "frac_fin": frac_fin,
        "frac_rst": frac_rst,
        "frac_psh": frac_psh,
        "frac_urg": frac_urg,
        "frac_ack": frac_ack,
        "dur_mean": dur_mean,
        "dur_std": dur_std,
        "dur_max": dur_max,
        "flow_iat_mean": iat_mean,
        "flow_iat_std": iat_std,
        "beacon_regularity": beacon,
        "frac_tcp": frac_tcp,
        "frac_udp": frac_udp,
        "frac_icmp": frac_icmp,
        "dns_query_count": float(dns_count),
        "dst_ip_entropy": shannon_entropy(daddrs),
        "dport_entropy": shannon_entropy(dports),
        "sport_entropy": shannon_entropy(sports)
    }
