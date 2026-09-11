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
    daddrs = flows_df.get("daddr", flows_df.get("Dst IP", pd.Series(["unknown"] * n_flows))).astype(str).tolist()
    dports = flows_df.get("dport", flows_df.get("Dst Port", pd.Series([0] * n_flows))).astype(int).tolist()
    sports = flows_df.get("sport", flows_df.get("Src Port", pd.Series([0] * n_flows))).astype(int).tolist()

    # Bytes & Packets
    tot_bytes = float(flows_df.get("tot_bytes", flows_df.get("TotLen Fwd Pkts", pd.Series([0] * n_flows))).sum())
    tot_pkts = float(flows_df.get("tot_pkts", flows_df.get("Tot Fwd Pkts", pd.Series([0] * n_flows))).sum())
    src_bytes = float(flows_df.get("src_bytes", pd.Series([tot_bytes * 0.5] * n_flows)).sum())
    dst_bytes = max(tot_bytes - src_bytes, 0.0)

    # Durations & Timestamps
    durs = flows_df.get("dur", flows_df.get("Flow Duration", pd.Series([0.0] * n_flows))).astype(float).values
    dur_mean = float(np.mean(durs))
    dur_std = float(np.std(durs))
    dur_max = float(np.max(durs))

    start_ts = flows_df.get("StartTime", flows_df.get("timestamp", None))
    if start_ts is not None and len(start_ts) > 1:
        try:
            ts_num = pd.to_numeric(start_ts, errors="coerce").fillna(0).values
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
    states = flows_df.get("state", flows_df.get("State", pd.Series([""] * n_flows))).astype(str).str.upper()
    frac_established = float((states.str.contains("CON|EST")).sum() / n_flows)
    frac_reset = float((states.str.contains("RST")).sum() / n_flows)
    frac_interrupted = float((states.str.contains("INT|URH")).sum() / n_flows)

    dirs = flows_df.get("dir", flows_df.get("Dir", pd.Series(["->"] * n_flows))).astype(str)
    frac_outbound = float((dirs == "->").sum() / n_flows)
    frac_inbound = float((dirs == "<-").sum() / n_flows)

    # TCP Flags expanded
    flags_col = flows_df.get("flags", pd.Series([""] * n_flows)).astype(str).str.upper()
    frac_syn_only = float((flags_col.str.contains("S") & ~flags_col.str.contains("A")).sum() / n_flows)
    frac_syn_ack = float((flags_col.str.contains("S") & flags_col.str.contains("A")).sum() / n_flows)
    frac_fin = float(flags_col.str.contains("F").sum() / n_flows)
    frac_rst = float(flags_col.str.contains("R").sum() / n_flows)
    frac_psh = float(flags_col.str.contains("P").sum() / n_flows)
    frac_urg = float(flags_col.str.contains("U").sum() / n_flows)
    frac_ack = float(flags_col.str.contains("A").sum() / n_flows)

    # Protocols
    protos = flows_df.get("proto", flows_df.get("Protocol", pd.Series(["tcp"] * n_flows))).astype(str).str.lower()
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
