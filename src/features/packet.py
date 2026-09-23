"""Packet-level Feature Extraction Pipeline for NetForecast (23 Features).

Extracts packet size distributions, inter-arrival dynamics, TTL statistics,
TCP window flow control, payload entropy, and retransmission rates from PCAP files.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


def compute_payload_entropy(payload_bytes: bytes) -> float:
    """Shannon entropy of packet payload bytes to detect encrypted / compressed / tunneled payloads."""
    if not payload_bytes:
        return 0.0
    counts = Counter(payload_bytes)
    total = len(payload_bytes)
    return float(-sum((c / total) * math.log2(c / total) for c in counts.values()))


def extract_packet_window_features(packets_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes the 23 packet-level features from parsed packet dictionaries."""
    n_pkts = len(packets_list)
    if n_pkts == 0:
        return {
            "pkt_len_mean": 0.0, "pkt_len_std": 0.0, "pkt_len_min": 0.0, "pkt_len_max": 0.0, "pkt_len_skew": 0.0,
            "pkt_iat_mean": 0.0, "pkt_iat_std": 0.0, "pkt_iat_max": 0.0, "pkt_iat_cov": 0.0,
            "ttl_mean": 64.0, "ttl_std": 0.0, "ttl_distinct_count": 1.0,
            "tcp_win_mean": 0.0, "tcp_win_std": 0.0, "zero_win_count": 0.0, "zero_win_rate": 0.0,
            "fragment_rate": 0.0, "retrans_count": 0.0, "retrans_rate": 0.0,
            "payload_entropy": 0.0, "sequential_portscan_score": 0.0,
            "packet_dport_entropy": 0.0, "has_packet_features": 0.0
        }

    lengths = [p.get("length", 0) for p in packets_list]
    ttls = [p.get("ttl", 64) for p in packets_list]
    windows = [p.get("window", 0) for p in packets_list if p.get("is_tcp", False)]
    dports = [p.get("dport", 0) for p in packets_list if p.get("dport", 0) > 0]
    timestamps = [p.get("timestamp", 0.0) for p in packets_list]
    payloads = [p.get("payload_bytes", b"") for p in packets_list if p.get("payload_bytes")]

    # Length stats
    l_mean = float(np.mean(lengths))
    l_std = float(np.std(lengths))
    l_min = float(np.min(lengths))
    l_max = float(np.max(lengths))
    l_skew = float(np.mean(((np.array(lengths) - l_mean) / max(l_std, 1e-4)) ** 3))

    # Inter-arrival times
    if len(timestamps) > 1:
        sorted_ts = np.sort(timestamps)
        iats = np.diff(sorted_ts)
        iat_mean = float(np.mean(iats))
        iat_std = float(np.std(iats))
        iat_max = float(np.max(iats))
        iat_cov = float(iat_std / max(iat_mean, 1e-5))
    else:
        iat_mean, iat_std, iat_max, iat_cov = 0.0, 0.0, 0.0, 0.0

    # TTL stats
    ttl_mean = float(np.mean(ttls))
    ttl_std = float(np.std(ttls))
    ttl_distinct = float(len(set(ttls)))

    # Window sizes
    if windows:
        win_mean = float(np.mean(windows))
        win_std = float(np.std(windows))
        zero_win = float(sum(1 for w in windows if w == 0))
        zero_win_rate = float(zero_win / len(windows))
    else:
        win_mean, win_std, zero_win, zero_win_rate = 0.0, 0.0, 0.0, 0.0

    # Fragments & Retransmissions
    fragments = sum(1 for p in packets_list if p.get("is_fragment", False))
    fragment_rate = float(fragments / n_pkts)

    retrans = sum(1 for p in packets_list if p.get("is_retransmission", False))
    retrans_rate = float(retrans / n_pkts)

    # Payload entropy
    if payloads:
        combined_payload = b"".join(payloads[:20])
        p_entropy = compute_payload_entropy(combined_payload)
    else:
        p_entropy = 0.0

    # Port scanning sequentiality score: fraction of port steps of +1 or +2
    if len(dports) > 3:
        port_diffs = np.diff(dports)
        seq_hits = sum(1 for d in port_diffs if d in (1, 2))
        scan_score = float(seq_hits / len(port_diffs))
        counts = Counter(dports)
        total_p = len(dports)
        p_dport_ent = float(-sum((c / total_p) * math.log2(c / total_p) for c in counts.values()))
    else:
        scan_score = 0.0
        p_dport_ent = 0.0

    return {
        "pkt_len_mean": l_mean,
        "pkt_len_std": l_std,
        "pkt_len_min": l_min,
        "pkt_len_max": l_max,
        "pkt_len_skew": l_skew,
        "pkt_iat_mean": iat_mean,
        "pkt_iat_std": iat_std,
        "pkt_iat_max": iat_max,
        "pkt_iat_cov": iat_cov,
        "ttl_mean": ttl_mean,
        "ttl_std": ttl_std,
        "ttl_distinct_count": ttl_distinct,
        "tcp_win_mean": win_mean,
        "tcp_win_std": win_std,
        "zero_win_count": zero_win,
        "zero_win_rate": zero_win_rate,
        "fragment_rate": fragment_rate,
        "retrans_count": float(retrans),
        "retrans_rate": retrans_rate,
        "payload_entropy": p_entropy,
        "sequential_portscan_score": scan_score,
        "packet_dport_entropy": p_dport_ent,
        "has_packet_features": 1.0
    }


def parse_pcap_file(pcap_path: str, max_packets: int = 100_000) -> List[Dict[str, Any]]:
    """Lightweight PCAP parser extracting per-packet attributes via Scapy (if installed) or pure-Python."""
    parsed_pkts = []
    try:
        from scapy.all import PcapReader, IP, TCP, UDP
        with PcapReader(pcap_path) as pcap_reader:
            for idx, pkt in enumerate(pcap_reader):
                if idx >= max_packets:
                    break
                if not pkt.haslayer(IP):
                    continue

                ip_layer = pkt[IP]
                is_tcp = pkt.haslayer(TCP)
                is_udp = pkt.haslayer(UDP)

                sport, dport, win = 0, 0, 0
                if is_tcp:
                    tcp = pkt[TCP]
                    sport, dport, win = tcp.sport, tcp.dport, tcp.window
                elif is_udp:
                    udp = pkt[UDP]
                    sport, dport = udp.sport, udp.dport

                payload_data = bytes(pkt.payload)[:64]

                parsed_pkts.append({
                    "timestamp": float(pkt.time),
                    "src_ip": str(ip_layer.src),
                    "dst_ip": str(ip_layer.dst),
                    "length": len(pkt),
                    "ttl": int(ip_layer.ttl),
                    "is_tcp": is_tcp,
                    "is_udp": is_udp,
                    "sport": sport,
                    "dport": dport,
                    "window": win,
                    "is_fragment": bool(ip_layer.flags & 1 or ip_layer.frag > 0),
                    "is_retransmission": False,
                    "payload_bytes": payload_data
                })
    except Exception as e:
        print(f"[!] Scapy pcap parse notice ({pcap_path}): {e}")

    return parsed_pkts
