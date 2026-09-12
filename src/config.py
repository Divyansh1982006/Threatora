"""Central configuration for Threatora Attack Forecasting World Model.

Implements NTRO PS 26153 specifications:
  - 60-second host-window state aggregation
  - 16-window rolling context (16 minutes)
  - 10-window forecast horizon (10-minute forward simulation)
  - 62 total features (39 flow-level + 23 packet-level)
  - LSTM-based Recurrent State Space Model (RSSM) dynamics
  - Checkpoint pipeline for external laptop training integration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

# --------------------------------------------------------------------------
# Paths & Directories
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for _dir in (RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, CHECKPOINT_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Cadence & Windowing
# --------------------------------------------------------------------------

WINDOW_SECONDS = 60.0        # 60s host-window cell (S_t)
SEQUENCE_LENGTH = 16         # 16 consecutive windows (16-minute rolling context)
FORECAST_HORIZON = 10        # K = 10 windows lookahead (10-minute simulation)
LOOKAHEAD_WINDOWS = 5        # Detection of upcoming attack within 5 minutes
MIN_FLOWS_PER_CELL = 2       # Minimum flows required to form a valid state cell
USE_PACKET_FEATURES = True   # Enable dual-tier packet + flow features

# --------------------------------------------------------------------------
# Network Address & Port Taxonomy (CTU-13 / Enterprise Monitored Ranges)
# --------------------------------------------------------------------------

INTERNAL_PREFIXES = ("147.32.",)
LATERAL_PORTS = frozenset({135, 137, 138, 139, 445, 3389, 5985, 5986, 22, 23})
C2_PORTS = frozenset({6667, 6668, 6669, 7000, 1863, 8080, 8000, 443, 53})

# CTU-13 scenario -> malware family mappings and splits
CTU13_FAMILIES = {
    1: "Neris", 2: "Neris", 3: "Rbot", 4: "Rbot", 5: "Virut",
    6: "Menti", 7: "Sogou", 8: "Murlo", 9: "Neris", 10: "Rbot",
    11: "Rbot", 12: "NSIS.ay", 13: "Virut",
}

TRAIN_SCENARIOS = (1, 2, 3, 4, 6, 7, 9, 10, 11)   # Neris, Rbot, Menti, Sogou
VAL_SCENARIOS = (12,)                              # NSIS.ay - unseen family
TEST_SCENARIOS = (5, 8, 13)                        # Virut, Murlo - unseen families

# --------------------------------------------------------------------------
# 62 Features Definition (39 Flow + 23 Packet)
# --------------------------------------------------------------------------

FLOW_FEATURE_COLS: List[str] = [
    # 1. Flow & Connection Volume (8)
    "n_flows", "n_unique_dst", "n_unique_dport", "n_unique_sport",
    "tot_bytes", "tot_pkts", "src_bytes", "dst_bytes",
    # 2. Flow Rates & Averages (6)
    "bytes_per_sec", "pkts_per_sec", "avg_pkt_size", "bytes_per_flow", "pkts_per_flow", "egress_ratio",
    # 3. Direction & State Breakdown (5)
    "frac_inbound", "frac_outbound", "frac_established", "frac_reset", "frac_interrupted",
    # 4. TCP Flags (Expanded) (7)
    "frac_syn_only", "frac_syn_ack", "frac_fin", "frac_rst", "frac_psh", "frac_urg", "frac_ack",
    # 5. Timing, Intervals & Beacons (6)
    "dur_mean", "dur_std", "dur_max", "flow_iat_mean", "flow_iat_std", "beacon_regularity",
    # 6. Protocol & Service Indicators (4)
    "frac_tcp", "frac_udp", "frac_icmp", "dns_query_count",
    # 7. Information Entropy (3)
    "dst_ip_entropy", "dport_entropy", "sport_entropy"
]

PACKET_FEATURE_COLS: List[str] = [
    # 1. Packet Size Statistics (5)
    "pkt_len_mean", "pkt_len_std", "pkt_len_min", "pkt_len_max", "pkt_len_skew",
    # 2. Time Inter-Arrival (IAT) (4)
    "pkt_iat_mean", "pkt_iat_std", "pkt_iat_max", "pkt_iat_cov",
    # 3. TTL & Hop Dynamics (3)
    "ttl_mean", "ttl_std", "ttl_distinct_count",
    # 4. TCP Window & Flow Control (4)
    "tcp_win_mean", "tcp_win_std", "zero_win_count", "zero_win_rate",
    # 5. Fragmentation & Retransmissions (3)
    "fragment_rate", "retrans_count", "retrans_rate",
    # 6. Payload & Protocol Attributes (4)
    "payload_entropy", "sequential_portscan_score", "packet_dport_entropy", "has_packet_features"
]

ALL_FEATURE_COLS: List[str] = FLOW_FEATURE_COLS + PACKET_FEATURE_COLS

assert len(FLOW_FEATURE_COLS) == 39, f"Expected 39 flow features, got {len(FLOW_FEATURE_COLS)}"
assert len(PACKET_FEATURE_COLS) == 23, f"Expected 23 packet features, got {len(PACKET_FEATURE_COLS)}"
assert len(ALL_FEATURE_COLS) == 62, f"Expected 62 total features, got {len(ALL_FEATURE_COLS)}"

# --------------------------------------------------------------------------
# Model Architecture & Hyperparameters
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    obs_dim: int = 62            # Total observation vector dimension
    embed_dim: int = 64          # Observation encoder output dimension e_t
    hidden_dim: int = 128        # LSTM hidden state (h_t) and cell state (c_t)
    num_stages: int = 5          # 5 MITRE ATT&CK stages
    n_heads: int = 4             # Causal attention heads
    dropout: float = 0.15        # Regularization dropout
    learning_rate: float = 8e-4  # Optimizer learning rate
    weight_decay: float = 1e-4   # L2 weight penalty
    recon_weight: float = 1.0    # Weight for MSE state reconstruction
    cls_weight: float = 2.0      # Weight for infiltration binary cross-entropy
    stage_weight: float = 1.5    # Weight for MITRE stage cross-entropy
    monte_carlo_rollouts: int = 16  # Rollout trajectories for uncertainty bands


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    max_content_length: int = 64 * 1024 * 1024  # 64 MB upload limit


default_model_config = ModelConfig()
default_server_config = ServerConfig()
