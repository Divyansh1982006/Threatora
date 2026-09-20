# Threatora Technical Architecture Specification

**SIH 2026 · Problem Statement 26153 · NTRO · Blockchain & Cybersecurity**

## Overview

Threatora is a proactive cyber defense system that replaces static flow classification with an **LSTM-based Recurrent State-Space World Model (RSSM)**. By learning the continuous state transition dynamics of network traffic — $P(S_{t+1} \mid S_t)$ — over 60-second window cells, the system rolls forward future trajectories without observations to forecast attacker kill-chain progression up to 10 minutes before full compromise.

```
       Observation x_t (62 Features)
                   │
            [Encoder MLP]
                   │
                   ▼
       Encoded Telemetry e_t
                   │
(h_t, c_t) = LSTM([h_t-1, z_t-1], c_t-1)  ◄── Deterministic Context & Memory
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
Posterior q(z_t | h_t, e_t)   Prior p(z_t | h_t)
(Observed Phase)              (Imagination Phase)
    │                             │
    └──────────────┬──────────────┘
                   ▼
         S_t = [h_t ; z_t]
          ╱        │        ╲
   [Decoder]    [Heads]   [Causal Attention]
   \hat{x}_t    • Risk    (Masked history weights)
                • MITRE
```

---

## 1. Feature Space (62 Features)

### Flow-Level Attributes (39 Features)
- **Volume & Diversity (8)**: `n_flows`, `n_unique_dst`, `n_unique_dport`, `n_unique_sport`, `tot_bytes`, `tot_pkts`, `src_bytes`, `dst_bytes`
- **Rates & Averages (6)**: `bytes_per_sec`, `pkts_per_sec`, `avg_pkt_size`, `bytes_per_flow`, `pkts_per_flow`, `egress_ratio`
- **Direction & Argus State (5)**: `frac_inbound`, `frac_outbound`, `frac_established`, `frac_reset`, `frac_interrupted`
- **TCP Flags Expanded (7)**: `frac_syn_only`, `frac_syn_ack`, `frac_fin`, `frac_rst`, `frac_psh`, `frac_urg`, `frac_ack`
- **Timing & Beacons (6)**: `dur_mean`, `dur_std`, `dur_max`, `flow_iat_mean`, `flow_iat_std`, `beacon_regularity`
- **Protocols & Services (4)**: `frac_tcp`, `frac_udp`, `frac_icmp`, `dns_query_count`
- **Entropy (3)**: `dst_ip_entropy`, `dport_entropy`, `sport_entropy`

### Packet-Level Attributes (23 Features)
- **Size Distribution (5)**: `pkt_len_mean`, `pkt_len_std`, `pkt_len_min`, `pkt_len_max`, `pkt_len_skew`
- **Timing Dynamics (4)**: `pkt_iat_mean`, `pkt_iat_std`, `pkt_iat_max`, `pkt_iat_cov`
- **TTL & Hop Statistics (3)**: `ttl_mean`, `ttl_std`, `ttl_distinct_count`
- **TCP Flow Control (4)**: `tcp_win_mean`, `tcp_win_std`, `zero_win_count`, `zero_win_rate`
- **Fragmentation & Loss (3)**: `fragment_rate`, `retrans_count`, `retrans_rate`
- **Payload & Sequencing (4)**: `payload_entropy`, `sequential_portscan_score`, `packet_dport_entropy`, `has_packet_features`

---

## 2. LSTM Recurrent State Space Model (RSSM)

### LSTM Recurrence vs Standard GRU
In network attack forecasting, attacks often span multi-minute dormancy periods (e.g., waiting between reconnaissance scanning and payload download). An **LSTM transition cell** preserves long-term state across cells via its cell state vector $c_t$, avoiding context vanishing:
$$(h_t, c_t) = \text{LSTM}([h_{t-1}, z_{t-1}], c_{t-1})$$

### Forward Simulation Rollout (`.imagine()`)
During imagination, no observations exist. The model rolls forward purely using its learned transition prior:
$$\text{For } k = 1 \dots K: \quad z_{t+k} \sim p(z_{t+k} \mid h_{t+k-1}), \quad (h_{t+k}, c_{t+k}) = \text{LSTM}([h_{t+k-1}, z_{t+k}], c_{t+k-1})$$
Predictions are generated directly from dreamed states $S_{t+k} = [h_{t+k}; z_{t+k}]$.

---

## 3. MITRE ATT&CK Stages

| Stage | Name | Tactic ID | Example Indicators |
| :---: | :--- | :---: | :--- |
| **0** | Benign | TA0000 | Normal baseline traffic |
| **1** | Reconnaissance | TA0043 | Rapid SYN probing, high port entropy, low byte/packet ratio |
| **2** | Initial Access | TA0001 | Public web exploit, binary download burst |
| **3** | Lateral Movement | TA0008 | Internal port 445/SMB fan-out across workstations |
| **4** | Command & Control | TA0011 | Periodic high beacon regularity, IRC/HTTP heartbeats |
| **5** | Exfiltration | TA0010 | Sustained large outbound byte volume, anomalous egress ratio |

---

## 4. Multi-Task Objective Function

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + w_{\text{cls}} \mathcal{L}_{\text{infilt}} + w_{\text{stage}} \mathcal{L}_{\text{stage}}$$
Where:
- $\mathcal{L}_{\text{recon}} = \frac{1}{D} \|\hat{x}_t - x_t\|^2$
- $\mathcal{L}_{\text{KL}} = \text{KL}(q(z_t \mid h_t, e_t) \parallel p(z_t \mid h_t))$
- $\mathcal{L}_{\text{infilt}} = -\left[ y_t \log \hat{y}_t + (1 - y_t) \log (1 - \hat{y}_t) \right]$
- $\mathcal{L}_{\text{stage}} = -\sum_{s=0}^5 \mathbb{I}(y_{\text{stage}}=s) \log P(s)$

For the Temporal Transformer World Model, the composite loss is:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{SmoothL1}}(\hat{S}_{t+1}, S_{t+1}) + 1.0 \cdot \mathcal{L}_{\text{BCE}}(\hat{Y}_t, Y_t) + 0.5 \cdot \mathcal{L}_{\text{CE}}(\hat{M}_t, M_t)$$

---

## 5. 16-Slot Zero-Leakage Canonical Representation

To guarantee mathematical independence from host network topology and prevent identity shortcut learning, all packet and flow data are mapped into a 16-dimensional continuous state vector:

| Slot | Feature Name | Description | Physical Source |
| :---: | :--- | :--- | :--- |
| `0` | `duration_norm` | Flow / bin temporal duration | $\text{dur} \in [0, \infty)$ |
| `1` | `byte_ratio` | Backward to forward byte ratio | $\text{dbytes} / (\text{sbytes} + 1\mu)$ |
| `2` | `packet_rate` | Aggregate packets per second | $(\text{spkts} + \text{dpkts}) / (\text{dur} + 1\mu)$ |
| `3` | `iat_mean` | Mean packet inter-arrival time (ms) | $(\text{sintpkt} + \text{dintpkt}) / 2.0$ |
| `4` | `iat_std` | Inter-arrival time jitter / variance | $(\text{sjit} + \text{djit}) / 2.0$ |
| `5` | `ttl_mean` | Mean IP Time-to-Live (hops) | $(\text{sttl} + \text{dttl}) / 2.0$ |
| `6` | `ttl_variance` | TTL asymmetric hop variance | $(\text{sttl} - \text{dttl})^2 / 4.0$ |
| `7` | `tcp_syn_ratio` | TCP SYN flag density in window | $\mathbb{I}(\text{SYN}) / N_{\text{pkts}}$ |
| `8` | `tcp_ack_ratio` | TCP ACK flag density in window | $\mathbb{I}(\text{ACK}) / N_{\text{pkts}}$ |
| `9` | `tcp_window_norm` | Normalized TCP advertising window | $(\text{swin} + \text{dwin}) / 131,070$ |
| `10` | `is_privileged_port` | Ingress privileged port indicator | $\mathbb{I}(0 < \text{dsport} < 1024) \in \{0.0, 1.0\}$ |
| `11` | `payload_entropy` | Normalized payload volume / entropy | $\text{res\_bdy\_len} / (\text{res\_bdy\_len} + 1500)$ |
| `12` | `tcp_rst_ratio` | TCP RST connection termination ratio | $\mathbb{I}(\text{RST}) / N_{\text{pkts}}$ |
| `13` | `fwd_bwd_packet_ratio`| Forward to backward packet imbalance | $\text{clip}(\text{spkts} / (\text{dpkts} + 1\mu), 0, 1000)$ |
| `14` | `payload_bytes_mean` | Mean payload bytes per packet | $\text{clip}((\text{sbytes} + \text{dbytes}) / (\text{spkts} + \text{dpkts} + 1\mu), 0, 65535)$ |
| `15` | `iat_max_norm` | Normalized maximum inter-arrival burst | $\ln(1 + \max(\text{sintpkt}, \text{dintpkt}))$ |

---

## 6. Full-Stack Ingestion & Dashboard Architecture

### 1. 2.5 GB Chunked Ingestion Pipeline (`/api/upload-chunk`)
```
[Client Browser] 
      │
      ▼ (10 MB Multipart Slices: file, filename, chunk_index, total_chunks)
[POST /api/upload-chunk]
      │
      ├── Append slice to uploads/{filename}.part
      │
      └── If chunk_index == total_chunks - 1:
            ├── Atomic rename: uploads/{filename}.part ➔ uploads/{filename}
            └── Trigger Low-Memory Parser & World Model Ingestion
```

### 2. Low-Memory Streaming Parser
- **PCAP / PCAPNG**: `FastPCAPParser` uses 4 MB chunk buffers and zero-copy `struct.unpack_from` unpacking (or `scapy.PcapReader`) to extract flow fields (`timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `packet_size`, `flags`) without loading 2.5 GB payloads into RAM.
- **NetFlow CSV / TSV**: Polars lazy frame evaluation (`pl.scan_csv`) streams and aggregates temporal 0.5s bins with zero data leakage.

### 3. Server-Sent Events (SSE) Engine (`/api/stream-telemetry`)
Broadcasts real-time attack forecasting horizons directly to client listeners:
- `risk_score`: Lookahead horizon risk projection
- `entropy`: Payload and port distribution entropy
- `target_node`: Active compromised or targeted endpoint
- `mitre_stage`: Active kill-chain phase (0..5)
- `packet_velocity`: Live ingestion throughput (pkts/sec)
- `timestamp`: Event timestamp (ISO 8601)

### 4. Interactive 3D Canvas Radar & Defense Terminal UI
- **Radar Canvas (`static/js/topology.js`)**: Real-time rendering of concentric range rings, rotating sweep arms, orbiting nodes with threat halos, dynamic packet pulses along communication edges, and full mouse drag-to-rotate 3D perspective controls.
- **Cyberpunk Dark Theme (`static/css/cyberpunk.css`, `static/css/style.css`)**: Dark terminal palette (`#0a0b0e` canvas, `#14171d` card containers, `#1e232d` borders) with Glowing Amber (`#f59e0b`), Breach Red (`#ef4444`), Terminal Cyan (`#06b6d4`), and Online Emerald (`#10b981`) accents.
- **Zero-Trust Role Elevation Modal (`static/js/app.js`)**: Intercepts unauthorized or elevated operations with an interactive 403 modal, allowing operators to authenticate up to Chief CISO (Level 5).

### 5. Prescriptive Defense Sandbox ("What-If")
Enables security operators to evaluate simulated interventions before applying them on production networks:
- **Actions**: "Isolate Source Subnet", "Throttle Privileged Ports", "Rate-Limit SYN Probing", "Block C2 Outbound IP".
- **Mechanism**: Modulates active continuous state window $S_t$ with action dampener vectors and re-evaluates forward trajectory $P(S_{t+k} \mid S_t, a_t)$ via the World Model to calculate projected risk reduction percentage.

### 6. Zero-Trust RBAC & State Store Ledger
- **Clearance Levels**:
  - `Level 1`: Guest / Observer
  - `Level 2`: SOC Analyst
  - `Level 3`: Senior SOC
  - `Level 4`: SecOps Lead
  - `Level 5`: Chief CISO
- **Persistence Ledger**: SQLAlchemy models backed by PostgreSQL (production) or SQLite (air-gapped local) tracking security incidents, asset status, and mitigation playbooks in `artifacts/data/threatora.db`.

