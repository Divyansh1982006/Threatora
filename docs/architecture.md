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
