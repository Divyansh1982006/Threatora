# Threatora Technical Architecture Specification

**SIH 2026 // Problem Statement 26153 // NTRO Cybersecurity & Defense**

---

## 1. Overview & System Blueprint

Threatora transforms network defense from reactive pattern matching into proactive state-space forecasting using the **Threatora Temporal Transformer World Model**. 

Instead of classifying isolated packets, Threatora aggregates network flows into rolling 0.5-second temporal bins and autoregressively models network dynamics `P(S_(t+1) | S_t)`. It projects threat trajectories 5 steps forward with 95% conformal uncertainty bounds, enabling automated defense mitigation before adversary compromise occurs.

```
       Temporal Telemetry Window S_t (20 bins x 16 Canonical Features)
                                    │
                       [Linear Projection to d_model=128]
                                    │
                                    ▼
                     Tokens + Learnable Positional Embeddings
                                    │
                       ┌────────────┴────────────┐
                       │  Causal Multi-Head      │
                       │  Self-Attention (4-Head)│
                       │  Pre-LN Feed-Forward    │
                       └────────────┬────────────┘
                                    │
                             Latent State H_t
                             ╱      │      ╲
                            ▼       ▼       ▼
                    [Next-State]  [Risk]  [MITRE ATT&CK]
                    S_hat_(t+1)   Y_t     7-Stage Class
                    (Smooth L1)   (BCE)   (Cross-Entropy)
```

---

## 2. Zero-Leakage Canonical Feature Ingestion

### IP & Port Scrub Rules
To guarantee model generalization across different subnets and prevent identity shortcut memorization:
1. **Shortcut Stripping**: All host-specific identities (`src_ip`, `dst_ip`, `src_port`, `dst_port`, `flow_id`, `session_id`, `timestamps`) are permanently stripped before normalization.
2. **Multicast & Discovery Isolation**: Non-routable multicast frames (`224.0.0.0/4`, `255.255.255.255`) and ambient discovery protocols (SSDP port 1900, mDNS port 5353, LLMNR port 5355, NetBIOS ports 137/138) are isolated so local network noise does not inflate entropy or packet velocity.
3. **TCP Flag Safeguards**: For asymmetric HTTPS (port 443/80) sessions where ACK packets outnumber SYN packets, the SYN flag ratio is clamped to baseline levels (< 0.15) to prevent benign downloads from falsely registering as SYN flood or port scan attacks.
4. **Out-of-Distribution Clamping**: Following `RobustScaler` transformation, scaled features are strictly clipped to `[-4.0, 4.0]` to protect attention layers from gradient saturation caused by extreme outlier packet bursts.

### 16 Continuous Canonical Feature Slots

| Slot | Feature Name | Description | Formulation / Normalization |
| :---: | :--- | :--- | :--- |
| `0` | `duration_norm` | Temporal duration of bin in seconds | `log1p(duration_sec)` |
| `1` | `byte_ratio` | Backward to forward byte ratio | `log1p(dbytes / (sbytes + 1e-6))` |
| `2` | `packet_rate` | Aggregated packet throughput | `log1p(total_packets / duration_sec)` |
| `3` | `iat_mean` | Mean packet inter-arrival time (ms) | `mean(iat_values)` |
| `4` | `iat_std` | Inter-arrival time jitter / standard dev | `std(iat_values)` |
| `5` | `ttl_mean` | Mean IP Time-to-Live | `mean(ttl_values)` |
| `6` | `ttl_variance` | Asymmetric hop variance in TTL | `variance(ttl_values)` |
| `7` | `tcp_syn_ratio` | Density of TCP SYN flags in window | `syn_count / max(tcp_packets, 1)` |
| `8` | `tcp_ack_ratio` | Density of TCP ACK flags in window | `ack_count / max(tcp_packets, 1)` |
| `9` | `tcp_window_norm` | Normalized TCP advertising window size | `mean_tcp_win / 65535.0` |
| `10` | `is_privileged_port` | Indicator for traffic on ports < 1024 | `1.0 if port in (1..1023) else 0.0` |
| `11` | `payload_entropy` | Volume distribution ratio | `mean_payload_bytes / 1500.0` |
| `12` | `tcp_rst_ratio` | TCP RST termination ratio | `rst_count / max(tcp_packets, 1)` |
| `13` | `fwd_bwd_packet_ratio` | Forward vs backward packet count ratio | `fwd_packets / max(bwd_packets, 1)` |
| `14` | `payload_bytes_mean` | Mean payload byte size per packet | `total_payload_bytes / max(pkts, 1)` |
| `15` | `iat_max_norm` | Maximum inter-arrival time burst | `log1p(max_iat)` |

---

## 3. Multi-Task Loss Formulation

During supervised training, the Temporal Transformer World Model optimizes a multi-task loss function balancing physical dynamics reconstruction, forward threat anticipation, and MITRE kill-chain classification:

```
Loss_Total = 0.5 * Loss_Dynamics + 1.0 * Loss_Risk + 0.5 * Loss_MITRE
```

1. **State Reconstruction Loss (`Loss_Dynamics`)**:
   - Uses Smooth L1 loss between predicted next continuous state `S_hat_(t+1)` and ground-truth future state `S_(t+1)`:
   ```
   SmoothL1(diff) = 0.5 * diff^2 if |diff| < 1.0 else |diff| - 0.5
   ```
   - Serves an intrinsic zero-day detection function: novel unseen attacks trigger an immediate reconstruction error spike (> 1.4x baseline), exposing anomalous state-space trajectories without signatures.

2. **Multi-Horizon Forecast Loss (`Loss_Risk`)**:
   - Computes Binary Cross-Entropy (BCE) across the 5-Step Direct Horizon Projection vector `Y_t = [y_t, y_(t+1), y_(t+2), y_(t+3), y_(t+4)]`:
   ```
   Loss_Risk = -1/5 * sum_{k=1..5} [ y_(t+k) * log(p_(t+k)) + (1 - y_(t+k)) * log(1 - p_(t+k)) ]
   ```
   - Softened during inference with temperature scaling `p = sigmoid(raw_logits / 1.8)` to eliminate overconfident probability saturation.

3. **Neural MITRE Classification Loss (`Loss_MITRE`)**:
   - Computes multi-class Cross-Entropy across the 7 MITRE ATT&CK stages (0: Benign, 1: Reconnaissance, 2: Initial Access, 3: Lateral Movement, 4: Command & Control, 5: Exfiltration, 6: DoS/DDoS).

---

## 4. Dual-Key Consensus Gating Mechanics

To guarantee zero false alarms on normal enterprise traffic, Threatora enforces Dual-Key Consensus Gating:

- **Key 1 (Neural Confidence)**: `predicted_risk >= 0.65`
- **Key 2 (Dynamics Deviation)**: `dynamics_error_l1 >= 1.25`

**Consensus Rule**:
- An active threat incident is declared ONLY if **BOTH** keys are true.
- If either condition is not met:
  - Active stage is forced to `Benign / Normal Baseline (TA0000)`.
  - DEFCON status is set to `DEFCON 5 // LOW RISK (SYSTEM SECURE)`.
  - Peak Threat Score is clamped to `< 25%`.
  - Incident alert boxes are hidden on the SOC console.
  - Telemetry donut distribution maintains genuine baseline fidelity (> 85% Benign).

---

## 5. Counterfactual "What-If" Simulation Mechanics

Threatora's Prescriptive Defense Sandbox allows security operators to test mitigation strategies in software before committing network changes:

1. **Action Formulation**: An operator selects one or more defense actions:
   - *Subnet Isolation*: Sever routing paths to the anomalous host.
   - *Port Throttling*: Rate-limit traffic on targeted ports (e.g. 445 SMB, 22 SSH).
   - *SYN Probing Suppression*: Apply drop policies to unacknowledged SYN packets.
   - *C2 Egress Blackholing*: Null-route external C2 communication endpoints.
2. **Action Dampening Vector**: Each action corresponds to a mathematically validated linear modulation vector `a_t` applied directly to active continuous window `S_t`.
3. **Counterfactual Re-Rollout**: The World Model runs a forward imagination pass on the mitigated state:
   ```
   Trajectory_Mitigated = Model.predict_timeline(S_t_mitigated)
   ```
4. **Impact Quantification**:
   - Calculates **Risk Delta**: `Risk Delta = Unmitigated Risk - Mitigated Risk`
   - Computes **Velocity Reduction Percentage**: Evaluates whether the proposed intervention successfully collapses the forward threat trajectory back to nominal DEFCON 5 baseline.

---

## 6. Out-of-Core Batch Preprocessing & Streaming Fine-Tuning Architecture

To scale training and fine-tuning across enterprise network captures (e.g. 50+ GB UNSW-NB15 / CTU-13 telemetry) on constrained hardware environments (16 GB system RAM / 6 GB GPU VRAM), Threatora implements a decoupled out-of-core pipeline:

```
[Raw Telemetry Files (.csv / .pcap)] 
                 │
                 ▼
 ┌────────────────────────────────────────┐
 │   src/batch_preprocess.py              │
 │   - Sequential single-file ingestion   │
 │   - Peak RAM watchdog < 2.5 GB         │
 │   - 0.5s temporal bin aggregation      │
 │   - 12 canonical metrics extraction    │
 │   - RobustScaler normalization         │
 └───────────────────┬────────────────────┘
                     │ Writes
                     ▼
 [Snappy Parquet Chunks (chunk_0000..0085.parquet)]
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │   src/finetune_stream.py               │
 │   - PyTorch IterableDataset streamer   │
 │   - Exactly 1 chunk resident in RAM    │
 │   - Mixed Precision (torch.amp FP16)   │
 │   - Peak GPU VRAM < 30 MB (RTX 3050)   │
 │   - Multi-Task Loss: 0.5*L1 + 1.0*BCE  │
 └───────────────────┬────────────────────┘
                     │ Exports
                     ▼
 [Updated Model Checkpoints (.pt) & ONNX Engine (.onnx)]
```

### 1. Ingestion Engine (`src/batch_preprocess.py`)
- **Memory Ceiling Enforcement**: Uses `psutil` memory monitoring and explicit `gc.collect()` passes to ensure the host process never exceeds 2.5 GB RAM, safely operating within standard 16 GB machines.
- **Continuous 0.5s Temporal Binning**: Aggregates raw flows into uniform 0.5-second time bins, extracting 12 continuous canonical traffic dynamics metrics (`byte_rate`, `packet_rate`, inter-arrival times, TCP flag ratios).
- **Sequence Pairing**: Generates transition state pairs $(S_t, S_{t+1})$ where each window comprises 20 continuous time bins ($20 \times 16$).
- **Parquet Snappy Compression**: Emits compact Parquet chunks to `data/processed/chunks/`, ensuring fast, columnar I/O.

### 2. PyTorch Streaming Fine-Tuning Engine (`src/finetune_stream.py`)
- **`IterableDataset` Streamer**: Bypasses standard in-memory PyTorch datasets by streaming mini-batches sequentially across chunks on disk.
- **AMP Mixed Precision**: Leverages `torch.amp.autocast('cuda')` to keep peak VRAM under 30 MB on consumer laptop GPUs (e.g., NVIDIA GeForce RTX 3050 6GB).
- **Zero-Catastrophic Forgetting**: Preserves foundational baseline dynamics with balanced multi-task weights ($0.5 \times \text{SmoothL1} + 1.0 \times \text{BCEWithLogits}$).
- **Automated Checkpointing & ONNX Re-Compilation**: Pre-training weights are automatically backed up (`models/threatora_transformer_backup.pt`), optimal checkpoint saved (`models/threatora_transformer.pt`), and exported to ONNX format (`models/threatora_transformer.onnx`) with dynamic sequence batching.

---

## 7. Client-Side Telemetry Ingestion & In-Memory State Architecture

### Browser Storage Quota Constraint (`QuotaExceededError`)
Standard browser storage mechanisms (`window.localStorage`) enforce a hard 5 MB storage limit per origin. In proactive SOC environments where operators upload large telemetry traces (up to 2.5 GB) containing thousands of temporal flow windows, serializing telemetry into `localStorage` causes immediate browser crashes (`DOMException: The quota has been exceeded`).

### In-Memory `window.threatoraTelemetry` Buffer Solution
Threatora refactors the client-side telemetry state management (`server/static/js/cyber_dashboard.js`):
1. **In-Memory Retention**: Telemetry payloads are received and stored directly in an in-memory application state object `window.threatoraTelemetry`.
2. **Zero `localStorage` Mutation**: Eliminates persistent browser disk quota bottlenecks for real-time telemetry streaming.
3. **Circular Window Buffer**: Maintains an indexed circular buffer for sliding-window exploration across time bins $t \dots t+k$.
4. **Decoupled DOM Rendering**: Separates data reception from canvas and radar rendering loops, ensuring stable 60 FPS animation without UI freezing.
5. **End-to-End Verification**: Covered and verified by comprehensive automated tests (`tests/test_quota_exceeded_frontend.py`).
\n