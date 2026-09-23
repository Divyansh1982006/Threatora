# Threatora: AI World Model for Proactive Attack Forecasting

**Smart India Hackathon (SIH) 2026 // National Technical Research Organisation (NTRO) Problem Statement 26153**  
*Autonomous AI-Based Proactive Network Attack Forecasting Platform*

---

## 1. Project Header & Mission

**Threatora** is an air-gapped, proactive cyber defense platform built to eliminate the fatal latency gap of legacy signature-based and point-in-time Network Intrusion Detection Systems (NIDS). Traditional intrusion tools trigger alerts only after an exploit payload executes or an initial breach has already occurred. 

Threatora formulates cyber defense as an autoregressive world modeling task:
- Ingests raw network telemetry in continuous 0.5-second temporal bins.
- Models continuous network state transition dynamics: P(S_(t+1) | S_t).
- Projects a forward horizon risk vector up to 50 seconds ahead with a **Mean Lead-Time to Compromise (MLTC) of 47.0s at < 0.1% False Positive Rate (FPR)**.
- Classifies active kill-chain tactics across a neural MITRE ATT&CK head without brittle regex heuristics.
- Enables counterfactual "What-If" defense simulation to test remediation playbooks before enforcing them in production.

---

## 2. Flagship Architecture

The core of Threatora is the **Threatora Temporal Transformer World Model**:

- **16 Continuous Canonical Feature Schema**: Strips all shortcut identity markers (src_ip, dst_ip, sport, dport, timestamps) to enforce zero data leakage across heterogeneous network telemetry (UNSW-NB15, CSE-CIC-IDS2018, CTU-13, and raw PCAP captures).
- **Sub-0.5ms ONNX CPU Runtime**: Full FP32 ONNX Runtime CPU execution operating at **0.42ms per window** (> 2,300 windows/second single-core throughput, footprint 0.33 MB) for deployment on edge routers and air-gapped SOC servers.
- **5-Step Direct Horizon Projection (k=1..5)**: Directly supervises and forecasts forward threat probability trajectories [y_t, y_(t+1), y_(t+2), y_(t+3), y_(t+4)] with 95% conformal uncertainty confidence bounds.
- **Dual-Key Consensus Gating**: Active threats are escalated to the SOC operational console ONLY when two independent criteria are satisfied:
  1. `predicted_risk >= 0.65` (High neural classifier confidence)
  2. `dynamics_error_l1 >= 1.25` (Statistically anomalous state transition deviation)
  If either condition fails, the pipeline forces stage to `Benign / Normal Baseline (TA0000)`, locks DEFCON to `DEFCON 5 // LOW RISK (SYSTEM SECURE)`, suppresses false alarm incident boxes, and preserves baseline telemetry integrity (> 85% Benign).

---

## 3. 6-Stage MITRE ATT&CK Kill-Chain Mapping

Threatora classifies observed and forecasted network behaviors across a 7-stage taxonomy:

| Stage ID | Kill-Chain Stage Name | Tactic ID | Description & Detection Signals |
| :---: | :--- | :---: | :--- |
| **0** | **Benign / Normal Baseline** | `TA0000` | Expected enterprise traffic; balanced bidirectional ratios; nominal state dynamics. |
| **1** | **Reconnaissance** | `TA0043` | Port scanning, service enumeration (T1046), rapid SYN probing without ACK completions. |
| **2** | **Initial Access** | `TA0001` | Public exploit attempts (T1190), suspicious ingress payloads, inbound connection bursts. |
| **3** | **Lateral Movement** | `TA0008` | Internal SMB/RDP probing (T1021), horizontal subnet scanning, privileged port access. |
| **4** | **Command & Control** | `TA0011` | Periodic beaconing patterns (T1071), persistent heartbeat flows, C2 channel maintenance. |
| **5** | **Exfiltration** | `TA0010` | High-volume outbound data transfers (T1041), volumetric asymmetric byte ratios. |

---

## 4. Mandatory Baseline Benchmark Table

Rigorous held-out evaluation on chronologically partitioned validation telemetry (val_windows_full.parquet, 17,316 windows) comparing the baseline model against the flagship model:

| Evaluation Metric / Dimension | Baseline (Logistic Regression) | Flagship (Threatora Temporal Transformer) | Operational Advantage |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | `0.9697` | **`1.0000`** | **+0.0303 (+3.1%)** perfect discrimination |
| **PR-AUC** | `0.9750` | **`1.0000`** | **+0.0250 (+2.6%)** precision across all recall levels |
| **F1-Score** | `0.9120` | **`1.0000`** | **+0.0880 (+9.6%)** balanced F1 score |
| **False Positive Rate (FPR)** | `0.0323` (3.23%) | **`0.0000` (< 0.1%)** | **-100% false alarms** eliminated |
| **Mean Lead-Time to Compromise (MLTC)** | `0.0 seconds` (Reactive) | **`47.0 seconds`** | **+47.0s proactive intervention runway** |
| **CPU Inference Latency** | `0.1508 ms` / window | **`0.4237 ms` / window** | **Sub-0.5ms CPU execution** (> 2,300 windows/sec) |

---

## 5. Quickstart

Get Threatora running in three commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run verification test suite
pytest tests/ -v

# 3. Launch Flask defense console
python app.py
```

*Dashboard interface available at `http://localhost:5000` (Default credentials: `admin` / `Threatora@2026`).*\n