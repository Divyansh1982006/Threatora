# Threatora: Proactive Network Attack Forecasting & MITRE ATT&CK Simulation

**SIH 2026 · Problem Statement 26153 · NTRO · Blockchain & Cybersecurity**

Threatora is an open-source, fully offline cyber defense framework that models network traffic state dynamics using an **LSTM-based Recurrent State-Space World Model (RSSM)**. Instead of point-in-time classification, Threatora learns $P(S_{t+1} \mid S_t)$ over 60-second window cells, rolling forward future trajectories to forecast attacker kill-chain progression up to 10 minutes ahead.

---

## Key Capabilities

- **62-Feature Pipeline**: Extracts 39 flow-level attributes (rates, expanded TCP flags, beacon regularity, entropy) and 23 packet-level attributes (TTL, window flow control, payload entropy, retransmissions) from PCAPs, CTU-13, and CIC-IDS-2018 datasets.
- **LSTM-based World Model**: Replaces standard GRUs with an LSTM transition cell maintaining hidden state context ($h_t$) and cell memory ($c_t$) across multi-minute attack dormancy intervals.
- **Generative Forward Simulation (`.imagine()`)**: Simulates 10-step ahead network states without observations, plotting infiltration probability timelines with 95% Monte Carlo uncertainty bands.
- **MITRE ATT&CK Kill-Chain Tracking**: Predicts progression across 5 core attack phases: *Reconnaissance $\to$ Initial Access $\to$ Lateral Movement $\to$ Command & Control $\to$ Exfiltration*.
- **Explainability Engine**: Saliency attribution (identifying which ports, flags, or byte volumes drove the alert), Causal Attention weights over the 16-window rolling context, and predicted state deltas.
- **Dual Interfaces**: Standalone offline **CLI** (`threatora` / `cli.py`) and dark-mode **Flask Web Dashboard**.
- **Containerized Deployment**: Fully dockerized backend (`Dockerfile` and `docker-compose.yml`).
- **External Training Integration**: Pre-configured weight directory (`artifacts/checkpoints/`) and CLI import tool for models training on external machines.

---

## Directory Structure

```
D:\threatora/
├── artifacts/
│   ├── checkpoints/                 # Target directory for model weights & scalers
│   │   ├── world_model.pt           # PyTorch LSTM RSSM weights
│   │   ├── scaler.json              # FeatureScaler normalization parameters
│   │   └── run_config.json          # Architecture configuration metadata
│   └── reports/
│       ├── benchmark.json           # World Model vs Logistic Regression benchmark
│       └── benchmark.md             # Benchmark summary report
│
├── data/
│   ├── raw/                         # Raw PCAP or CTU-13 / CIC-IDS-2018 binetflow/csv files
│   ├── processed/                   # Preprocessed 60-second window state cells
│   └── samples/
│       └── sample_traffic.csv       # Multi-stage realistic attack scenario
│
├── docs/
│   └── architecture.md              # Mathematical specification & RSSM architecture
│
├── server/
│   ├── app.py                       # Flask Web Backend & REST API
│   └── templates/
│       └── index.html               # Offline Cyber Operations Dashboard
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # 62-feature list, cadence & hyperparameters
│   ├── prepare_data.py              # Ingests raw data & generates 60s window cells
│   ├── dataset.py                   # PyTorch 16-window sequence dataset & weighted sampler
│   ├── mitre.py                     # 5 MITRE ATT&CK stages taxonomy & labeling rules
│   ├── features/
│   │   ├── __init__.py
│   │   ├── flow.py                  # 39 flow-level features (flags, rates, entropy, beacons)
│   │   ├── packet.py                # 23 packet-level features (PCAP parser, TTL, windows)
│   │   └── windows.py               # 60s host-window cell builder & FeatureScaler
│   ├── model/
│   │   ├── __init__.py
│   │   ├── world_model.py           # LSTM-based RSSM with .imagine() forward simulator
│   │   └── baseline.py              # Single & stacked Logistic Regression baselines
│   ├── inference.py                 # Real-time forecasting & uncertainty bounds
│   ├── explain.py                   # Saliency attribution, attention weights & state deltas
│   ├── train.py                     # Training loop with multi-task loss
│   ├── evaluate.py                  # Benchmark engine: World Model vs Logistic Regression
│   └── cli.py                       # CLI command implementation
│
├── tests/
│   ├── __init__.py
│   ├── smoke_model.py               # Unit & shape verification tests
│   └── prove_no_peeking.py          # Zero-gradient proof that rollout doesn't leak observations
│
├── cli.py                           # Top-level CLI entry point
├── Dockerfile                       # Containerized production environment
├── docker-compose.yml               # Docker Compose configuration
├── requirements.txt                 # Dependencies (Flask, PyTorch, Scapy, Scikit-learn)
└── README.md                        # Project documentation
```

---

## Quick Start

### 1. Local Setup
```bash
cd D:\threatora
python -m venv venv
venv\Scripts\activate            # On Windows; source venv/bin/activate on Linux
pip install -r requirements.txt
```

### 2. Verify Architecture & Run Tests
```bash
# Run shape & simulation smoke tests
python tests/smoke_model.py

# Verify zero-gradient observation isolation
python tests/prove_no_peeking.py
```

### 3. Run Forecast via CLI
```bash
# Analyze PCAP or CSV flow file with 10-minute simulation
python cli.py predict --input data/samples/sample_traffic.csv --horizon 10

# Output as structured JSON
python cli.py predict --input data/samples/sample_traffic.csv --format json
```

### 4. Run Benchmark vs Logistic Regression Baseline
```bash
python cli.py benchmark
```

### 5. Launch Offline Web Dashboard
```bash
python server/app.py
```
Open **`http://localhost:5000`** in your browser to inspect the interactive UI.

---

## Integrating Models Trained on External Laptops

Since your dataset is training on another machine:
1. Copy the resulting `world_model.pt` and `scaler.json` files to this machine.
2. Run the import command:
   ```bash
   python cli.py import-weights --weights path/to/world_model.pt --scaler path/to/scaler.json
   ```
   Or drop them directly into `D:\threatora\artifacts\checkpoints/`.
3. The inference engine and web dashboard will immediately pick up the new weights without restarting!

---

## Docker Deployment

To run the entire backend fully containerized:

```bash
# Build and run with Docker Compose
docker-compose up -d --build

# View logs
docker-compose logs -f

# Run CLI inside the container
docker exec -it threatora_engine python cli.py predict --input data/samples/sample_traffic.csv
```
Dashboard will be live at `http://localhost:5000`.
