# Threatora: Proactive Network Attack Forecasting & MITRE ATT&CK Simulation

**SIH 2026 · Problem Statement 26153 · NTRO · Blockchain & Cybersecurity**

Threatora is an open-source, fully offline cyber defense framework that models network traffic state dynamics using an **LSTM-based World Model**. Instead of point-in-time classification, Threatora learns $P(S_{t+1} \mid S_t)$ over 60-second window cells, rolling forward future trajectories to forecast attacker kill-chain progression up to 10 minutes ahead.

---

## Key Capabilities

- **62-Feature Pipeline**: Extracts 39 flow-level attributes (rates, expanded TCP flags, beacon regularity, entropy) and 23 packet-level attributes (TTL, window flow control, payload entropy, retransmissions) from network traffic.
- **LSTM-based World Model**: An LSTM transition cell maintaining hidden state context ($h_t$) and cell memory ($c_t$) across multi-minute attack dormancy intervals.
- **Generative Forward Simulation (`.imagine()`)**: Simulates 10-step ahead network states without observations, plotting infiltration probability timelines with 95% Monte Carlo uncertainty bands.
- **MITRE ATT&CK Kill-Chain Tracking**: Predicts progression across 5 core attack phases: *Reconnaissance → Initial Access → Lateral Movement → Command & Control → Exfiltration*.
- **Explainability Engine**: Saliency attribution (identifying which ports, flags, or byte volumes drove the alert), Causal Attention weights over the 16-window rolling context, and predicted state deltas.
- **Dual Interfaces**: Standalone offline **CLI** (`threatora` / `cli.py`) and dark-mode **Flask Web Dashboard**.
- **Containerized Deployment**: Fully dockerized backend (`Dockerfile` and `docker-compose.yml`).

---

## Dataset

Threatora is designed to train on a **custom, internally curated dataset** collected from real network environments. Training is performed on a dedicated external machine and the resulting model weights are transferred here for inference.

> The dataset pipeline is built to ingest raw network captures and labeled flow records specific to the threat scenarios defined in PS-26153. No third-party public dataset is bundled — the dataset will be generated and maintained by the team as part of the SIH deliverable.

---

## Quick Start

### 1. Local Setup
```bash
cd threatora/
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
# Analyze CSV flow file with 10-minute simulation
python cli.py predict --input data/samples/sample_traffic.csv --horizon 10

# Output as structured JSON
python cli.py predict --input data/samples/sample_traffic.csv --format json
```

### 4. Run Benchmark vs Logistic Regression Baseline
```bash
python cli.py benchmark
```

### 5. Launch Threatora Server (1-Click or Manual)
```bash
# 1-Click Server Launcher (Auto-detects IP & configures multi-laptop access)
start_server.bat

# Or launch directly via Python:
python server/app.py
```
Open **`http://localhost:5000`** (or your LAN IP `http://<YOUR_IP>:5000` from another laptop) in your browser. Default operator credentials: `admin` / `Threatora@2026`.

### 6. Tactical Cyber CLI Terminal (Multi-Laptop Client)
```bash
# 1-Click CLI Client Launcher (Prompts for Server IP or connects locally)
run_cli.bat

# Or launch directly specifying remote host:
python cli.py console --api-url http://<SERVER_IP>:5000
```
Authenticates seamlessly against the central Threatora server using `/login admin Threatora@2026`.

---

## Integrating Models Trained on External Machines

Since training is performed on a separate dedicated system:

1. Copy the resulting `world_model.pt` and `scaler.json` files to this machine.
2. Run the import command:
   ```bash
   python cli.py import-weights --weights path/to/world_model.pt --scaler path/to/scaler.json
   ```
   Or drop them directly into `artifacts/checkpoints/`.
3. The inference engine and web dashboard will immediately pick up the new weights without restarting.

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
