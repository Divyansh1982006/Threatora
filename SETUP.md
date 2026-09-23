# Threatora — Setup & Deployment Guide

**SIH 2026 · PS-26153 · NTRO**

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.14 | Tested on 3.13 / 3.14 |
| pip | Latest | `python -m pip install --upgrade pip` |
| Git | Any | For cloning / pushing |
| RAM | ≥ 4 GB | Transformer World Model runs on CPU |
| OS | Windows / Linux / macOS | Windows tested |

> **No GPU required.** Threatora runs fully on CPU in offline mode.

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Divyansh1982006/Threatora.git
cd Threatora
```

---

## Step 2 — Create a Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `torch` — Temporal Transformer World Model inference
- `flask` + `werkzeug` — Web backend
- `sqlalchemy` — SQLite state ledger
- `pandas`, `numpy`, `scikit-learn` — Feature pipeline
- `scapy` — PCAP parsing
- `rich` — CLI output
- `requests`, `pytest` — Testing

> ⏳ First install may take 3–5 minutes due to PyTorch download size.

---

## Step 3.5 — CTU-13 Dataset Setup & Extraction

Threatora integrates the **CTU-13 Botnet Benchmark Dataset** across 13 full-capture scenarios:

### Included Processed Dataset (`data/processed/`)
- `scenario_01.parquet` through `scenario_13.parquet`: **258,229 preprocessed host-window state cells** with 60-second temporal aggregation and lookahead targets.
- `scenario_01_edges.parquet` through `scenario_13_edges.parquet`: Communication topology graphs.
- `ingest_report.json`: Dataset metrics, cell counts, and malicious activity rates.
- `processed_cells.csv`: 16-canonical feature training-ready matrix.

### Extraction & Sample Utilities (`scripts/`)
1. **Extracting raw CTU-13 archives**:
   If downloading the complete `CTU-13-Dataset.tar.bz2`, extract only `.binetflow` files without unpacking 54 GB of unneeded PCAPs:
   ```bash
   python scripts/extract_binetflow.py --archive data/raw/CTU-13-Dataset.tar.bz2
   ```

2. **Generating transition sample (`host-becomes-infected.csv`)**:
   Splice clean and botnet traffic to create a machine transitioning from benign to infected:
   ```bash
   python scripts/make_transition_sample.py
   ```

3. **Running data preparation pipeline**:
   ```bash
   python -m src.prepare_data
   ```

### MITRE ATT&CK Taxonomy & Engine (`src/mitre.py`)
Threatora maps flows directly into the 6-stage ATT&CK taxonomy:
- **0. Benign (`-`)**: Normal enterprise traffic
- **1. Reconnaissance (`TA0043 / T1046`)**: Probing, scan attempts, DNS discovery
- **2. Initial Access (`TA0001 / T1190`)**: Malicious payload download, binary retrieval
- **3. Lateral Movement (`TA0008 / T1021`)**: Internal administrative traffic (SMB, RDP)
- **4. Command & Control (`TA0011 / T1071`)**: Periodic C2 beacons, IRC channels, fast-flux
- **5. Exfiltration (`TA0010 / T1041`)**: Outbound volumetric data transfers, spam relays

---

## Step 4 — Start the Server

You can launch Threatora in either of the two operational modes:

### Option A — Threatora Production Defense Server (Port 5000)
```bash
python app.py
```
Or use the automated Windows launcher:
```powershell
start_server.bat
```

Expected output:
```text
[*] Starting Threatora Defense Server on 0.0.0.0:5000 (Max Upload: 2.5 GB)...
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
 * Running on http://192.168.0.105:5000
```

Open **http://127.0.0.1:5000** in your browser.

**Key Endpoints Available:**
- `GET /`: Defense Terminal UI with 3D Canvas Radar (`static/js/topology.js`)
- `POST /api/upload-chunk`: 10 MB multipart binary chunked uploader (supports up to 2.5 GB)
- `GET /api/stream-telemetry`: Server-Sent Events (SSE) real-time attack forecasting stream
- `GET /dashboard`: Operations HUD (Telemetry & World Model)
- `GET /visualizations`: Network Topology Studio & What-If Sandbox
- `GET /mitigation`: Zero-Trust Mitigation & Asset Ledger
- `GET /api/health`: Service health and CUDA/CPU device status
---

## Step 5 — Login / Register

- Visit **http://127.0.0.1:5000/login**
- Register a new operator account at **/register**
- Default Operator credentials: `admin` / `Threatora@2026`
- Or use the **API Key** for headless access:
```
X-API-Key: threatora-zero-trust
```

---

## Step 6 — Verify Backend (Run Test Suite)

Run the comprehensive pytest test suite:

```bash
pytest tests/ -v
```

Expected result:
```text
================= 37 passed, 1 skipped in 28.14s =================
```

### What the test suite covers:
| Test Module | Coverage |
|---|---|
| `test_auth.py` | Zero-trust session auth, passwords, API keys, role enforcement |
| `test_dynamic_adapters.py` | Schema registry, zero-leakage stripping, 16-slot mapping, Smooth L1 |
| `test_flask_soc_pipeline.py` | Fast PCAP parser speed, multi-stage attack diversity, counterfactuals |
| `test_dual_world_model.py` | Flow & packet world model inference, late fusion, decision thresholds |
| `test_transformer_model.py` | Attention heads, learnable positional embeddings, predict_timeline() |
| `test_dynamic_topology_mitigation.py` | Dynamic topology node derivation, mitigation playbooks, host isolation |

---

## Step 7 — Load Trained Model Weights

Since training is done on a separate machine, transfer the weights and load them:

### Option A — CLI Import (Recommended)

```bash
python cli.py import-weights \
  --weights path/to/world_model.pt \
  --scaler path/to/scaler.json
```

### Option B — Manual Copy

Copy files directly into the checkpoints directory:

```
artifacts/
└── checkpoints/
    ├── world_model.pt       ← PyTorch Transformer weights
    ├── scaler.json          ← FeatureScaler normalization params
    └── run_config.json      ← Architecture config (optional)
```

The server picks up weights **automatically** — no restart needed.

---

## Step 8 — Generate Benchmark Report (Optional)

```bash
python cli.py benchmark
```

This compares the Temporal Transformer World Model against a Logistic Regression baseline
and saves the report to `artifacts/reports/benchmark.json`.

---

## Step 9 — Run CLI Forecast

### A. Forecast from Flow CSV / BinetFlow
```bash
# Multi-stage attack sample
python cli.py predict --flow data/samples/sample_traffic.csv

# Real transition sample (benign host compromised)
python cli.py predict --flow data/samples/host-becomes-infected.csv

# CTU-13 Scenario 5 (Virut Fast-Flux capture)
python cli.py predict --flow F:\SIH\5\capture20110815-2.binetflow
```

### B. Dual-Modality Inference (Flow + PCAP)
```bash
# Run simultaneous dual-branch inference on CTU-13 Scenario 5
python cli.py predict \
  --flow F:\SIH\5\capture20110815-2.binetflow \
  --packet F:\SIH\5\botnet-capture-20110815-fast-flux.pcap
```

### C. Output Formats
```bash
# Output formatted JSON report
python cli.py predict --flow data/samples/host-becomes-infected.csv --format json
```

---

## Step 10 — Interactive Tactical Cyber Terminal (Metasploit-Style)

Threatora includes a state-of-the-art interactive tactical console with **Zero-Trust Web Authentication** integrated directly with the web portal.

### Launching the Console

```bash
# Interactive mode (prompts for web portal username & passphrase)
python cli.py console

# Direct login with credentials
python cli.py console -u admin -p Threatora@2026

# Headless / automation mode (uses system API key)
python cli.py console --no-auth
```

### Web Portal Default Operator Credentials
- **Username:** `admin`
- **Passphrase:** `Threatora@2026`

### Key Tactical Directives

| Category | Command | Description |
|----------|---------|-------------|
| **Executive HUD** | `dashboard` | Full-spectrum 3-panel Cyber HUD (Engine, Threat Radar, Zero-Trust Posture) |
| **Network Topology** | `topology` / `netmap` | Visual network tree of enterprise assets across WAN, DMZ, and Internal Subnets |
| **Live Packet Radar** | `monitor` / `sniff` | Animated real-time NetFlow frame monitor with flow vectors & risk gauges |
| **OSI Defense Matrix** | `layers` / `osi` | 7-Layer OSI attack surface & 16-canonical feature mapping |
| **Autonomous Triage** | `quickscan` | 1-Click autonomous workflow: Scan → Lock Target → Forecast → Explain → Mitigate |
| **Target Management** | `targets` / `use <#>` | Numbered target inventory and Metasploit-style target selector |
| **Live Telemetry** | `scan --live` | Ingest real-time telemetry stream and evaluate MITRE ATT&CK stages |
| **State Forecasting** | `forecast --steps 5` | 5-Step Direct Horizon Projection (k=1..5) with 95% CI bounds |
| **Explainable AI** | `explain` | Saliency attribution waterfall chart & SHAP feature ranking |
| **What-If Simulation** | `simulate -a ISOLATE_HOST` | Counterfactual trajectory simulation & risk reduction comparison |
| **Containment** | `mitigate --isolate` | Enforce Zero-Trust host isolation directive with audit log |
| **Identity & Access** | `whoami` / `login` / `logout` | Inspect operator clearance badge or switch user credentials |
| **Ledger Inspection** | `assets` / `incidents` / `playbooks` | Query enterprise assets, security incident log, and containment playbooks |

---

## Step 11 — Multi-Laptop Network Setup & Remote Cloud Deployment

Threatora is engineered for seamless distributed operations across multiple laptops during hackathons, demonstrations, and air-gapped field setups.

```
┌──────────────────────────────────────────────────────────┐
│             LAPTOP A (Host Central Server)               │
│  - Runs: start_server.bat (or python app.py)             │
│  - Binds: 0.0.0.0:5000 (Max Upload: 2.5 GB)              │
│  - IP: 172.16.190.142 (detected automatically)           │
│  - SQLite Ledger: artifacts/data/threatora.db            │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│       LAPTOP B (Client)       │ │       LAPTOP C (Client)       │
│  Web Browser Operator Console │ │  Tactical Cyber CLI Terminal  │
│  http://172.16.190.142:5000   │ │  run_cli.bat 172.16.190.142   │
│  Login: admin / Threatora@2026│ │  Login: /login admin ...      │
└───────────────────────────────┘ └───────────────────────────────┘
```

---

### Method A — Same Wi-Fi / Local Area Network (Recommended for SIH / Hackathon)

#### 1. Server Host Setup (Laptop A):
1. Simply double-click **`start_server.bat`** (or run `python app.py`).
2. The launcher automatically detects your active network interface IP (e.g. `172.16.190.142`) and prints the remote access URLs.
3. **Windows Firewall Rule (One-Time Setup):**
   If other laptops cannot reach port 5000, run this command in **PowerShell (Run as Administrator)**:
   ```powershell
   New-NetFirewallRule -DisplayName "Threatora Server (Port 5000)" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
   ```

#### 2. Client Web Access (Laptop B / Judges / Operators):
1. Open any web browser on Laptop B (connected to same Wi-Fi / LAN).
2. Navigate to:
   ```
   http://<SERVER_IP>:5000
   Example: http://172.16.190.142:5000
   ```
3. Login using the default operator credentials:
   - **Username:** `admin`
   - **Passphrase:** `Threatora@2026`
4. The full spectrum web dashboard, topology visualizer, what-if counterfactual simulator, and mitigation actions work live in real-time.

#### 3. Client CLI Access (Laptop C / Security Analyst Terminal):
1. Transfer or clone the Threatora repository onto Laptop C.
2. Launch the CLI connected to Laptop A using either method:
   - **Option 1 (1-Click Batch):**
     Double-click **`run_cli.bat`** and enter the Server IP (e.g. `172.16.190.142`).
   - **Option 2 (Direct Command):**
     ```bash
     python cli.py console --api-url http://172.16.190.142:5000
     ```
   - **Option 3 (Environment Variable):**
     ```bash
     # Windows PowerShell
     $env:THREATORA_API_URL = "http://172.16.190.142:5000"
     python cli.py console
     ```
3. Authenticate using the web credentials:
   ```
   THREATORA (unauth) > /login admin Threatora@2026
   ```
4. All CLI commands (`monitor`, `topology`, `forecast`, `explain`, `mitigate`, `simulate`) directly communicate with the remote server's API and state ledger!

---

### Method B — Instant Public Tunnel (Over Internet / Separate Mobile Hotspots)

If the laptops are on different Wi-Fi networks or public venue Wi-Fi blocks peer-to-peer traffic:

#### Using ngrok:
1. On Laptop A (where server is running):
   ```bash
   ngrok http 5000
   ```
2. Copy the public HTTPS URL (e.g., `https://a1b2-c3d4.ngrok-free.app`).
3. On any client laptop anywhere in the world:
   - **Browser:** Open `https://a1b2-c3d4.ngrok-free.app`
   - **CLI:** `python cli.py console --api-url https://a1b2-c3d4.ngrok-free.app`

#### Using localtunnel (Free, No Sign-up Required):
```bash
npx localtunnel --port 5000
```

---

### Method C — 1-Click Cloud Deployment (Render / Railway / Fly.io)

You can deploy the Threatora Core Server directly from this GitHub repository to free cloud hosting.

#### Deploy on Render (Free Web Service):
1. Go to [dashboard.render.com](https://dashboard.render.com/) and click **New + → Web Service**.
2. Connect your GitHub repository: `Divyansh1982006/Threatora`.
3. Configure settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app`
4. Add Environment Variables:
   - `HOST`: `0.0.0.0`
   - `PORT`: `5000` (Render overrides this dynamically)
   - `THREATORA_API_KEY`: `threatora-zero-trust`
   - `SECRET_KEY`: `threatora-enterprise-jwt-key-2026`
5. Click **Deploy Web Service**. Once deployed, Render provides a URL (e.g., `https://threatora-api.onrender.com`).
6. Anyone can connect their browser or CLI:
   ```bash
   python cli.py console --api-url https://threatora-api.onrender.com
   ```

---

## API Key Reference

All API endpoints require authentication.  
Use the system API key for automated / headless access:

```
Header:  X-API-Key: threatora-zero-trust
   OR
Header:  Authorization: Bearer threatora-zero-trust
```

To change the key, set the environment variable before starting the server:

```bash
# Windows PowerShell
$env:THREATORA_API_KEY = "your-custom-key"
python app.py

# Linux / macOS
THREATORA_API_KEY="your-custom-key" python app.py
```

---

## Common Issues & Fixes

### `ModuleNotFoundError: No module named 'flask'`
```bash
pip install -r requirements.txt
```

### `fatal: detected dubious ownership` (Git)
```bash
git config --global --add safe.directory F:/SIH/Threatora
```

### `WinError 10061 — Connection refused`
Server is not running. Start it first:
```bash
python app.py
```

### Port 5000 already in use
```bash
# Find and kill the process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

### `psycopg2` / `psycopg` install error on Windows
These are optional PostgreSQL drivers. The app falls back to **SQLite** automatically.
You can safely ignore these errors — the server will work fine with SQLite.

---

## Docker Deployment (Optional)

```bash
# Build and start all services
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Run CLI inside container
docker exec -it threatora_engine python cli.py predict \
  --input data/samples/sample_traffic.csv

# Stop services
docker-compose down
```

Dashboard: **http://localhost:5000**

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start server | `python app.py` (or `start_server.bat`) |
| Run all tests | `pytest tests/ -v` |
| CLI predict | `python cli.py predict --input <file>` |
| Load weights | `python cli.py import-weights --weights <file> --scaler <file>` |
| Run benchmark | `python cli.py benchmark` |
| Interactive console | `python cli.py console` |

---

*Last updated: September 2026 · Threatora v1.0 · SIH PS-26153*
