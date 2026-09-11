# Threatora — Setup & Deployment Guide

**SIH 2026 · PS-26153 · NTRO**

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.14 | Tested on 3.13 / 3.14 |
| pip | Latest | `python -m pip install --upgrade pip` |
| Git | Any | For cloning / pushing |
| RAM | ≥ 4 GB | LSTM model runs on CPU |
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
- `torch` — LSTM World Model inference
- `flask` + `werkzeug` — Web backend
- `sqlalchemy` — SQLite state ledger
- `pandas`, `numpy`, `scikit-learn` — Feature pipeline
- `scapy` — PCAP parsing
- `rich` — CLI output
- `requests`, `pytest` — Testing

> ⏳ First install may take 3–5 minutes due to PyTorch download size.

---

## Step 4 — Start the Server

```bash
python server/app.py
```

Expected output:
```
[+] Threatora PostgreSQL/SQLite State Ledger initialized.
[*] Checkpoint not found at artifacts/checkpoints/world_model.pt.
    Ready to receive weights from training pipeline.
 * Running on http://127.0.0.1:5000
 * Running on http://0.0.0.0:5000
```

Open **http://127.0.0.1:5000** in your browser.

> The server runs without trained weights. It uses random-initialized LSTM weights
> until you load a trained model (see Step 7).

---

## Step 5 — Login / Register

- Visit **http://127.0.0.1:5000/login**
- Register a new operator account at **/register**
- Or use the **API Key** for headless access:

```
X-API-Key: threatora-zero-trust
```

---

## Step 6 — Verify Backend (Run Test Suite)

Make sure the server is running, then in a **separate terminal**:

```bash
python tests/test_backend.py
```

Expected result:
```
  Passed  : 76/76
  Failed  : 0/76
  Warnings: 1       ← benchmark pending (normal)

  All tests passed! Backend is fully operational.
```

### What the tests cover:

| # | Feature | Endpoint |
|---|---------|----------|
| 1 | Health Check | `GET /api/health` |
| 2 | Auth (API Key, Bearer, Register, Login) | `/api/v1/auth/me`, `/login`, `/register` |
| 3 | Telemetry Demo | `GET /api/demo` |
| 4 | Telemetry GET | `GET /api/v1/telemetry` |
| 5 | Telemetry POST (JSON flows) | `POST /api/v1/telemetry` |
| 6 | File Upload (CSV) | `POST /api/upload` |
| 7 | Mitigation Playbooks | `GET /api/v1/playbooks` |
| 8 | Asset Inventory | `GET /api/v1/assets` |
| 9 | Incidents Log | `GET /api/v1/incidents` |
| 10 | Mitigation Action | `POST /api/v1/mitigate` |
| 11 | Simulation Actions | `GET /api/v1/simulate/actions` |
| 12 | What-If Simulation | `POST /api/v1/simulate` |
| 13 | Benchmark Report | `GET /api/benchmark` |

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
    ├── world_model.pt       ← PyTorch LSTM weights
    ├── scaler.json          ← FeatureScaler normalization params
    └── run_config.json      ← Architecture config (optional)
```

The server picks up weights **automatically** — no restart needed.

---

## Step 8 — Generate Benchmark Report (Optional)

```bash
python cli.py benchmark
```

This compares the LSTM World Model against a Logistic Regression baseline
and saves the report to `artifacts/reports/benchmark.json`.

---

## Step 9 — Run CLI Forecast

```bash
# Forecast from a CSV flow file (10-minute horizon)
python cli.py predict --input data/samples/sample_traffic.csv --horizon 10

# Output as JSON
python cli.py predict --input data/samples/sample_traffic.csv --format json
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
| **Autonomous Triage** | `quickscan` | 1-Click autonomous workflow: Scan → Lock Target → Forecast → Explain → Mitigate |
| **Target Management** | `targets` / `use <#>` | Numbered target inventory and Metasploit-style target selector |
| **Live Telemetry** | `scan --live` | Ingest real-time telemetry stream and evaluate MITRE ATT&CK stages |
| **State Forecasting** | `forecast --steps 10` | 10-step forward Monte Carlo rollout (.imagine mode) with 95% CI |
| **Explainable AI** | `explain` | Saliency attribution waterfall chart & SHAP feature ranking |
| **What-If Simulation** | `simulate -a ISOLATE_HOST` | Counterfactual trajectory simulation & risk reduction comparison |
| **Containment** | `mitigate --isolate` | Enforce Zero-Trust host isolation directive with audit log |
| **Identity & Access** | `whoami` / `login` / `logout` | Inspect operator clearance badge or switch user credentials |
| **Ledger Inspection** | `assets` / `incidents` / `playbooks` | Query enterprise assets, security incident log, and containment playbooks |

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
python server/app.py

# Linux / macOS
THREATORA_API_KEY="your-custom-key" python server/app.py
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
python server/app.py
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
| Start server | `python server/app.py` |
| Run all tests | `python tests/test_backend.py` |
| CLI predict | `python cli.py predict --input <file>` |
| Load weights | `python cli.py import-weights --weights <file> --scaler <file>` |
| Run benchmark | `python cli.py benchmark` |
| Smoke test model | `python tests/smoke_model.py` |

---

*Last updated: September 2026 · Threatora v1.0 · SIH PS-26153*
