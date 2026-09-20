# 🌐 Threatora Production Deployment Guide

This guide details how to deploy **Threatora** to **Render**, cloud VPS providers, Linux systemd daemons, Docker Compose, or run an instant live demo.

---

## 🚀 Option 1: Deploy on Render (Free Tier - Step by Step)

We have already configured the optimal settings for Render's free tier in the `render.yaml` file (handling low memory and startup timeouts). The easiest way to deploy is using Render's Blueprint feature.

### Step 1: Push your code to GitHub
Make sure all your latest code, including the `render.yaml` file, is pushed to your GitHub repository (`Divyansh1982006/Threatora`).

### Step 2: Create a Blueprint on Render
1. Go to [https://dashboard.render.com/](https://dashboard.render.com/) and sign in.
2. Click **New +** in the top navigation bar and select **Blueprint**.
3. Under **Connect a repository**, select `Divyansh1982006/Threatora` (or paste `https://github.com/Divyansh1982006/Threatora.git`).
4. Render will automatically read the `render.yaml` file in your repository.
5. Provide a **Service Group Name** (e.g., `Threatora-App`) and click **Apply**.

### Step 3: Wait for Deployment
- Render will now automatically configure the Web Service with the exact `Free` tier settings, build commands, and start commands required.
- Build command: `pip install --upgrade pip && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --preload app:app`
- Wait for the build to finish (it might take 5-10 minutes since PyTorch is heavy).
- Once live, you will get your public URL (e.g., `https://threatora-xxxx.onrender.com`).
- **Note:** Because it's the free tier, the very first time you load the page (or upload a file), it might take ~30 seconds as the ML models load into memory. After that, it will be fast.

---

## ⚡ Option 2: Instant 30-Second Public Demo (Localtunnel)

If your local Threatora server is running on port 5000 and you want an instant live URL right now to view on your phone or send to evaluators:

```powershell
npx localtunnel --port 5000
```
- This gives you a secure live public link: `https://xxxx.loca.lt`
- Access it on any phone or browser anywhere in the world!

---

## 🐳 Option 3: Docker Compose (VPS / AWS / DigitalOcean)

To run on an Ubuntu/Debian Linux VPS:

```bash
git clone https://github.com/Divyansh1982006/Threatora.git
cd Threatora
docker compose up -d --build
```
- Access at: `http://YOUR_SERVER_IP:5000`
- Includes containerized PostgreSQL database and automatic volume persistence.

---

## 🐧 Option 4: Linux Systemd Daemon (`deploy/threatora.service`)

To configure Threatora as a persistent background daemon on an air-gapped or Linux server:

1. Copy the systemd service file:
   ```bash
   sudo cp deploy/threatora.service /etc/systemd/system/threatora.service
   ```
2. Reload systemd and enable service on boot:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable threatora
   sudo systemctl start threatora
   ```
3. Check daemon status:
   ```bash
   sudo systemctl status threatora
   ```

---

## 📊 Operational Interface Options

| Interface | Entrypoint | Default Port | Primary Purpose |
| :--- | :--- | :---: | :--- |
| **Threatora Defense Terminal** | `python app.py` | `5000` | 2.5GB chunked uploader, SSE live telemetry, Canvas 3D radar, 403 role elevation |
| **Streamlit SOC Dashboard** | `streamlit run src/dashboard/app.py` | `8501` | Multi-horizon risk fan-chart, attention heatmap, saliency waterfall, What-If sandbox |
| **Tactical Cyber CLI** | `python cli.py console` | N/A | Terminal Metasploit-style HUD, state rollout, live packet sniffer, Zero-Trust login |

---

## ⚡ Production Server Capabilities
- **Upload Ceiling**: Configured for up to **2.5 GB** (`2684354560` bytes) via chunked multipart uploads (`/api/upload-chunk`).
- **Low-Memory Ingestion**: Streams packet/flow data in 10 MB chunks and 4 MB binary buffers directly into `uploads/` with zero OOM risk.
- **Server-Sent Events (SSE)**: Live telemetry broadcasting on `/api/stream-telemetry` without polling overhead.
- **Unified Entrypoint**: Run `python app.py` (or `gunicorn app:app`) to start the production WSGI server binding `0.0.0.0:5000`.

---

## ⚙️ Environment Variables Reference

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Network binding interface address |
| `PORT` | `5000` | HTTP listening port |
| `MAX_UPLOAD_MB` | `2560` | Maximum upload size ceiling in megabytes (2.5 GB) |
| `SECRET_KEY` | `threatora_zero_trust_defense_2026` | Flask session and cryptographic signing key |
| `THREATORA_API_KEY` | `threatora-zero-trust` | Master API key for headless CLI and automated access |
| `DATABASE_URL` | `sqlite:///artifacts/data/threatora.db` | Database connection URI (PostgreSQL or SQLite) |
| `FLASK_ENV` | `production` | Environment mode (`production` or `development`) |

---

## 🛡️ Default Operator Credentials & RBAC Clearance
- **Admin (Chief CISO / Level 5)**: `admin` / `Threatora@2026`
- **Operator (Senior SOC / Level 3)**: `operator` / `threatora_operator_secure_2026`
- **API Key**: `threatora-zero-trust`
- **Clearance Levels**:
  - `Level 1`: Guest / Observer (Terminal radar & SSE feed)
  - `Level 2`: SOC Analyst (Data ingestion, flow exploration)
  - `Level 3`: Senior SOC (Mitigation review, What-If simulation)
  - `Level 4`: SecOps Lead (Containment playbook execution)
  - `Level 5`: Chief CISO (Full administrative override & air-gap lockdown)

