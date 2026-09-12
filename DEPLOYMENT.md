# 🌐 Threatora Production Deployment Guide

This guide details how to deploy **Threatora** to **Render**, cloud providers, or run an instant live demo.

---

## 🚀 Option 1: Deploy on Render (Recommended Free Cloud Hosting)

Render provides free hosting for web services with automatic SSL, custom domains, and continuous deployment from GitHub.

### Step 1: Connect your GitHub Repository
1. Go to [https://dashboard.render.com/](https://dashboard.render.com/) and sign in.
2. Click **New +** in the top navigation bar and select **Web Service**.
3. Under **Connect a repository**, select `Divyansh1982006/Threatora` (or paste `https://github.com/Divyansh1982006/Threatora.git`).

---

### Step 2: Configure Service Settings

You can deploy using either **Docker** (Recommended) or **Native Python**:

#### Method A: Docker Deployment (Pre-configured via `render.yaml` / `Dockerfile`)
- **Name**: `threatora-engine` (or your choice)
- **Region**: Oregon (US West) or Singapore
- **Branch**: `main`
- **Runtime**: `Docker`
- **Instance Type**: `Free`
- **Advanced -> Environment Variables**:
  - `PORT`: `5000`
  - `FLASK_ENV`: `production`
  - `PYTHONUNBUFFERED`: `1`
  - `SECRET_KEY`: *(Generate a secure random string or let Render create one)*
  - *(Optional)* `DATABASE_URL`: *(Leave empty to use built-in SQLite, or add Render PostgreSQL URL)*

#### Method B: Native Python Environment
- **Name**: `threatora-web`
- **Runtime**: `Python 3`
- **Build Command**:
  ```bash
  pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 server.app:app
  ```

---

### Step 3: Click "Create Web Service"
- Render will pull your repository, build the dependencies, and deploy the service.
- Once live, you will get your public URL:
  `https://threatora-engine.onrender.com`

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

## 🛡️ Default Operator Credentials
- **Username**: `operator`
- **Password**: `threatora_operator_secure_2026`
- **Admin**: `admin` / `threatora_admin_secure_2026`
