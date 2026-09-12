# 🌐 Threatora Production Deployment Guide

This guide details how to deploy **Threatora** to **Render**, cloud providers, or run an instant live demo.

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

## 🛡️ Default Operator Credentials
- **Username**: `operator`
- **Password**: `threatora_operator_secure_2026`
- **Admin**: `admin` / `threatora_admin_secure_2026`
