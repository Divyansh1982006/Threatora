"""Views and System Health Blueprint."""

from __future__ import annotations

import json
from flask import Blueprint, render_template, jsonify, current_app, session
from src.config import ALL_FEATURE_COLS, REPORTS_DIR
from src.db.session import get_db_context
from src.db.models import User
from .auth import login_required, get_current_role, get_role_info, ROLE_PERMISSIONS

views_bp = Blueprint("views", __name__)


def _get_view_context(active_page: str = "home") -> dict:
    """Helper to assemble standard user and RBAC context for views."""
    role = get_current_role()
    role_info = get_role_info(role)

    user_id = session.get("user_id")
    user = {
        "id": 1,
        "username": session.get("username", "admin"),
        "full_name": session.get("full_name", "Alex Kelly"),
        "role": role,
        "role_info": role_info,
    }
    if user_id:
        try:
            with get_db_context() as db:
                db_user = db.query(User).filter_by(id=user_id).first()
                if db_user:
                    user = {
                        "id": db_user.id,
                        "username": db_user.username,
                        "email": db_user.email,
                        "full_name": db_user.full_name,
                        "role": role,
                        "role_info": role_info,
                        "is_active": db_user.is_active,
                        "created_at": db_user.created_at,
                        "last_login": db_user.last_login,
                    }
        except Exception:
            pass

    available_roles = [
        {"key": k, **v}
        for k, v in ROLE_PERMISSIONS.items()
    ]

    return {
        "user": user,
        "role_info": role_info,
        "available_roles": available_roles,
        "current_role": role,
        "active_page": active_page,
    }


@views_bp.route("/")
@login_required
def index():
    """Enterprise Landing / Home Page with Project Details & ASCII Art."""
    ctx = _get_view_context(active_page="home")
    return render_template("home.html", **ctx)


@views_bp.route("/dashboard")
def dashboard():
    """Serves the SOC Operations Dashboard (Telemetry & World Model HUD)."""
    ctx = _get_view_context(active_page="dashboard")
    return render_template("dashboard.html", **ctx)


@views_bp.route("/visualizations")
@views_bp.route("/topology")
def visualizations():
    """Deep Visualization Studio (Interactive Topology Map & What-If Sandbox)."""
    ctx = _get_view_context(active_page="visualizations")
    return render_template("visualizations.html", **ctx)


@views_bp.route("/mitigation")
def mitigation_view():
    """Threat Mitigation Center (Incidents Forensics, Playbooks, Asset Ledger)."""
    ctx = _get_view_context(active_page="mitigation")
    return render_template("mitigation_page.html", **ctx)


@views_bp.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint exposing device backend and model configuration."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return jsonify({
        "service": "Threatora World Model & Mitigation Engine",
        "status": "healthy",
        "model_architecture": "Threatora Temporal Transformer World Model",
        "obs_dimension": 16,
        "device": device,
        "offline_mode": True,
        "database_connected": True,
    })


@views_bp.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    """Returns benchmark comparison against Logistic Regression."""
    bench_file = REPORTS_DIR / "benchmark.json"
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({
        "status": "pending",
        "message": "Benchmark not yet generated. Run `python cli.py benchmark`."
    })
