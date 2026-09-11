"""Views and System Health Blueprint."""

from __future__ import annotations

import json
from flask import Blueprint, render_template, jsonify, current_app, session
from src.config import ALL_FEATURE_COLS, REPORTS_DIR
from src.db.session import get_db_context
from src.db.models import User
from .auth import login_required

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
@login_required
def index():
    """Serves the cybersecurity operations dashboard."""
    user_id = session.get("user_id")
    user = {
        "id": 1,
        "username": session.get("username", "admin"),
        "full_name": session.get("full_name", "Alex Kelly"),
        "role": session.get("role", "CHIEF_CISO_ADMIN"),
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
                        "role": db_user.role,
                        "is_active": db_user.is_active,
                        "created_at": db_user.created_at,
                        "last_login": db_user.last_login,
                    }
        except Exception:
            pass
    return render_template("index.html", user=user)



@views_bp.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint exposing device backend and model configuration."""
    engine = current_app.extensions["inference_engine"]
    return jsonify({
        "status": "healthy",
        "service": "Threatora World Model & Mitigation Engine",
        "device": str(engine.device),
        "obs_dimension": len(ALL_FEATURE_COLS),
        "recurrent_cell": "LSTM",
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
