"""Threatora Flask WSGI Server Application Factory (NTRO PS 26153).

Implements:
  - Factory Pattern (create_app)
  - Modular Blueprints: Views, Telemetry, Mitigation, Simulation
  - Lazy singleton initialization: PyTorch models are loaded on FIRST request,
    not at startup. This prevents Gunicorn from timing-out on Render free tier
    where cold-start model loading can take 20-60 seconds.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from flask import Flask, request, jsonify


def create_app(config: dict = None) -> Flask:
    """Application factory for Threatora Production WSGI Server."""
    root_path = Path(__file__).resolve().parent
    template_folder = root_path / "templates"
    static_folder = root_path / "static"

    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
    )
    # 1 GB upload limit for dataset telemetry captures
    max_upload_mb = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
    app.config["MAX_CONTENT_LENGTH"] = max_upload_mb * 1024 * 1024
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "threatora_zero_trust_super_secret_key_2026")

    if config:
        app.config.update(config)

    # Initialize State Store Ledger & Asset Seeding
    try:
        from src.db.session import init_db
        init_db()
        print("[+] Threatora PostgreSQL/SQLite State Ledger initialized.")
    except Exception as e:
        print(f"[!] Warning: Database initialization encountered error: {e}")

    # -------------------------------------------------------------------------
    # Lazy Singleton Initialization
    # PyTorch + World Model are heavy — defer loading until the FIRST request
    # so Gunicorn binds its port immediately and Render doesn't see a 502.
    # -------------------------------------------------------------------------
    _engines_loaded = threading.Event()
    _engines_lock = threading.Lock()
    _engine_error: list = []  # mutable container for cross-closure error capture

    def _load_engines():
        """Load all ML engine singletons. Called once, thread-safe."""
        with _engines_lock:
            if app.extensions.get("inference_engine") is not None:
                return  # Already loaded by another concurrent request
            try:
                print("[*] Loading Threatora ML engines (first request — this may take ~30s on free tier)...")
                from src.inference import InferenceEngine
                from src.mitigation import MitigationEngine
                from src.simulation import WhatIfSimulationEngine

                inference_engine = InferenceEngine()
                mitigation_engine = MitigationEngine(anomaly_threshold=0.5)
                simulation_engine = WhatIfSimulationEngine(
                    model=inference_engine.model,
                    device=inference_engine.device
                )

                app.extensions["inference_engine"] = inference_engine
                app.extensions["mitigation_engine"] = mitigation_engine
                app.extensions["simulation_engine"] = simulation_engine
                print("[+] Threatora ML engines loaded and ready.")
                _engine_error.clear()
            except Exception as exc:
                err_msg = f"Engine initialization failed: {exc}"
                print(f"[!!!] {err_msg}")
                _engine_error.append(str(exc))
            finally:
                _engines_loaded.set()

    @app.before_request
    def ensure_engines_ready():
        """Block the first request until ML engines are loaded."""
        # Skip health checks so Render's health endpoint works immediately
        if request.path in ("/health", "/healthz", "/ping", "/"):
            return None

        if app.extensions.get("inference_engine") is None:
            if not _engines_loaded.is_set():
                _load_engines()
            # If loading failed, return 503 for API calls
            if _engine_error and request.path.startswith("/api/"):
                return jsonify({
                    "status": "error",
                    "error": "ServiceUnavailable",
                    "message": f"ML engine failed to initialize: {_engine_error[-1]}"
                }), 503

    # Register Modular Blueprints
    from .blueprints.auth import auth_bp
    from .blueprints.views import views_bp
    from .blueprints.telemetry import telemetry_bp
    from .blueprints.mitigation import mitigation_bp
    from .blueprints.simulation import simulation_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(views_bp)
    app.register_blueprint(telemetry_bp)
    app.register_blueprint(mitigation_bp)
    app.register_blueprint(simulation_bp)

    @app.route("/health")
    def health_check():
        """Lightweight health endpoint — always responds 200 instantly."""
        engines_status = "ready" if app.extensions.get("inference_engine") else "loading"
        return jsonify({"status": "ok", "engines": engines_status}), 200

    @app.errorhandler(413)
    def handle_file_too_large(e):
        max_mb = app.config.get("MAX_CONTENT_LENGTH", 0) // (1024 * 1024)
        return jsonify({
            "status": "error",
            "code": 413,
            "error": "RequestEntityTooLarge",
            "message": f"File exceeds the maximum upload size ({max_mb} MB). "
                       f"Please reduce the file size or split the capture.",
        }), 413

    @app.errorhandler(Exception)
    def handle_api_exception(e):
        if request.path.startswith("/api/"):
            code = getattr(e, "code", 500)
            description = getattr(e, "description", str(e))
            return jsonify({
                "status": "error",
                "code": code,
                "error": type(e).__name__,
                "message": description,
            }), code
        return e

    return app
