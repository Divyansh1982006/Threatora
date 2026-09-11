"""Threatora Flask WSGI Server Application Factory (NTRO PS 26153).

Implements:
  - Factory Pattern (create_app)
  - Modular Blueprints: Views, Telemetry, Mitigation, Simulation
  - Worker-safe Singleton initialization for PyTorch LSTM World Model & Mitigation Engine
"""

from __future__ import annotations

import os
from pathlib import Path
from flask import Flask

from src.db.session import init_db
from src.inference import InferenceEngine
from src.mitigation import MitigationEngine
from src.simulation import WhatIfSimulationEngine


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
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload limit
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "threatora_zero_trust_super_secret_key_2026")

    if config:
        app.config.update(config)

    # Initialize State Store Ledger & Asset Seeding
    try:
        init_db()
        print("[+] Threatora PostgreSQL/SQLite State Ledger initialized.")
    except Exception as e:
        print(f"[!] Warning: Database initialization encountered error: {e}")

    # Instantiate Inference, Mitigation, and Simulation Singletons
    inference_engine = InferenceEngine()
    mitigation_engine = MitigationEngine(anomaly_threshold=0.5)
    simulation_engine = WhatIfSimulationEngine(
        model=inference_engine.model,
        device=inference_engine.device
    )

    # Attach singletons to Flask extensions for blueprint dependency injection
    app.extensions["inference_engine"] = inference_engine
    app.extensions["mitigation_engine"] = mitigation_engine
    app.extensions["simulation_engine"] = simulation_engine

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

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key, X-Simulate-Role"
        return response

    return app
