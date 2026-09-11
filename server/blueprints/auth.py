"""Threatora Zero-Trust Authentication Blueprint (NTRO PS 26153).

Provides:
  - @login_required decorator protecting Operator Dashboard and sensitive endpoints
  - Session-based cookie auth for web portal
  - API Key / Bearer Token support for CLI and automated integration testing
  - User verification, last-login audit timestamps, and password validation
"""

from __future__ import annotations

import os
from functools import wraps
from datetime import datetime, timezone
from typing import Optional

from flask import (
    Blueprint,
    request,
    session,
    redirect,
    url_for,
    render_template,
    jsonify,
    current_app,
)

from src.db.session import get_db_context
from src.db.models import User, utc_now

auth_bp = Blueprint("auth", __name__)

# System API Key for headless scripts, tests, and CLI automated tasks
SYSTEM_API_KEY = os.environ.get("THREATORA_API_KEY", "threatora-zero-trust")


def is_authenticated() -> bool:
    """Check if the current request is authenticated via session or API token."""
    # 1. Check Flask session
    if session.get("user_id"):
        return True

    # 2. Check X-API-Key or Authorization header for CLI / Automated Test Suite
    api_key = request.headers.get("X-API-Key")
    auth_header = request.headers.get("Authorization", "")
    if api_key and api_key == SYSTEM_API_KEY:
        return True
    if auth_header.startswith("Bearer ") and auth_header[7:] == SYSTEM_API_KEY:
        return True

    return False


def login_required(f):
    """Decorator requiring operator authentication before executing route."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_authenticated():
            if request.path.startswith("/api/"):
                return jsonify({
                    "status": "error",
                    "error": "Unauthorized",
                    "message": "Operator authentication required to access this endpoint."
                }), 401
            return redirect(url_for("auth.login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Operator authentication portal."""
    if request.method == "GET":
        if is_authenticated():
            return redirect(url_for("views.index"))
        return render_template("login.html", error=None)

    # POST authentication attempt
    is_json = request.is_json
    data = request.get_json() if is_json else request.form

    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        err_msg = "Please provide both Operator Username and Security Passphrase."
        if is_json:
            return jsonify({"status": "error", "message": err_msg}), 400
        return render_template("login.html", error=err_msg), 400

    with get_db_context() as db:
        user = db.query(User).filter(
            (User.username == username) | (User.email == username)
        ).first()

        if user and user.is_active and user.check_password(password):
            # Establish session
            session.permanent = True
            session["user_id"] = user.id
            session["username"] = user.username
            session["full_name"] = user.full_name
            session["role"] = user.role

            # Update audit timestamp
            user.last_login = utc_now()
            db.commit()

            if is_json:
                return jsonify({
                    "status": "success",
                    "message": "Authentication successful.",
                    "user": user.to_dict()
                })

            next_url = request.args.get("next")
            if not next_url or not next_url.startswith("/"):
                next_url = url_for("views.index")
            return redirect(next_url)

        # Authentication failed
        err_msg = "Invalid Operator Credentials or Inactive Account."
        if is_json:
            return jsonify({"status": "error", "message": err_msg}), 401
        return render_template("login.html", error=err_msg), 401


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Create new Operator / Analyst identity."""
    if request.method == "GET":
        if is_authenticated():
            return redirect(url_for("views.index"))
        return render_template("register.html", error=None)

    is_json = request.is_json
    data = request.get_json() if is_json else request.form

    full_name = (data.get("full_name") or "").strip() or "Security Analyst"
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""
    role = (data.get("role") or "SOC_ANALYST").strip()

    if not username or not email or not password:
        err_msg = "Please provide Full Name, Operator Username, Email, and Passphrase."
        if is_json:
            return jsonify({"status": "error", "message": err_msg}), 400
        return render_template("register.html", error=err_msg), 400

    if len(password) < 6:
        err_msg = "Passphrase must be at least 6 characters long."
        if is_json:
            return jsonify({"status": "error", "message": err_msg}), 400
        return render_template("register.html", error=err_msg), 400

    with get_db_context() as db:
        existing = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing:
            err_msg = "An operator with this username or email already exists."
            if is_json:
                return jsonify({"status": "error", "message": err_msg}), 400
            return render_template("register.html", error=err_msg), 400

        new_user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            is_active=True,
        )
        new_user.set_password(password)
        db.add(new_user)
        db.commit()

        # Automatically establish session upon registration
        session.permanent = True
        session["user_id"] = new_user.id
        session["username"] = new_user.username
        session["full_name"] = new_user.full_name
        session["role"] = new_user.role

        if is_json:
            return jsonify({
                "status": "success",
                "message": "Operator identity created successfully.",
                "user": new_user.to_dict()
            }), 201

        return redirect(url_for("views.index"))


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """Terminates operator session."""
    session.clear()
    if request.is_json:
        return jsonify({"status": "success", "message": "Operator session terminated."})
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Operator Profile and Zero-Trust Clearance portal."""
    user_id = session.get("user_id")

    with get_db_context() as db:
        user = None
        if user_id:
            user = db.query(User).filter_by(id=user_id).first()
        if not user:
            # Fallback to current admin user or first active user
            user = db.query(User).first()

        if not user:
            return redirect(url_for("auth.login"))

        def get_user_dict(u):
            return {
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at,
                "last_login": u.last_login,
            }

        if request.method == "GET":
            return render_template(
                "profile.html",
                user=get_user_dict(user),
                api_key=SYSTEM_API_KEY,
                success=None,
                error=None,
            )

        # Handle POST actions
        action = request.form.get("action", "update_profile")

        if action == "update_profile":
            full_name = (request.form.get("full_name") or "").strip()
            email = (request.form.get("email") or "").strip()
            role = (request.form.get("role") or "").strip()

            if not full_name or not email:
                return render_template(
                    "profile.html",
                    user=get_user_dict(user),
                    api_key=SYSTEM_API_KEY,
                    success=None,
                    error="Full Name and Security Email cannot be empty.",
                ), 400

            existing = db.query(User).filter(User.email == email, User.id != user.id).first()
            if existing:
                return render_template(
                    "profile.html",
                    user=get_user_dict(user),
                    api_key=SYSTEM_API_KEY,
                    success=None,
                    error=f"Email '{email}' is already registered to another operator.",
                ), 400

            user.full_name = full_name
            user.email = email
            if role:
                user.role = role
            db.commit()

            # Refresh session variables
            session["full_name"] = user.full_name
            session["role"] = user.role

            return render_template(
                "profile.html",
                user=get_user_dict(user),
                api_key=SYSTEM_API_KEY,
                success="Operator profile successfully updated.",
                error=None,
            )

        elif action == "change_password":
            current_pass = request.form.get("current_password") or ""
            new_pass = request.form.get("new_password") or ""
            confirm_pass = request.form.get("confirm_password") or ""

            if not current_pass or not new_pass or not confirm_pass:
                return render_template(
                    "profile.html",
                    user=get_user_dict(user),
                    api_key=SYSTEM_API_KEY,
                    success=None,
                    error="All password fields are required.",
                ), 400

            if not user.check_password(current_pass):
                return render_template(
                    "profile.html",
                    user=get_user_dict(user),
                    api_key=SYSTEM_API_KEY,
                    success=None,
                    error="Current passphrase does not match our records.",
                ), 400

            if len(new_pass) < 6:
                return render_template(
                    "profile.html",
                    user=get_user_dict(user),
                    api_key=SYSTEM_API_KEY,
                    success=None,
                    error="New passphrase must be at least 6 characters long.",
                ), 400

            if new_pass != confirm_pass:
                return render_template(
                    "profile.html",
                    user=get_user_dict(user),
                    api_key=SYSTEM_API_KEY,
                    success=None,
                    error="New passphrase and confirmation passphrase do not match.",
                ), 400

            user.set_password(new_pass)
            db.commit()

            return render_template(
                "profile.html",
                user=get_user_dict(user),
                api_key=SYSTEM_API_KEY,
                success="Security passphrase updated successfully.",
                error=None,
            )

        return render_template(
            "profile.html",
            user=get_user_dict(user),
            api_key=SYSTEM_API_KEY,
            success=None,
            error=None,
        )


@auth_bp.route("/api/v1/auth/me", methods=["GET"])
@login_required
def me():
    """Returns currently authenticated operator profile."""
    user_id = session.get("user_id")
    if user_id:
        with get_db_context() as db:
            user = db.query(User).filter_by(id=user_id).first()
            if user:
                return jsonify({"status": "success", "user": user.to_dict()})

    # Headless system session
    return jsonify({
        "status": "success",
        "user": {
            "id": 0,
            "username": "system-service",
            "full_name": "Automated Security Agent",
            "role": "SYSTEM_SERVICE",
            "is_active": True
        }
    })

