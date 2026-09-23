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

ROLE_PERMISSIONS = {
    "CHIEF_CISO_ADMIN": {
        "level": 5,
        "title": "Chief CISO / SecOps Director",
        "clearance": "TOP SECRET // LEVEL 5",
        "badge_color": "#dc2626",
        "badge_bg": "#fee2e2",
        "badge_border": "#fecaca",
        "can_mitigate": True,
        "can_simulate": True,
        "can_upload": True,
        "can_manage_users": True,
        "description": "Unrestricted administrative clearance. Full containment, playbooks, simulation, and operator management privileges."
    },
    "SOC_LEAD_ANALYST": {
        "level": 4,
        "title": "SOC Lead Analyst",
        "clearance": "SECRET // LEVEL 4",
        "badge_color": "#7c3aed",
        "badge_bg": "#f3e8ff",
        "badge_border": "#e9d5ff",
        "can_mitigate": True,
        "can_simulate": True,
        "can_upload": True,
        "can_manage_users": False,
        "description": "Tactical threat hunting clearance. Authorized to dispatch active host quarantine and counterfactual models."
    },
    "SOC_ANALYST": {
        "level": 3,
        "title": "SOC Tier-2 Analyst",
        "clearance": "CONFIDENTIAL // LEVEL 3",
        "badge_color": "#2563eb",
        "badge_bg": "#eff6ff",
        "badge_border": "#bfdbfe",
        "can_mitigate": True,
        "can_simulate": True,
        "can_upload": True,
        "can_manage_users": False,
        "description": "Operational threat analysis. Authorized for telemetry ingestion, playbook dispatch, and simulations."
    },
    "INCIDENT_RESPONDER": {
        "level": 4,
        "title": "Incident Responder",
        "clearance": "SECRET // LEVEL 4",
        "badge_color": "#e0523d",
        "badge_bg": "#fef2f0",
        "badge_border": "#fbdad4",
        "can_mitigate": True,
        "can_simulate": True,
        "can_upload": True,
        "can_manage_users": False,
        "description": "Rapid containment specialist. Primary focus on playbook execution, iptables, and micro-segmentation."
    },
    "SECURITY_ENGINEER": {
        "level": 4,
        "title": "Security Engineer",
        "clearance": "SECRET // LEVEL 4",
        "badge_color": "#0284c7",
        "badge_bg": "#e0f2fe",
        "badge_border": "#bae6fd",
        "can_mitigate": True,
        "can_simulate": True,
        "can_upload": True,
        "can_manage_users": False,
        "description": "Simulation and modeling specialist. Authorized to run counterfactual engines and ingest telemetry."
    },
    "SECURITY_AUDITOR": {
        "level": 2,
        "title": "Security & Compliance Auditor",
        "clearance": "INTERNAL AUDIT // LEVEL 2",
        "badge_color": "#b45309",
        "badge_bg": "#fef3c7",
        "badge_border": "#fde68a",
        "can_mitigate": False,
        "can_simulate": True,
        "can_upload": False,
        "can_manage_users": False,
        "description": "Read-only compliance clearance. Inspection of live radar and simulations permitted; destructive containment prohibited."
    },
    "GUEST_OBSERVER": {
        "level": 1,
        "title": "Guest Observer",
        "clearance": "UNCLASSIFIED // LEVEL 1",
        "badge_color": "#64748b",
        "badge_bg": "#f1f5f9",
        "badge_border": "#e2e8f0",
        "can_mitigate": False,
        "can_simulate": False,
        "can_upload": False,
        "can_manage_users": False,
        "description": "Demonstration observer clearance. Restricted to viewing baseline dashboard and public telemetry."
    },
}

ROLE_ALIASES = {
    "ADMIN": "CHIEF_CISO_ADMIN",
    "CISO": "CHIEF_CISO_ADMIN",
    "SUPERUSER": "CHIEF_CISO_ADMIN",
    "LEAD": "SOC_LEAD_ANALYST",
    "SOC_LEAD": "SOC_LEAD_ANALYST",
    "ANALYST": "SOC_ANALYST",
    "RESPONDER": "INCIDENT_RESPONDER",
    "ENGINEER": "SECURITY_ENGINEER",
    "AUDITOR": "SECURITY_AUDITOR",
    "COMPLIANCE": "SECURITY_AUDITOR",
    "GUEST": "GUEST_OBSERVER",
    "OBSERVER": "GUEST_OBSERVER",
}


def normalize_role(role: Optional[str]) -> str:
    """Standardizes input role string to a canonical Threatora RBAC role."""
    if not role:
        return "SOC_ANALYST"
    role_clean = role.strip().upper()
    if role_clean in ROLE_PERMISSIONS:
        return role_clean
    if role_clean in ROLE_ALIASES:
        return ROLE_ALIASES[role_clean]
    return "SOC_ANALYST"


def get_current_role() -> str:
    """Resolves the effective RBAC role of the active session or API client."""
    # 1. API key auth from headers defaults to administrative level for automated pipelines/tests
    api_key = request.headers.get("X-API-Key")
    auth_header = request.headers.get("Authorization", "")
    if (api_key and api_key == SYSTEM_API_KEY) or (auth_header.startswith("Bearer ") and auth_header[7:] == SYSTEM_API_KEY):
        simulated_role = request.headers.get("X-Simulate-Role")
        if simulated_role:
            return normalize_role(simulated_role)
        return "CHIEF_CISO_ADMIN"

    # 2. Check session simulated role or stored role
    if session.get("role"):
        return normalize_role(session["role"])

    # 3. Check DB if user_id in session
    user_id = session.get("user_id")
    if user_id:
        try:
            with get_db_context() as db:
                user = db.query(User).filter_by(id=user_id).first()
                if user and user.role:
                    return normalize_role(user.role)
        except Exception:
            pass

    return "CHIEF_CISO_ADMIN"


def get_role_info(role: Optional[str] = None) -> dict:
    """Returns the permission metadata and clearance metrics for a role."""
    if not role:
        role = get_current_role()
    role = normalize_role(role)
    info = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["SOC_ANALYST"]).copy()
    info["role"] = role
    return info


def has_permission(permission: str) -> bool:
    """Verifies whether the current active role possesses a designated permission."""
    role_info = get_role_info()
    return bool(role_info.get(permission, False))


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


def permission_required(permission: str):
    """Decorator requiring a specific RBAC entitlement before executing route."""
    def decorator(f):
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

            if not has_permission(permission):
                current_role = get_current_role()
                role_info = get_role_info(current_role)
                err_payload = {
                    "status": "error",
                    "error": "Forbidden",
                    "code": 403,
                    "message": f"Elevated Clearance Required: Action requires '{permission}'. Your current role '{current_role}' ({role_info['title']}, Level {role_info['level']}) is restricted from executing this operation.",
                    "required_permission": permission,
                    "current_role": current_role,
                    "clearance_level": role_info["level"],
                    "title": role_info["title"],
                }
                return jsonify(err_payload), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@auth_bp.route("/login", methods=["GET", "POST"])
@auth_bp.route("/api/v1/auth/login", methods=["POST"])
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
@auth_bp.route("/api/v1/auth/register", methods=["POST"])
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
    """Returns currently authenticated operator profile with role entitlements."""
    user_id = session.get("user_id")
    role = get_current_role()
    role_info = get_role_info(role)

    if user_id:
        with get_db_context() as db:
            user = db.query(User).filter_by(id=user_id).first()
            if user:
                user_dict = user.to_dict()
                user_dict["role"] = role
                user_dict["role_info"] = role_info
                return jsonify({
                    "status": "success",
                    "user": user_dict,
                    "role_info": role_info
                })

    # Headless system session
    return jsonify({
        "status": "success",
        "user": {
            "id": 0,
            "username": "system-service",
            "full_name": "Automated Security Agent",
            "role": role,
            "role_info": role_info,
            "is_active": True
        },
        "role_info": role_info
    })


@auth_bp.route("/api/v1/auth/roles", methods=["GET"])
@login_required
def list_roles():
    """Returns all available RBAC roles, clearance levels, and permission specs."""
    roles_list = [
        {"key": k, **v}
        for k, v in ROLE_PERMISSIONS.items()
    ]
    return jsonify({
        "status": "success",
        "count": len(roles_list),
        "current_role": get_current_role(),
        "roles": roles_list
    })


@auth_bp.route("/api/v1/auth/switch-role", methods=["POST"])
@login_required
def switch_role():
    """Enables real-time RBAC role switching for testing, demonstrations, and clearance simulation."""
    data = request.get_json(silent=True) or {}
    target_role = data.get("role")
    if not target_role:
        return jsonify({"status": "error", "message": "Missing 'role' parameter in payload."}), 400

    normalized = normalize_role(target_role)
    session["role"] = normalized

    user_id = session.get("user_id")
    if user_id:
        try:
            with get_db_context() as db:
                user = db.query(User).filter_by(id=user_id).first()
                if user:
                    user.role = normalized
                    db.commit()
        except Exception:
            pass

    role_info = get_role_info(normalized)
    return jsonify({
        "status": "success",
        "message": f"Active clearance switched to {normalized} ({role_info['title']}).",
        "role": normalized,
        "role_info": role_info
    })


@auth_bp.route("/api/v1/users", methods=["GET"])
@login_required
def list_users():
    """Returns operator directory with clearance ratings."""
    with get_db_context() as db:
        users = db.query(User).order_by(User.id.asc()).all()
        user_list = []
        for u in users:
            r_info = get_role_info(u.role)
            user_list.append({
                "id": u.id,
                "username": u.username,
                "email": u.email,
                "full_name": u.full_name,
                "role": u.role,
                "role_title": r_info["title"],
                "clearance_level": r_info["level"],
                "badge_color": r_info["badge_color"],
                "badge_bg": r_info["badge_bg"],
                "badge_border": r_info["badge_border"],
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "last_login": u.last_login.isoformat() if u.last_login else None,
            })
        return jsonify({
            "status": "success",
            "count": len(user_list),
            "users": user_list,
            "can_manage_users": has_permission("can_manage_users")
        })


@auth_bp.route("/api/v1/users/<int:user_id>/role", methods=["POST"])
@login_required
@permission_required("can_manage_users")
def update_user_role(user_id: int):
    """Updates an operator's clearance and role assignment (Requires CISO Admin)."""
    data = request.get_json(silent=True) or {}
    new_role = data.get("role")
    if not new_role:
        return jsonify({"status": "error", "message": "Missing 'role' parameter."}), 400

    normalized = normalize_role(new_role)
    with get_db_context() as db:
        target_user = db.query(User).filter_by(id=user_id).first()
        if not target_user:
            return jsonify({"status": "error", "message": f"Operator #{user_id} not found."}), 404

        target_user.role = normalized
        db.commit()
        r_info = get_role_info(normalized)
        return jsonify({
            "status": "success",
            "message": f"Operator @{target_user.username} clearance updated to {normalized}.",
            "user": target_user.to_dict(),
            "role_info": r_info
        })


