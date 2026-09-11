"""Automated Threat Mitigation & Asset Ledger Blueprint."""

from __future__ import annotations

from flask import Blueprint, request, jsonify, current_app
from src.db.session import get_db_context
from src.db.models import Asset, Incident, MitigationPlaybook
from server.blueprints.auth import login_required, permission_required

mitigation_bp = Blueprint("mitigation", __name__)


@mitigation_bp.route("/api/v1/playbooks", methods=["GET"])
def get_playbooks():
    """Returns all mitigation playbooks with optional target or status filtering."""
    target_ip = request.args.get("target")
    status = request.args.get("status")

    with get_db_context() as db:
        query = db.query(MitigationPlaybook)
        if target_ip:
            query = query.filter(MitigationPlaybook.target_ip == target_ip)
        if status:
            query = query.filter(MitigationPlaybook.status == status.upper())

        playbooks = [pb.to_dict() for pb in query.order_by(MitigationPlaybook.created_at.desc()).limit(100).all()]
        return jsonify({"status": "success", "count": len(playbooks), "playbooks": playbooks})


@mitigation_bp.route("/api/v1/mitigate", methods=["POST"])
@login_required
@permission_required("can_mitigate")
def execute_mitigate():
    """Dispatches 1-click mitigation containment action against a target or playbook."""
    mitigation_engine = current_app.extensions["mitigation_engine"]
    data = request.get_json(silent=True) or {}
    playbook_uid = data.get("playbook_uid")
    target_ip = data.get("target_ip")
    action = data.get("action", "isolate")

    if playbook_uid:
        result = mitigation_engine.execute_playbook(playbook_uid)
        return jsonify(result), (200 if result.get("status") in ("success", "warning") else 400)

    if target_ip and action == "isolate":
        with get_db_context() as db:
            pb = db.query(MitigationPlaybook).filter(
                MitigationPlaybook.target_ip == target_ip
            ).order_by(MitigationPlaybook.created_at.desc()).first()

            if pb:
                result = mitigation_engine.execute_playbook(pb.playbook_uid)
                return jsonify(result)

            asset = db.query(Asset).filter(Asset.ip_address == target_ip).first()
            if not asset:
                asset = Asset(ip_address=target_ip, hostname=f"host-{target_ip.replace('.', '-')}", status="ISOLATED")
                db.add(asset)
            else:
                asset.status = "ISOLATED"
            db.commit()

            return jsonify({
                "status": "success",
                "message": f"Host {target_ip} manually isolated in asset inventory.",
                "target_ip": target_ip,
                "host_status": "ISOLATED",
                "execution_log": f"Manual operator isolation action dispatched for {target_ip}."
            })

    return jsonify({"status": "error", "message": "Must provide 'playbook_uid' or 'target_ip' with action 'isolate'."}), 400


@mitigation_bp.route("/api/v1/assets", methods=["GET"])
def get_assets():
    """Retrieves enterprise network asset inventory from PostgreSQL ledger."""
    with get_db_context() as db:
        assets = [a.to_dict() for a in db.query(Asset).order_by(Asset.criticality.desc(), Asset.ip_address.asc()).all()]
        return jsonify({"status": "success", "count": len(assets), "assets": assets})


@mitigation_bp.route("/api/v1/incidents", methods=["GET"])
def get_incidents():
    """Retrieves security incidents and forensics log from PostgreSQL ledger."""
    with get_db_context() as db:
        incidents = [
            inc.to_dict()
            for inc in db.query(Incident).order_by(Incident.created_at.desc()).limit(100).all()
        ]
        return jsonify({"status": "success", "count": len(incidents), "incidents": incidents})
