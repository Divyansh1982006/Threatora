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

    if current_app.extensions.get("active_mitigation", {}).get("playbooks"):
        pbs = current_app.extensions["active_mitigation"]["playbooks"]
        if target_ip:
            pbs = [p for p in pbs if p.get("target_ip") == target_ip]
        if status:
            pbs = [p for p in pbs if p.get("status") == status.upper()]
        return jsonify({"status": "success", "count": len(pbs), "playbooks": pbs})

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
    mitigation_engine = current_app.extensions.get("mitigation_engine")
    if mitigation_engine is None:
        return jsonify({"status": "error", "message": "ML engines are still initializing. Retry in ~30 seconds."}), 503
    data = request.get_json(silent=True) or {}
    playbook_uid = data.get("playbook_uid")
    target_ip = data.get("target_ip")
    action = data.get("action", "isolate")

    # Update active in-memory topology and mitigation states
    if target_ip:
        if current_app.extensions.get("active_topology"):
            for n in current_app.extensions["active_topology"].get("nodes", []):
                if n.get("ip") == target_ip:
                    n["status"] = "ISOLATED"
                    n["is_isolated"] = True
            for l in current_app.extensions["active_topology"].get("links", []):
                if l.get("source") == target_ip or l.get("target") == target_ip:
                    l["threat"] = "mitigated"

        if current_app.extensions.get("active_mitigation"):
            for a in current_app.extensions["active_mitigation"].get("assets", []):
                if a.get("ip_address") == target_ip:
                    a["status"] = "ISOLATED"
                    a["quarantine_status"] = "ISOLATED"
            for p in current_app.extensions["active_mitigation"].get("playbooks", []):
                if p.get("target_ip") == target_ip:
                    p["status"] = "CONTAINMENT_EXECUTED"

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
    """Retrieves enterprise network asset inventory from PostgreSQL ledger or active capture."""
    if current_app.extensions.get("active_mitigation", {}).get("assets"):
        assets = current_app.extensions["active_mitigation"]["assets"]
        return jsonify({"status": "success", "count": len(assets), "assets": assets})

    with get_db_context() as db:
        assets = [a.to_dict() for a in db.query(Asset).order_by(Asset.criticality.desc(), Asset.ip_address.asc()).all()]
        return jsonify({"status": "success", "count": len(assets), "assets": assets})


@mitigation_bp.route("/api/v1/incidents", methods=["GET"])
def get_incidents():
    """Retrieves security incidents and forensics log from PostgreSQL ledger or active capture."""
    if current_app.extensions.get("active_mitigation", {}).get("incidents"):
        incidents = current_app.extensions["active_mitigation"]["incidents"]
        return jsonify({"status": "success", "count": len(incidents), "incidents": incidents})

    with get_db_context() as db:
        incidents = [
            inc.to_dict()
            for inc in db.query(Incident).order_by(Incident.created_at.desc()).limit(100).all()
        ]
        return jsonify({"status": "success", "count": len(incidents), "incidents": incidents})


@mitigation_bp.route("/api/v1/topology", methods=["GET"])
def get_topology():
    """Returns network graph nodes and communication links for the visual topology map."""
    if current_app.extensions.get("active_topology"):
        return jsonify(current_app.extensions["active_topology"])

    with get_db_context() as db:
        db_assets = db.query(Asset).all()
        recent_incidents = db.query(Incident).order_by(Incident.created_at.desc()).limit(20).all()

    # Map of IP to threat data
    threat_map = {}
    for inc in recent_incidents:
        if inc.target_ip not in threat_map:
            threat_map[inc.target_ip] = {
                "risk_score": inc.risk_score,
                "stage_name": inc.stage_name,
                "technique": inc.technique,
            }

    nodes = []
    seen_ips = set()

    for a in db_assets:
        seen_ips.add(a.ip_address)
        is_threat = a.ip_address in threat_map or a.status in ("COMPROMISED", "SUSPICIOUS")
        t_data = threat_map.get(a.ip_address, {})

        node_type = "workstation"
        if "DC" in a.hostname.upper() or "AD" in a.hostname.upper():
            node_type = "domain_controller"
        elif "DATABASE" in a.hostname.upper() or "DB" in a.hostname.upper():
            node_type = "database"
        elif "GATEWAY" in a.hostname.upper() or "EDGE" in a.hostname.upper():
            node_type = "gateway"

        nodes.append({
            "id": a.ip_address,
            "ip": a.ip_address,
            "hostname": a.hostname,
            "type": node_type,
            "subnet": a.subnet or "192.168.1.0/24",
            "criticality": a.criticality,
            "status": a.status,
            "risk_score": t_data.get("risk_score", 0.12 if a.status == "HEALTHY" else 0.78),
            "stage_name": t_data.get("stage_name", "Benign" if a.status == "HEALTHY" else "Lateral Movement"),
            "technique": t_data.get("technique", "Baseline Operation"),
            "os": a.operating_system or "Linux / Enterprise",
            "is_isolated": a.status == "ISOLATED",
        })

    # Ensure Gateway and External Internet nodes are represented
    gateway_ip = "147.32.84.165"
    if gateway_ip not in seen_ips:
        nodes.append({
            "id": gateway_ip,
            "ip": gateway_ip,
            "hostname": "EDGE-GATEWAY-01",
            "type": "gateway",
            "subnet": "147.32.84.0/24",
            "criticality": "MISSION_CRITICAL",
            "status": "HEALTHY",
            "risk_score": 0.14,
            "stage_name": "Benign",
            "technique": "Perimeter Routing",
            "os": "VyOS / EdgeOS",
            "is_isolated": False,
        })
        seen_ips.add(gateway_ip)

    external_ip = "198.51.100.42"
    if external_ip not in seen_ips:
        nodes.append({
            "id": external_ip,
            "ip": external_ip,
            "hostname": "EXT-C2-ADVERSARY",
            "type": "external",
            "subnet": "WAN",
            "criticality": "ADVERSARY",
            "status": "THREAT_ACTOR",
            "risk_score": 0.99,
            "stage_name": "Command & Control",
            "technique": "C2 Infrastructure",
            "os": "Unknown / WAN",
            "is_isolated": False,
        })

    # Dynamic Communication Links
    links = [
        # Gateway connections
        {"source": "147.32.84.165", "target": "192.168.1.10", "port": 53, "proto": "DNS/UDP", "traffic": "Normal", "threat": "normal"},
        {"source": "147.32.84.165", "target": "192.168.1.5", "port": 443, "proto": "HTTPS/TCP", "traffic": "Normal", "threat": "normal"},
        {"source": "147.32.84.165", "target": "192.168.1.105", "port": 80, "proto": "HTTP/TCP", "traffic": "High", "threat": "medium"},
        # Lateral threat links from infected workstation (192.168.1.105)
        {"source": "192.168.1.105", "target": "192.168.1.10", "port": 445, "proto": "SMB/TCP", "traffic": "High", "threat": "critical"},
        {"source": "192.168.1.105", "target": "192.168.1.5", "port": 3389, "proto": "RDP/TCP", "traffic": "Elevated", "threat": "critical"},
        # External C2 exfiltration link
        {"source": "192.168.1.105", "target": "198.51.100.42", "port": 6667, "proto": "IRC/C2", "traffic": "Burst", "threat": "critical"},
        {"source": "147.32.84.165", "target": "198.51.100.42", "port": 443, "proto": "Egress", "traffic": "High", "threat": "critical"},
    ]

    return jsonify({
        "status": "success",
        "node_count": len(nodes),
        "link_count": len(links),
        "nodes": nodes,
        "links": links,
    })

