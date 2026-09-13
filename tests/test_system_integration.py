"""Comprehensive End-to-End Zero-Trust Integration Test for Threatora.

Validates:
  1. Database initialization and asset inventory seeding
  2. Inference pipeline and Anomaly / Kill Chain prediction
  3. Mitigation Engine playbook compilation and damage assessment
  4. Flask REST API v1 endpoints (/health, /demo, /playbooks, /mitigate, /assets, /incidents)
  5. 1-Click mitigation host isolation state transitions
"""

import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.session import init_db, get_db_context
from src.db.models import Asset, Incident, MitigationPlaybook
from src.mitigation import MitigationEngine
from server.app import app


def run_integration_audit():
    print("=" * 70)
    print(" [+] THREATORA SYSTEM INTEGRATION & ZERO-TRUST AUDIT")
    print("=" * 70)

    # 1. Database Initialization
    print("[*] 1. Initializing State Store & Asset Inventory...")
    init_db()
    with get_db_context() as db:
        assets = db.query(Asset).all()
        assert len(assets) >= 5, f"Expected >= 5 assets, found {len(assets)}"
        print(f"  [+] State Store seeded: {len(assets)} enterprise assets verified.")

    # 2. Flask Test Client
    print("[*] 2. Testing API Gateway Endpoints...")
    client = app.test_client()

    # Health Check
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.data}"
    health_data = health_resp.get_json()
    assert health_data["status"] == "healthy"
    print(f"  [+] /api/health passed. Device: {health_data.get('device')}")

    # Assets Query
    assets_resp = client.get("/api/v1/assets")
    assert assets_resp.status_code == 200
    assets_data = assets_resp.get_json()
    assert assets_data["count"] >= 5
    print(f"  [+] /api/v1/assets passed. Total assets returned: {assets_data['count']}")

    # Live Telemetry Ingestion & Mitigation Synthesis
    print("[*] 3. Testing Ingestion, LSTM Inference & Playbook Synthesis...")
    demo_resp = client.get("/api/demo")
    assert demo_resp.status_code == 200, f"Demo failed: {demo_resp.data}"
    demo_data = demo_resp.get_json()
    assert demo_data["status"] == "success"
    assert "hosts" in demo_data and len(demo_data["hosts"]) > 0
    assert "playbooks" in demo_data and len(demo_data["playbooks"]) > 0

    lead_host = demo_data["hosts"][0]
    lead_pb = demo_data["playbooks"][0]
    print(f"  [+] Anomaly Inference Passed. Flagged Host: {lead_host['host_ip']} (Risk: {lead_host['current_risk_score']*100:.1f}%)")
    print(f"  [+] ATT&CK Stage: {lead_host['current_stage']['name']}")
    print(f"  [+] Mitigation Playbook Synthesized: {lead_pb['playbook_uid']} ({lead_pb['kill_chain_stage']})")
    print(f"      Damage Assessment: {lead_pb['damage_assessment'][:80]}...")

    # Query Playbooks via REST
    pb_resp = client.get("/api/v1/playbooks")
    assert pb_resp.status_code == 200
    pb_data = pb_resp.get_json()
    assert pb_data["count"] > 0
    print(f"  [+] /api/v1/playbooks passed. Total playbooks: {pb_data['count']}")

    # Execute Mitigation
    print("[*] 4. Testing 1-Click Zero-Trust Containment Dispatch...")
    headers = {"X-API-Key": "threatora-zero-trust"}
    target_pb_uid = lead_pb["playbook_uid"]
    mitigate_resp = client.post("/api/v1/mitigate", json={"playbook_uid": target_pb_uid}, headers=headers)
    assert mitigate_resp.status_code == 200
    mit_data = mitigate_resp.get_json()
    assert mit_data["status"] == "success"
    assert mit_data["host_status"] == "ISOLATED"
    print(f"  [+] /api/v1/mitigate successfully isolated {mit_data['target_ip']}.")
    print(f"      Audit Log: {mit_data['execution_log']}")

    # Verify Asset state in Database
    with get_db_context() as db:
        updated_asset = db.query(Asset).filter(Asset.ip_address == mit_data["target_ip"]).first()
        assert updated_asset is not None
        assert updated_asset.status == "ISOLATED"
        print(f"  [+] Database verified: Asset {updated_asset.ip_address} status is {updated_asset.status}.")

    # Query Incidents
    inc_resp = client.get("/api/v1/incidents")
    assert inc_resp.status_code == 200
    inc_data = inc_resp.get_json()
    assert inc_data["count"] > 0
    print(f"  [+] /api/v1/incidents passed. Total recorded incidents: {inc_data['count']}")

    # 5. What-If Counterfactual Action Simulation Engine
    print("[*] 5. Testing What-If Counterfactual Simulation Engine...")
    actions_resp = client.get("/api/v1/simulate/actions")
    assert actions_resp.status_code == 200
    actions_data = actions_resp.get_json()
    assert actions_data["count"] >= 5
    print(f"  [+] /api/v1/simulate/actions passed. Total actions available: {actions_data['count']}")

    sim_resp = client.post("/api/v1/simulate", json={
        "action": "BLOCK_MANAGEMENT_PORTS",
        "target_ip": lead_host['host_ip'],
        "horizon": 10
    }, headers=headers)
    assert sim_resp.status_code == 200
    sim_data = sim_resp.get_json()
    assert sim_data["status"] == "success"
    assert "pct_risk_reduction" in sim_data
    assert len(sim_data["timeline"]) == 10
    print(f"  [+] /api/v1/simulate passed for target {sim_data['target_ip']}.")
    print(f"      Action: {sim_data['action_name']} -> Risk Reduction: {sim_data['pct_risk_reduction']}% ({sim_data['effectiveness']})")
    print(f"      Verdict: {sim_data['tactical_verdict'][:80]}...")

    print("\n" + "=" * 70)
    print(" [+] ALL ZERO-TRUST INTEGRATION TESTS PASSED WITH 100% INTEGRITY")
    print("=" * 70)


if __name__ == "__main__":
    run_integration_audit()
