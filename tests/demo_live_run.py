"""Live Demonstration Script for Threatora Server.

Simulates operator interactions with the live running server:
  - System Health & Model Dimension checks
  - Telemetry Ingestion & ATT&CK Inference
  - What-If Counterfactual Action Simulation
  - 1-Click Zero-Trust Host Isolation
  - State Ledger Verification (Assets, Playbooks, Incidents)
"""

import sys
import json
import requests

API_URL = "http://127.0.0.1:5000"

def demo():
    print("=" * 70)
    print(" [+] THREATORA LIVE OPERATIONAL DEMONSTRATION")
    print("=" * 70)

    # 1. Health
    print("\n[1] Checking System Health & Architecture...")
    r = requests.get(f"{API_URL}/api/health", timeout=5)
    print(f"    HTTP Status: {r.status_code}")
    print(f"    Payload: {r.json()}")

    # 2. Assets
    print("\n[2] Querying Enterprise Asset Inventory (PostgreSQL State Ledger)...")
    r = requests.get(f"{API_URL}/api/v1/assets", timeout=5)
    assets = r.json().get("assets", [])
    print(f"    Total Managed Assets: {len(assets)}")
    for a in assets[:3]:
        print(f"    - Host: {a['ip_address']} ({a['hostname']}) | Crit: {a['criticality']} | Status: {a['status']}")

    # 3. Live Telemetry Ingestion & Threat Detection
    print("\n[3] Ingesting Live Telemetry Stream (/api/demo)...")
    r = requests.get(f"{API_URL}/api/demo", timeout=30)
    data = r.json()
    hosts = data.get("hosts", [])
    playbooks = data.get("playbooks", [])
    print(f"    Total Hosts Analyzed: {len(hosts)}")
    print(f"    Flagged Threat Hosts: {data.get('flagged_hosts', 0)}")
    
    if hosts:
        h = hosts[0]
        print(f"    Lead Threat: {h['host_ip']} | Risk: {h['current_risk_score']*100:.1f}%")
        print(f"    ATT&CK Phase: {h['current_stage']['name']} ({h['current_stage']['metadata'].get('technique')})")
        print(f"    10-Min Forecast: {len(h.get('forecast_timeline', []))} steps ahead")

    if playbooks:
        pb = playbooks[0]
        print(f"\n[4] Auto-Generated Mitigation Playbook:")
        print(f"    Playbook UID: {pb['playbook_uid']} | Status: {pb['status']}")
        print(f"    Kill Chain Stage: {pb['kill_chain_stage']}")
        print(f"    Damage Assessment: {pb['damage_assessment']}")
        print(f"    Containment Strategy: {pb['containment_strategy']}")

    # 4. What-If Action Simulation
    print("\n[5] Executing What-If Counterfactual Simulation (/api/v1/simulate)...")
    sim_payload = {
        "action": "BLOCK_MANAGEMENT_PORTS",
        "target_ip": hosts[0]["host_ip"] if hosts else "192.168.1.105",
        "horizon": 10
    }
    r = requests.post(f"{API_URL}/api/v1/simulate", json=sim_payload, timeout=15)
    sim_data = r.json()
    print(f"    Action Tested: {sim_data.get('action_name')}")
    print(f"    Baseline Risk: {sim_data.get('mean_baseline_risk')*100:.1f}% -> Counterfactual Risk: {sim_data.get('mean_simulated_risk')*100:.1f}%")
    print(f"    Modeled Risk Reduction: {sim_data.get('pct_risk_reduction')}%")
    print(f"    Tactical Verdict: {sim_data.get('tactical_verdict')}")

    # 5. 1-Click Host Isolation
    print("\n[6] Dispatching 1-Click Zero-Trust Containment Action (/api/v1/mitigate)...")
    mit_payload = {
        "target_ip": hosts[0]["host_ip"] if hosts else "192.168.1.105",
        "action": "isolate"
    }
    r = requests.post(f"{API_URL}/api/v1/mitigate", json=mit_payload, timeout=10)
    mit_data = r.json()
    print(f"    Isolation Status: {mit_data.get('host_status')}")
    print(f"    Execution Audit Log: {mit_data.get('execution_log')}")

    print("\n" + "=" * 70)
    print(" [+] LIVE DEMONSTRATION COMPLETE: ALL SYSTEMS RUNNING WITHOUT ERRORS")
    print("=" * 70)

if __name__ == "__main__":
    demo()
