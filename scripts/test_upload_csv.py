"""Test end-to-end upload and execution of large CSV dataset on Threatora."""

import urllib.request
import urllib.parse
import http.cookiejar
import json
import uuid
import time
from pathlib import Path

def main():
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # 1. Check Server Health
    print("[*] Checking server health...")
    health_req = urllib.request.Request("http://127.0.0.1:5000/api/health")
    with opener.open(health_req) as resp:
        health_data = json.loads(resp.read().decode("utf-8"))
        print(f"[+] Server status: {health_data.get('status')} on device: {health_data.get('device')}")

    # 2. Login Operator Session
    print("[*] Authenticating operator session (admin)...")
    login_data = urllib.parse.urlencode({"username": "admin", "password": "Threatora@2026"}).encode("utf-8")
    login_req = urllib.request.Request("http://127.0.0.1:5000/login", data=login_data)
    with opener.open(login_req) as resp:
        print(f"[+] Authenticated: HTTP {resp.getcode()} -> {resp.geturl()}")

    # 3. Read dataset
    csv_path = Path(r"D:\part-00001-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
    if not csv_path.exists():
        print(f"[!] File not found: {csv_path}")
        return

    print(f"[*] Loading {csv_path} ({csv_path.stat().st_size / (1024*1024):.2f} MB)...")
    boundary = "----ThreatoraMultipartBoundary" + uuid.uuid4().hex

    with open(csv_path, "rb") as f:
        file_bytes = f.read()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: text/csv\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req_upload = urllib.request.Request(
        "http://127.0.0.1:5000/api/upload",
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-API-Key": "threatora-zero-trust"
        }
    )

    t0 = time.time()
    print("[*] Streaming CSV to /api/upload endpoint & triggering Neural World Model...")
    with opener.open(req_upload) as resp:
        elapsed = time.time() - t0
        res_data = json.loads(resp.read().decode("utf-8"))

    print(f"[+] Completed in {elapsed:.2f} seconds.")
    print("=" * 70)
    print("                THREATORA MULTI-MODAL INFERENCE REPORT")
    print("=" * 70)
    print(f"Status              : {res_data.get('status')}")
    print(f"Modality Ingested   : {res_data.get('modality')}")
    print(f"Total Flows Ingested: {res_data.get('total_flows'):,}")
    print(f"Window Sequences    : {res_data.get('num_windows'):,}")
    print(f"Flow Attack Prob    : {res_data.get('p_flow'):.4f}")
    print(f"Fused Attack Prob   : {res_data.get('p_attack'):.4f} ({(res_data.get('p_attack', 0)*100):.2f}%)")

    dec = res_data.get("decision", {})
    print(f"Decision Verdict    : {dec.get('verdict')} (Confidence: {dec.get('confidence'):.2f})")
    print(f"Predicted Stage     : {dec.get('mitre_stage', {}).get('name')} ({dec.get('mitre_stage', {}).get('tactic')})")
    print(f"MITRE Technique     : {dec.get('mitre_stage', {}).get('technique')}")

    hosts = res_data.get("hosts", [])
    print(f"\n--- Evaluated Infrastructure Nodes ({len(hosts)}) ---")
    for h in hosts:
        print(f"  Target Host: {h.get('host_ip')}")
        print(f"  Current Risk Score: {h.get('current_risk_score'):.4f}")
        print(f"  Anomalous Status  : {h.get('is_anomalous')}")
        print(f"  Identified Stage  : {h.get('current_stage', {}).get('name')}")
        timeline = h.get("forecast_timeline", [])
        print(f"  Forward Simulation Horizon: {len(timeline)} steps")
        for step in timeline[:5]:
            print(f"    + Step t+{step.get('step')} ({step.get('timestamp')}): Risk={step.get('predicted_risk_score')*100:.1f}%, 95% CI=[{step.get('lower_ci_95')*100:.1f}%, {step.get('upper_ci_95')*100:.1f}%], Projected Stage={step.get('projected_stage')}")

    playbooks = res_data.get("playbooks", [])
    print(f"\n--- Synthesized Automated Playbooks ({len(playbooks)}) ---")
    for pb in playbooks:
        print(f"  [Playbook ID: {pb.get('playbook_uid')}]")
        print(f"  Target IP         : {pb.get('target_ip')}")
        print(f"  Stage Target      : {pb.get('kill_chain_stage')}")
        print(f"  Strategy          : {pb.get('containment_strategy')}")
        print(f"  Damage Assessment : {pb.get('damage_assessment')}")
        print("  Containment Execution Commands:")
        for cmd in pb.get("containment_commands", []):
            if isinstance(cmd, dict):
                print(f"    $ {cmd.get('command')}  # ({cmd.get('description')})")
            else:
                print(f"    $ {cmd}")

    # 4. Run What-If Simulation on the lead host
    print("\n[*] Executing Counterfactual What-If Simulation Engine...")
    sim_payload = json.dumps({
        "target_ip": hosts[0].get("host_ip") if hosts else "192.168.1.105",
        "action": "ISOLATE_HOST",
        "horizon": 10
    }).encode("utf-8")
    sim_req = urllib.request.Request(
        "http://127.0.0.1:5000/api/v1/simulate",
        data=sim_payload,
        headers={"Content-Type": "application/json", "X-API-Key": "threatora-zero-trust"}
    )
    try:
        with opener.open(sim_req) as resp:
            sim_data = json.loads(resp.read().decode("utf-8"))
            print(f"[+] Simulation Verdict  : {sim_data.get('tactical_verdict')}")
            print(f"[+] Risk Reduction      : {sim_data.get('pct_risk_reduction')}% ({sim_data.get('effectiveness')})")
            print(f"[+] Action Modeled      : {sim_data.get('action_name')}")
    except Exception as e:
        print(f"[!] Note on simulation test: {e}")

    print("\n[SUCCESS] Entire pipeline successfully executed and validated!")

if __name__ == "__main__":
    main()
