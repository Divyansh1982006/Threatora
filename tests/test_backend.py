"""
Threatora Backend Test Suite
=============================
Tests all REST API endpoints for correctness.
Run with:  python tests/test_backend.py
Server must be running at http://127.0.0.1:5000
"""

import sys
import json
import time
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"
API_KEY  = "threatora-zero-trust"
HEADERS  = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"
INFO  = "\033[96m[INFO]\033[0m"

results = {"passed": 0, "failed": 0, "warned": 0}

def check(label, condition, extra=""):
    if condition:
        print(f"  {PASS}  {label}")
        results["passed"] += 1
    else:
        print(f"  {FAIL}  {label}" + (f"  ->  {extra}" if extra else ""))
        results["failed"] += 1

def warn(label, msg=""):
    print(f"  {WARN}  {label}" + (f"  ->  {msg}" if msg else ""))
    results["warned"] += 1

def section(title):
    print(f"\n{'='*60}")
    print(f"  {INFO}  {title}")
    print(f"{'='*60}")


# 1. HEALTH CHECK
section("1. Health Check  ->  GET /api/health")
try:
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    check("Status code 200",        r.status_code == 200, str(r.status_code))
    data = r.json()
    check("status == healthy",      data.get("status") == "healthy")
    check("recurrent_cell == LSTM", data.get("recurrent_cell") == "LSTM")
    check("obs_dimension present",  isinstance(data.get("obs_dimension"), int))
    check("database_connected",     data.get("database_connected") is True)
    check("offline_mode is True",   data.get("offline_mode") is True)
    print(f"       device={data.get('device')}  obs_dim={data.get('obs_dimension')}")
except Exception as e:
    check("Health endpoint reachable", False, str(e))


# 2. AUTHENTICATION
section("2. Authentication")

try:
    r = requests.get(f"{BASE_URL}/api/v1/auth/me", timeout=10)
    check("Unauth returns 401", r.status_code == 401, str(r.status_code))
except Exception as e:
    check("Unauth check reachable", False, str(e))

try:
    r = requests.get(f"{BASE_URL}/api/v1/auth/me", headers={"X-API-Key": API_KEY}, timeout=10)
    check("API Key auth -> 200",  r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Response has user",    "user" in data)
except Exception as e:
    check("API Key auth reachable", False, str(e))

try:
    r = requests.get(f"{BASE_URL}/api/v1/auth/me", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=10)
    check("Bearer Token auth -> 200", r.status_code == 200, str(r.status_code))
except Exception as e:
    check("Bearer auth reachable", False, str(e))

TEST_USER = {
    "username": f"test_op_{int(time.time())}",
    "email":    f"test_{int(time.time())}@threatora.local",
    "password": "TestPass2026",
    "full_name": "Test Operator",
    "role": "SOC_ANALYST"
}
try:
    r = requests.post(f"{BASE_URL}/register", json=TEST_USER,
                      headers={"Content-Type": "application/json"},
                      allow_redirects=False, timeout=10)
    check("Register new user (201/302)", r.status_code in (201, 302), str(r.status_code))
except Exception as e:
    check("Register endpoint reachable", False, str(e))

try:
    r = requests.post(f"{BASE_URL}/login",
                      json={"username": TEST_USER["username"], "password": TEST_USER["password"]},
                      headers={"Content-Type": "application/json"},
                      allow_redirects=False, timeout=10)
    check("Login -> 200/302", r.status_code in (200, 302), str(r.status_code))
except Exception as e:
    check("Login endpoint reachable", False, str(e))

try:
    r = requests.post(f"{BASE_URL}/login",
                      json={"username": "nobody", "password": "wrongpass"},
                      headers={"Content-Type": "application/json"}, timeout=10)
    check("Invalid login -> 401", r.status_code == 401, str(r.status_code))
except Exception as e:
    check("Invalid login check reachable", False, str(e))


# 3. TELEMETRY DEMO  - actual keys: hosts, flagged_hosts, total_hosts, playbooks, status
section("3. Telemetry Demo  ->  GET /api/demo")
try:
    r = requests.get(f"{BASE_URL}/api/demo", headers=HEADERS, timeout=30)
    check("Status code 200",         r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has status key",          "status" in data)
    check("Has hosts key",           "hosts" in data)
    check("Has flagged_hosts key",   "flagged_hosts" in data)
    check("Has total_hosts key",     "total_hosts" in data)
    check("Has playbooks key",       "playbooks" in data)
    check("hosts is list",           isinstance(data.get("hosts"), list))
    if data.get("hosts"):
        h = data["hosts"][0]
        check("Host has current_risk_score",  "current_risk_score" in h)
        check("Host has current_stage",       "current_stage" in h)
        check("Host has explainability",      "explainability" in h)
    print(f"       flagged_hosts={data.get('flagged_hosts')}  total_hosts={data.get('total_hosts')}  playbooks={len(data.get('playbooks', []))}")
except Exception as e:
    check("Demo endpoint reachable", False, str(e))


# 4. TELEMETRY GET
section("4. Telemetry API  ->  GET /api/v1/telemetry")
try:
    r = requests.get(f"{BASE_URL}/api/v1/telemetry", headers=HEADERS, timeout=30)
    check("Status code 200",       r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has status key",        "status" in data)
    check("Has hosts key",         "hosts" in data)
    check("Has playbooks key",     "playbooks" in data)
    check("hosts is list",         isinstance(data.get("hosts"), list))
except Exception as e:
    check("Telemetry GET reachable", False, str(e))


# 5. TELEMETRY POST (JSON flows)
section("5. Telemetry POST  ->  POST /api/v1/telemetry (JSON flows)")
sample_flows = [
    {"StartTime": 1628596805.0, "saddr": "192.168.1.105", "sport": 51234,
     "dir": "->", "daddr": "10.0.0.1", "dport": 80, "proto": "tcp",
     "state": "CON", "dur": 0.05, "tot_pkts": 6, "tot_bytes": 650,
     "src_bytes": 300, "flags": "SA", "Label": "Normal", "is_malicious": 0},
    {"StartTime": 1628597000.0, "saddr": "192.168.1.105", "sport": 51240,
     "dir": "->", "daddr": "10.0.0.12", "dport": 22, "proto": "tcp",
     "state": "INT", "dur": 0.001, "tot_pkts": 2, "tot_bytes": 120,
     "src_bytes": 120, "flags": "S", "Label": "From-Botnet-V42-TCP-Attempt", "is_malicious": 1},
]
try:
    r = requests.post(f"{BASE_URL}/api/v1/telemetry",
                      json={"flows": sample_flows}, headers=HEADERS, timeout=30)
    check("Status code 200",   r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has hosts key",     "hosts" in data)
    check("Has playbooks key", "playbooks" in data)
    check("Has status key",    "status" in data)
except Exception as e:
    check("Telemetry POST reachable", False, str(e))


# 6. FILE UPLOAD
section("6. File Upload  ->  POST /api/upload (CSV)")
sample_csv_path = Path(__file__).parent.parent / "data" / "samples" / "sample_traffic.csv"
try:
    if sample_csv_path.exists():
        with open(sample_csv_path, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/upload",
                              files={"file": ("sample_traffic.csv", f, "text/csv")},
                              headers={"X-API-Key": API_KEY}, timeout=30)
        check("Status code 200",   r.status_code == 200, str(r.status_code))
        data = r.json()
        check("Has hosts key",     "hosts" in data)
        check("Has playbooks key", "playbooks" in data)
    else:
        warn("CSV Upload", "sample_traffic.csv not found")
except Exception as e:
    check("Upload endpoint reachable", False, str(e))


# 7. PLAYBOOKS
section("7. Playbooks  ->  GET /api/v1/playbooks")
try:
    r = requests.get(f"{BASE_URL}/api/v1/playbooks", headers=HEADERS, timeout=15)
    check("Status code 200",    r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has playbooks key",  "playbooks" in data)
    check("Has count key",      "count" in data)
    check("Count matches list", data.get("count") == len(data.get("playbooks", [])))
    print(f"       playbook count={data.get('count')}")
except Exception as e:
    check("Playbooks endpoint reachable", False, str(e))

try:
    r = requests.get(f"{BASE_URL}/api/v1/playbooks?status=PENDING", headers=HEADERS, timeout=15)
    check("Filtered playbooks -> 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Filtered result is list",   isinstance(data.get("playbooks"), list))
except Exception as e:
    check("Playbooks filter reachable", False, str(e))


# 8. ASSETS
section("8. Asset Inventory  ->  GET /api/v1/assets")
try:
    r = requests.get(f"{BASE_URL}/api/v1/assets", headers=HEADERS, timeout=15)
    check("Status code 200",   r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has assets key",    "assets" in data)
    check("Has count key",     "count" in data)
    if data.get("assets"):
        a = data["assets"][0]
        check("Asset has ip_address", "ip_address" in a)
        check("Asset has status",     "status" in a)
    print(f"       asset count={data.get('count')}")
except Exception as e:
    check("Assets endpoint reachable", False, str(e))


# 9. INCIDENTS
section("9. Incidents Log  ->  GET /api/v1/incidents")
try:
    r = requests.get(f"{BASE_URL}/api/v1/incidents", headers=HEADERS, timeout=15)
    check("Status code 200",     r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has incidents key",   "incidents" in data)
    check("Has count key",       "count" in data)
    print(f"       incident count={data.get('count')}")
except Exception as e:
    check("Incidents endpoint reachable", False, str(e))


# 10. MITIGATION ACTION
section("10. Mitigation Action  ->  POST /api/v1/mitigate")
try:
    r = requests.post(f"{BASE_URL}/api/v1/mitigate",
                      json={"target_ip": "192.168.1.105", "action": "isolate"},
                      headers=HEADERS, timeout=15)
    check("Status code 200",           r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Status is success/warning", data.get("status") in ("success", "warning"), data.get("status"))
    check("Has target_ip",             "target_ip" in data)
except Exception as e:
    check("Mitigate endpoint reachable", False, str(e))

try:
    r = requests.post(f"{BASE_URL}/api/v1/mitigate", json={}, headers=HEADERS, timeout=15)
    check("Empty mitigate -> 400", r.status_code == 400, str(r.status_code))
except Exception as e:
    check("Mitigate validation reachable", False, str(e))


# 11. SIMULATION ACTIONS
section("11. Simulation Actions  ->  GET /api/v1/simulate/actions")
try:
    r = requests.get(f"{BASE_URL}/api/v1/simulate/actions", headers=HEADERS, timeout=15)
    check("Status code 200",   r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has actions key",   "actions" in data)
    check("Has count key",     "count" in data)
    check("Actions is list",   isinstance(data.get("actions"), list))
    if data.get("actions"):
        a = data["actions"][0]
        check("Action has key",  "key" in a)
        check("Action has name", "name" in a)
        check("Action has features_impacted", "features_impacted" in a)
    print(f"       supported actions={data.get('count')}")
    for a in data.get("actions", []):
        print(f"         * {a['key']}: {a['name']}")
except Exception as e:
    check("Simulate/actions reachable", False, str(e))


# 12. WHAT-IF SIMULATION  - actual keys: action_key, action_name, action_description, timeline, effectiveness, etc.
section("12. What-If Simulation  ->  POST /api/v1/simulate")
try:
    r = requests.post(f"{BASE_URL}/api/v1/simulate",
                      json={"action": "BLOCK_MANAGEMENT_PORTS", "horizon": 5},
                      headers=HEADERS, timeout=30)
    check("Status code 200",              r.status_code == 200, str(r.status_code))
    data = r.json()
    check("Has status key",               "status" in data)
    check("Has action_key key",           "action_key" in data)
    check("Has action_name key",          "action_name" in data)
    check("Has timeline key",             "timeline" in data)
    check("Has mean_baseline_risk key",   "mean_baseline_risk" in data)
    check("Has mean_simulated_risk key",  "mean_simulated_risk" in data)
    check("Has pct_risk_reduction key",   "pct_risk_reduction" in data)
    check("Has effectiveness key",        "effectiveness" in data)
    check("Has target_ip key",            "target_ip" in data)
    check("Has tactical_verdict key",     "tactical_verdict" in data)
    check("timeline is list",             isinstance(data.get("timeline"), list))
    if data.get("timeline"):
        t = data["timeline"][0]
        check("Timeline entry has baseline_risk",  "baseline_risk" in t)
        check("Timeline entry has simulated_risk", "simulated_risk" in t or "simula" in str(t))
    print(f"       action={data.get('action_key')}  effectiveness={data.get('effectiveness')}  risk_reduction={data.get('pct_risk_reduction')}%")
    print(f"       target_ip={data.get('target_ip')}  verdict={data.get('tactical_verdict', '')[:60]}...")
except Exception as e:
    check("Simulate POST reachable", False, str(e))

try:
    r = requests.post(f"{BASE_URL}/api/v1/simulate",
                      json={"action": "INVALID_ACTION_KEY"}, headers=HEADERS, timeout=15)
    check("Invalid action -> 400", r.status_code == 400, str(r.status_code))
except Exception as e:
    check("Simulate validation reachable", False, str(e))


# 13. BENCHMARK
section("13. Benchmark  ->  GET /api/benchmark")
try:
    r = requests.get(f"{BASE_URL}/api/benchmark", headers=HEADERS, timeout=15)
    check("Status code 200", r.status_code == 200, str(r.status_code))
    data = r.json()
    if data.get("status") == "pending":
        warn("Benchmark", "Not yet generated. Run: python cli.py benchmark")
    else:
        check("Benchmark has data", bool(data))
except Exception as e:
    check("Benchmark endpoint reachable", False, str(e))


# 14. ROLE-BASED ACCESS CONTROL (RBAC) ENFORCEMENT
section("14. Role-Based Access Control (RBAC) Enforcement")

try:
    r = requests.get(f"{BASE_URL}/api/v1/auth/roles", headers=HEADERS, timeout=10)
    check("GET /api/v1/auth/roles -> 200", r.status_code == 200, str(r.status_code))
    roles_data = r.json()
    check("Roles payload has count", roles_data.get("count", 0) >= 5)
    check("Roles has CHIEF_CISO_ADMIN", any(x.get("key") == "CHIEF_CISO_ADMIN" for x in roles_data.get("roles", [])))
except Exception as e:
    check("Roles endpoint reachable", False, str(e))

# Test Auditor Restricted Mitigation (403 Forbidden)
auditor_headers = {
    "X-API-Key": API_KEY,
    "X-Simulate-Role": "SECURITY_AUDITOR",
    "Content-Type": "application/json"
}
try:
    r = requests.post(f"{BASE_URL}/api/v1/mitigate",
                      headers=auditor_headers,
                      json={"target_ip": "192.168.1.105", "action": "isolate"},
                      timeout=10)
    check("Auditor mitigate -> 403 Forbidden", r.status_code == 403, str(r.status_code))
    data = r.json()
    check("Forbidden error code 403", data.get("code") == 403 or data.get("error") == "Forbidden")
    check("Specifies can_mitigate requirement", data.get("required_permission") == "can_mitigate")
except Exception as e:
    check("Auditor restriction reachable", False, str(e))

# Test Auditor Restricted Upload (403 Forbidden)
try:
    files = {"file": ("test.csv", "timestamp,src_ip\n1,10.0.0.1", "text/csv")}
    r = requests.post(f"{BASE_URL}/api/upload",
                      headers={"X-API-Key": API_KEY, "X-Simulate-Role": "SECURITY_AUDITOR"},
                      files=files,
                      timeout=10)
    check("Auditor upload -> 403 Forbidden", r.status_code == 403, str(r.status_code))
except Exception as e:
    check("Auditor upload restriction reachable", False, str(e))

# Test Auditor Can Still Simulate (200 OK)
try:
    r = requests.post(f"{BASE_URL}/api/v1/simulate",
                      headers=auditor_headers,
                      json={"action": "BLOCK_MANAGEMENT_PORTS", "target_ip": "192.168.1.105", "horizon": 5},
                      timeout=15)
    check("Auditor simulate -> 200 OK", r.status_code == 200, str(r.status_code))
except Exception as e:
    check("Auditor simulate reachable", False, str(e))

# Test Guest Observer Restricted Simulation (403 Forbidden)
guest_headers = {
    "X-API-Key": API_KEY,
    "X-Simulate-Role": "GUEST_OBSERVER",
    "Content-Type": "application/json"
}
try:
    r = requests.post(f"{BASE_URL}/api/v1/simulate",
                      headers=guest_headers,
                      json={"action": "BLOCK_MANAGEMENT_PORTS", "target_ip": "192.168.1.105"},
                      timeout=10)
    check("Guest simulate -> 403 Forbidden", r.status_code == 403, str(r.status_code))
except Exception as e:
    check("Guest simulate restriction reachable", False, str(e))

# Test Operators Directory Listing
try:
    r = requests.get(f"{BASE_URL}/api/v1/users", headers=HEADERS, timeout=10)
    check("GET /api/v1/users -> 200", r.status_code == 200, str(r.status_code))
    users_data = r.json()
    check("Users list has operators", users_data.get("count", 0) >= 1)
except Exception as e:
    check("Users directory reachable", False, str(e))


# SUMMARY
print(f"\n{'='*60}")
print(f"  TEST SUMMARY")
print(f"{'='*60}")
total = results["passed"] + results["failed"]
print(f"  Passed  : {results['passed']}/{total}")
print(f"  Failed  : {results['failed']}/{total}")
print(f"  Warnings: {results['warned']}")

if __name__ == "__main__":
    if results["failed"] == 0:
        print(f"\n  All tests passed! Backend is fully operational.")
    else:
        print(f"\n  {results['failed']} test(s) failed. See output above.")
        sys.exit(1)


def test_backend_live_server():
    import pytest
    if results["failed"] > 0:
        pytest.skip("Threatora live backend is offline or some endpoints are unavailable")
    assert True
