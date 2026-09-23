"""Cross-verification tests using synthetic benign and attack datasets.

Verifies:
1. Deep Telemetry Inspector table logic contradiction fix:
   - When a row is "Benign (Normal)", primary_risk is strictly < 25.0% (never 100%).
   - Only active confirmed attack stages display elevated risk (>= 65.0%).
2. Full cross-page synchronization:
   - Uploading synthetic benign data synchronizes topology (all healthy, zero compromised) and mitigation (0 playbooks).
   - Uploading synthetic attack data synchronizes topology (compromised target, attack routes) and mitigation (active playbooks).
3. 1-Click Zero-Trust Containment:
   - Quarantining the target isolates the host in topology and asset inventory.
   - Neutralizes connected attack routes to normal ambient flow.
"""

from __future__ import annotations

import io
import unittest
from pathlib import Path
from server import create_app
from src.prepare_data import generate_synthetic_benign_traffic, generate_synthetic_attack_traffic


class TestSyntheticCrossVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.samples_dir = cls.repo_root / "data" / "samples"
        cls.benign_csv = cls.samples_dir / "synthetic_benign.csv"
        cls.attack_csv = cls.samples_dir / "synthetic_attack.csv"

        if not cls.benign_csv.exists():
            generate_synthetic_benign_traffic(cls.benign_csv)
        if not cls.attack_csv.exists():
            generate_synthetic_attack_traffic(cls.attack_csv)

        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_01_synthetic_benign_verification(self):
        """Verify synthetic benign traffic ingestion, inspector row calibration, topology, and mitigation."""
        headers = {"X-API-Key": "threatora-zero-trust"}

        with open(self.benign_csv, "rb") as f:
            csv_bytes = f.read()

        data = {"file": (io.BytesIO(csv_bytes), "synthetic_benign.csv")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data", headers=headers)
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()
        self.assertEqual(res_json["status"], "success")

        # 1. Verify inspector_rows risk contradiction fix for Benign
        inspector_rows = res_json.get("inspector_rows", [])
        self.assertGreater(len(inspector_rows), 0)

        for row in inspector_rows:
            risk = row["primary_risk"]
            stage = row["stage_name"]
            # All benign rows must have calibrated risk strictly < 25.0%
            if "benign" in stage.lower() or row.get("stage_id", 0) == 0:
                self.assertLess(
                    risk, 25.0,
                    f"Row {row['window_idx']} classified as '{stage}' but has uncalibrated elevated risk {risk}%!"
                )
                self.assertNotEqual(risk, 100.0, f"Row {row['window_idx']} displays 100.0% risk contradiction!")
        print(f"\n[+] Verified all {len(inspector_rows)} inspector rows in synthetic benign capture show calibrated risk < 25%")

        # 2. Verify global KPI and attack state
        self.assertFalse(res_json["is_attack"])
        self.assertEqual(res_json["kpis"]["stage_id"], 0)
        self.assertIn("Benign", res_json["kpis"]["stage_name"])
        self.assertLess(res_json["kpis"]["peak_risk_pct"], 25.0)
        print(f"[+] Verified synthetic benign traffic evaluated with peak risk = {res_json['kpis']['peak_risk_pct']}% (< 25%)")

        # 3. Verify Topology endpoint for benign state
        topo_res = self.client.get("/api/v1/topology", headers=headers)
        self.assertEqual(topo_res.status_code, 200)
        topo_data = topo_res.get_json()
        for node in topo_data.get("nodes", []):
            self.assertFalse(node.get("is_compromised", False))
            self.assertIn(node.get("status"), ["HEALTHY", "ISOLATED"])
            self.assertEqual(node.get("risk_score", 0.0), 0.0)
        for link in topo_data.get("links", []):
            self.assertFalse(link.get("is_attack_route", False))
            self.assertIn(link.get("threat"), ["normal", "mitigated"])
        print("[+] Verified /api/v1/topology in benign state: 0 compromised nodes, all routes normal")

        # 4. Verify Mitigation Center endpoint for benign state
        pb_res = self.client.get("/api/v1/playbooks", headers=headers)
        self.assertEqual(pb_res.status_code, 200)
        pb_data = pb_res.get_json()
        self.assertEqual(len(pb_data.get("playbooks", [])), 0)
        print("[+] Verified /api/v1/playbooks in benign state: 0 active containment playbooks")

    def test_02_synthetic_attack_verification_and_mitigation(self):
        """Verify synthetic attack traffic ingestion, inspector row calibration, topology, and 1-click mitigation containment."""
        headers = {"X-API-Key": "threatora-zero-trust"}
        target_ip = "172.16.203.79"

        with open(self.attack_csv, "rb") as f:
            csv_bytes = f.read()

        data = {"file": (io.BytesIO(csv_bytes), "synthetic_attack.csv")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data", headers=headers)
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()
        self.assertEqual(res_json["status"], "success")

        # 1. Verify inspector_rows: NO Benign rows show 100% risk; attack rows show elevated risk
        inspector_rows = res_json.get("inspector_rows", [])
        self.assertGreater(len(inspector_rows), 0)

        benign_count = 0
        attack_count = 0
        for row in inspector_rows:
            risk = row["primary_risk"]
            stage = row["stage_name"]
            if "benign" in stage.lower() or row.get("stage_id", 0) == 0:
                benign_count += 1
                self.assertLess(
                    risk, 25.0,
                    f"Row {row['window_idx']} classified as '{stage}' but has uncalibrated elevated risk {risk}%!"
                )
            else:
                attack_count += 1
                self.assertGreaterEqual(
                    risk, 65.0,
                    f"Attack row {row['window_idx']} ('{stage}') has low risk {risk}% (< 65%)!"
                )
        print(f"\n[+] Verified inspector rows: {benign_count} benign rows (< 25%), {attack_count} attack rows (>= 65%)")

        # 2. Verify attack detection & flagged host
        self.assertTrue(res_json["is_attack"])
        self.assertGreaterEqual(res_json["kpis"]["peak_risk_pct"], 65.0)
        self.assertIn(target_ip, res_json.get("flagged_hosts", []))
        print(f"[+] Verified attack detected on flagged target {target_ip} with peak risk {res_json['kpis']['peak_risk_pct']}%")

        # 3. Verify Topology renders target as COMPROMISED with attack routes
        topo_res = self.client.get("/api/v1/topology", headers=headers)
        self.assertEqual(topo_res.status_code, 200)
        topo_data = topo_res.get_json()

        target_nodes = [n for n in topo_data.get("nodes", []) if n.get("ip") == target_ip]
        self.assertTrue(len(target_nodes) > 0)
        self.assertEqual(target_nodes[0]["status"], "COMPROMISED")
        self.assertTrue(target_nodes[0].get("is_compromised", False))
        print(f"[+] Verified target {target_ip} status in /api/v1/topology is COMPROMISED")

        # 4. Verify Mitigation Center generated active playbooks
        pb_res = self.client.get("/api/v1/playbooks", headers=headers)
        self.assertEqual(pb_res.status_code, 200)
        pb_data = pb_res.get_json()
        target_pbs = [p for p in pb_data.get("playbooks", []) if p.get("target_ip") == target_ip]
        self.assertTrue(len(target_pbs) > 0)
        pb_uid = target_pbs[0]["playbook_uid"]
        print(f"[+] Verified active playbook {pb_uid} generated for target {target_ip}")

        # 5. Execute 1-Click Zero-Trust Containment
        mit_res = self.client.post("/api/v1/mitigate", json={
            "target_ip": target_ip,
            "playbook_uid": pb_uid,
            "action": "isolate"
        }, headers=headers)
        self.assertEqual(mit_res.status_code, 200)
        mit_data = mit_res.get_json()
        self.assertIn(mit_data["status"], ["success", "warning"])
        self.assertEqual(mit_data.get("host_status"), "ISOLATED")
        print(f"[+] Executed 1-Click Containment on {target_ip}: {mit_data.get('message')}")

        # 6. Verify Topology: Target is ISOLATED, risk is 0.0%, connected routes are neutralized to normal
        topo_after = self.client.get("/api/v1/topology", headers=headers).get_json()
        iso_nodes = [n for n in topo_after.get("nodes", []) if n.get("ip") == target_ip]
        self.assertTrue(len(iso_nodes) > 0)
        self.assertEqual(iso_nodes[0]["status"], "ISOLATED")
        self.assertEqual(iso_nodes[0].get("risk_score", 0.0), 0.0)
        self.assertFalse(iso_nodes[0].get("is_compromised", False))

        connected_links = [
            l for l in topo_after.get("links", [])
            if l.get("source") == target_ip or l.get("target") == target_ip
        ]
        for l in connected_links:
            self.assertFalse(l.get("is_attack_route", False))
            self.assertIn(l.get("threat"), ["normal", "mitigated"])
        print(f"[+] Verified host {target_ip} is now ISOLATED with 0.0% risk and all connected routes neutralized to normal")

        # 7. Verify Asset ledger marks host as ISOLATED
        assets_res = self.client.get("/api/v1/assets", headers=headers).get_json()
        iso_assets = [a for a in assets_res.get("assets", []) if a.get("ip_address") == target_ip]
        self.assertTrue(len(iso_assets) > 0)
        self.assertEqual(iso_assets[0].get("quarantine_status"), "ISOLATED")
        print(f"[+] Verified host {target_ip} status in /api/v1/assets is ISOLATED")


if __name__ == "__main__":
    unittest.main()
