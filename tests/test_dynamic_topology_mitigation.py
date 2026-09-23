"""Integration Test: Dynamic Network Topology & Mitigation Center
Verifies that after uploading a capture file (PCAP or CSV):
1. Topology nodes and links are extracted from real traffic endpoints.
2. /api/v1/topology returns the uploaded file's nodes, links, subnets, and gateway.
3. /api/v1/assets returns the dynamic asset inventory matching the capture.
4. /api/v1/playbooks returns playbooks for compromised hosts in the capture.
5. /api/v1/mitigate successfully quarantines an endpoint and updates topology and asset status to ISOLATED.
6. /api/simulate computes counterfactual risk reduction.
"""

import io
import json
import unittest
from pathlib import Path

from server import create_app


class TestDynamicTopologyAndMitigation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.pcap_file = cls.repo_root / "data" / "UNSW-NB15 Dataset" / "pcap files" / "17-2-15" / "1.pcap"
        cls.app = create_app({"TESTING": True, "LOGIN_DISABLED": True})
        cls.client = cls.app.test_client()

    def test_01_upload_generates_dynamic_topology_and_mitigation(self):
        """Upload a PCAP and verify dynamic topology, assets, and playbooks."""
        if not self.pcap_file.exists():
            self.skipTest("UNSW-NB15 PCAP not found locally")

        headers = {"X-API-Key": "threatora-zero-trust"}

        # Read first 500KB of the PCAP to simulate a quick upload
        with open(self.pcap_file, "rb") as f:
            pcap_bytes = f.read(512 * 1024)

        data = {
            "file": (io.BytesIO(pcap_bytes), "test_capture.pcap")
        }
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data", headers=headers)
        if res.status_code != 200:
            print("ERROR RESPONSE:", res.status_code, res.data.decode('utf-8', errors='ignore'))
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()
        self.assertEqual(res_json["status"], "success")

        # Verify meta contains topology
        meta = res_json.get("meta", {})
        self.assertIn("topology", meta)
        topo = meta["topology"]
        self.assertIn("nodes", topo)
        self.assertIn("links", topo)
        self.assertGreater(len(topo["nodes"]), 0, "Must extract at least 1 host node")

        uploaded_ips = [n["ip"] for n in topo["nodes"]]
        print(f"\n[+] Uploaded PCAP yielded {len(topo['nodes'])} dynamic nodes: {uploaded_ips[:5]}...")

        # 2. Query /api/v1/topology
        topo_res = self.client.get("/api/v1/topology", headers=headers)
        self.assertEqual(topo_res.status_code, 200)
        topo_data = topo_res.get_json()
        self.assertEqual(topo_data["status"], "success")
        self.assertEqual(len(topo_data["nodes"]), len(topo["nodes"]))
        
        # Verify the nodes are from the uploaded capture, not old hardcoded static IPs
        returned_ips = [n["ip"] for n in topo_data["nodes"]]
        self.assertEqual(returned_ips, uploaded_ips)
        print(f"[+] /api/v1/topology returned active capture endpoints: {returned_ips[:5]}")

        # 3. Query /api/v1/assets
        assets_res = self.client.get("/api/v1/assets", headers=headers)
        self.assertEqual(assets_res.status_code, 200)
        assets_data = assets_res.get_json()
        self.assertEqual(assets_data["status"], "success")
        asset_ips = [a["ip_address"] for a in assets_data["assets"]]
        for ip in uploaded_ips[:3]:
            self.assertIn(ip, asset_ips, f"Asset {ip} should be in /api/v1/assets")
        print(f"[+] /api/v1/assets contains {len(assets_data['assets'])} active assets")

        # 4. Query /api/v1/playbooks
        pb_res = self.client.get("/api/v1/playbooks", headers=headers)
        self.assertEqual(pb_res.status_code, 200)
        pb_data = pb_res.get_json()
        self.assertEqual(pb_data["status"], "success")
        print(f"[+] /api/v1/playbooks returned {len(pb_data['playbooks'])} active playbooks")

        # 5. Test 1-Click Mitigation Isolation
        target_ip = uploaded_ips[0]
        mit_res = self.client.post("/api/v1/mitigate", json={
            "target_ip": target_ip,
            "action": "isolate"
        }, headers=headers)
        self.assertEqual(mit_res.status_code, 200)
        mit_data = mit_res.get_json()
        self.assertIn(mit_data["status"], ["success", "warning"])
        print(f"[+] Quarantined host {target_ip}: {mit_data.get('message')}")

        # 6. Verify /api/v1/topology now marks target_ip as ISOLATED with normal routes
        topo_res_after = self.client.get("/api/v1/topology", headers=headers)
        topo_after_data = topo_res_after.get_json()
        isolated_nodes = [n for n in topo_after_data["nodes"] if n["ip"] == target_ip]
        self.assertTrue(len(isolated_nodes) > 0)
        self.assertEqual(isolated_nodes[0]["status"], "ISOLATED")
        self.assertEqual(isolated_nodes[0].get("risk_score", 0.0), 0.0)
        self.assertFalse(isolated_nodes[0].get("is_compromised", False))

        # Verify connected links are neutralized
        connected_links = [
            l for l in topo_after_data.get("links", [])
            if l.get("source") == target_ip or l.get("target") == target_ip
        ]
        for l in connected_links:
            self.assertFalse(l.get("is_attack_route", False))
            self.assertIn(l.get("threat"), ["normal", "mitigated"])
        print(f"[+] Verified host {target_ip} status is ISOLATED with all connected routes neutralized to normal")

        # 7. Verify /api/v1/assets now marks target_ip as ISOLATED
        assets_after = self.client.get("/api/v1/assets", headers=headers).get_json()
        isolated_assets = [a for a in assets_after["assets"] if a["ip_address"] == target_ip]
        self.assertTrue(len(isolated_assets) > 0)
        self.assertEqual(isolated_assets[0]["quarantine_status"], "ISOLATED")
        print(f"[+] Verified host {target_ip} status in /api/v1/assets is ISOLATED")

        # 8. Test What-If Simulation
        sim_res = self.client.post("/api/simulate", json={
            "isolate_subnet": True,
            "throttle_privileged_ports": True,
        }, headers=headers)
        self.assertEqual(sim_res.status_code, 200)
        sim_data = sim_res.get_json()
        self.assertEqual(sim_data["status"], "success")
        self.assertIn("mitigated_primary_risk", sim_data)
        self.assertIn("risk_reduction_pct", sim_data)
        print(f"[+] What-If Simulation: Risk Reduction = {sim_data['risk_reduction_pct']}%")

    def test_02_csv_upload_generates_dynamic_topology(self):
        """Upload a CSV stream and verify dynamic topology and assets."""
        csv_file = self.repo_root / "data" / "samples" / "sample_traffic.csv"
        if not csv_file.exists():
            self.skipTest("sample_traffic.csv not found locally")

        headers = {"X-API-Key": "threatora-zero-trust"}
        with open(csv_file, "rb") as f:
            csv_bytes = f.read()

        data = {"file": (io.BytesIO(csv_bytes), "sample_traffic.csv")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data", headers=headers)
        self.assertEqual(res.status_code, 200)
        res_json = res.get_json()
        self.assertEqual(res_json["status"], "success")

        # Verify topology in response meta
        meta = res_json.get("meta", {})
        self.assertIn("topology", meta)
        topo = meta["topology"]
        self.assertGreater(len(topo["nodes"]), 0)
        print(f"\n[+] Uploaded CSV yielded {len(topo['nodes'])} dynamic nodes")

        # Verify /api/v1/topology returns these nodes
        topo_res = self.client.get("/api/v1/topology", headers=headers)
        self.assertEqual(topo_res.status_code, 200)
        topo_data = topo_res.get_json()
        self.assertEqual(len(topo_data["nodes"]), len(topo["nodes"]))
        print(f"[+] /api/v1/topology verified for CSV stream")

    def test_03_zero_trust_containment_dynamic_playbook(self):
        """Verify dynamic client-generated playbook execution isolates host and normalizes routes."""
        import time
        headers = {"X-API-Key": "threatora-zero-trust"}
        topo_before = self.client.get("/api/v1/topology", headers=headers).get_json()
        target_ip = topo_before["nodes"][0]["ip"]
        dynamic_pb_uid = f"PB-{int(time.time() * 1000)}"

        # Dispatch 1-Click Zero-Trust Containment
        res = self.client.post("/api/v1/mitigate", json={
            "target_ip": target_ip,
            "playbook_uid": dynamic_pb_uid,
            "action": "isolate"
        }, headers=headers)

        if res.status_code != 200:
            print("ERROR RES:", res.status_code, res.get_data(as_text=True))
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data["status"], ["success", "warning"])
        self.assertEqual(data["host_status"], "ISOLATED")
        self.assertEqual(data["target_ip"], target_ip)
        print(f"\n[+] 1-Click Zero-Trust containment executed for dynamic playbook {dynamic_pb_uid}: {data['message']}")

        # Verify /api/v1/topology has host isolated and routes normal
        topo_res = self.client.get("/api/v1/topology", headers=headers)
        self.assertEqual(topo_res.status_code, 200)
        topo_data = topo_res.get_json()

        host_nodes = [n for n in topo_data.get("nodes", []) if n.get("ip") == target_ip]
        self.assertTrue(len(host_nodes) > 0)
        self.assertEqual(host_nodes[0]["status"], "ISOLATED")
        self.assertEqual(host_nodes[0].get("risk_score", 0.0), 0.0)
        self.assertFalse(host_nodes[0].get("is_compromised", False))

        # Check connected links are neutralized
        connected_links = [
            l for l in topo_data.get("links", [])
            if l.get("source") == target_ip or l.get("target") == target_ip
        ]
        for l in connected_links:
            self.assertFalse(l.get("is_attack_route", False))
            self.assertIn(l.get("threat"), ["normal", "mitigated"])
        print(f"[+] Verified host {target_ip} is ISOLATED and all connected routes are normal")



if __name__ == "__main__":
    unittest.main()
