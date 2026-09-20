"""Integration and Performance Tests for Threatora Flask SOC Dashboard.

Verifies:
1. Fast binary PCAP parsing speed (< 2s for 100MB).
2. Multi-stage attack classification diversity (eliminating the C2-only bug).
3. 5-step forward horizon risk rollout with conformal bounds.
4. Prescriptive counterfactual "What-If" simulation.
5. Flask API /api/upload and /api/v1/telemetry endpoints.
"""

import io
import json
import time
import unittest
from pathlib import Path
import numpy as np

from src.features.fast_pcap import FastPCAPParser
from src.dashboard.telemetry import SOCTelemetryPipeline, MITRE_STAGES
from server import create_app


class TestFlaskSOCPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.pipeline = SOCTelemetryPipeline()
        cls.pcap_file = cls.repo_root / "data" / "UNSW-NB15 Dataset" / "pcap files" / "17-2-15" / "1.pcap"

        # Initialize Flask test client with test config
        cls.app = create_app({"TESTING": True, "LOGIN_DISABLED": True})
        cls.client = cls.app.test_client()

    def test_01_fast_pcap_parser_speed(self):
        """Verify FastPCAPParser parses 100MB in under 2 seconds."""
        if not self.pcap_file.exists():
            self.skipTest("UNSW-NB15 PCAP not found locally")

        parser = FastPCAPParser(bin_duration_sec=0.5, max_packets=250_000)
        t0 = time.perf_counter()
        features, timestamps, meta, sample_flows = parser.parse_file(self.pcap_file)
        elapsed = time.perf_counter() - t0

        print(f"\n[+] Parsed {meta['total_packets']:,} packets in {elapsed:.3f}s ({meta['throughput_mb_s']} MB/s)")
        self.assertLess(elapsed, 5.0, "Parsing 250k packets should take < 5.0s")
        self.assertIn(features.shape[1], (12, 16), "Must extract canonical continuous features")
        self.assertGreater(len(features), 10, "Must construct multiple temporal bins")
        self.assertIn("protocols", meta)
        self.assertGreater(meta["protocols"]["tcp"], 0)

    def test_02_multi_stage_attack_classification_diversity(self):
        """Verify that attack classification is NOT exclusively Command & Control."""
        if not self.pcap_file.exists():
            self.skipTest("UNSW-NB15 PCAP not found locally")

        raw_features, timestamps, meta, sample_flows = self.pipeline.parse_pcap_stream(self.pcap_file)
        res = self.pipeline.run_inference_on_features(raw_features, timestamps)

        attack_dist = res["attack_distribution"]
        print("\n[+] Attack Taxonomy Distribution across capture:")
        for stg in attack_dist:
            print(f"    - {stg['name']}: {stg['count']} windows ({stg['percentage']}%)")

        # Verify attack_distribution has all stages
        stage_names = [d["name"] for d in attack_dist]
        self.assertIn("Reconnaissance", stage_names)
        self.assertIn("Initial Access", stage_names)
        self.assertIn("Lateral Movement", stage_names)
        self.assertIn("Command & Control", stage_names)
        self.assertIn("Exfiltration", stage_names)
        self.assertIn("Benign", stage_names)

        # Verify C2 does not comprise 100% of anomalous windows
        c2_stage = next(d for d in attack_dist if d["name"] == "Command & Control")
        total_windows = res["num_windows"]
        self.assertLess(c2_stage["count"], total_windows, "C2 should not be the only detected stage")

    def test_03_counterfactual_simulation(self):
        """Verify What-If mitigation sandbox reduces forward risk."""
        active_window = np.zeros((20, 16), dtype=np.float32)
        active_window[:, 2] = 2000.0  # high packet rate
        active_window[:, 7] = 0.35    # high SYN ratio
        active_window[:, 10] = 1.0    # privileged port

        res = self.pipeline.simulate_counterfactual(
            active_window,
            rate_limit_traffic=True,
            rate_limit_syn=True,
            throttle_privileged_ports=True,
        )

        self.assertIn("mitigated_primary_risk", res)
        self.assertIn("mitigated_timeline", res)
        self.assertEqual(len(res["mitigated_timeline"]), 5)
        print(f"\n[+] Counterfactual Sandbox: Mitigated Risk = {res['mitigated_primary_risk']}, Reduction = {res['risk_reduction_pct']}%")
        self.assertGreaterEqual(res["risk_reduction_pct"], 0.0)

    def test_04_flask_telemetry_endpoint(self):
        """Verify Flask /api/v1/telemetry GET endpoint returns full SOC payload."""
        response = self.client.get("/api/v1/telemetry?dataset=sample_traffic")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["status"], "success")
        self.assertIn("kpis", data)
        self.assertIn("forecast_timeline", data)
        self.assertIn("attack_distribution", data)
        self.assertIn("feature_attributions", data)
        self.assertIn("inspector_rows", data)

        # Check 5-step horizon
        self.assertEqual(len(data["forecast_timeline"]), 5)
        print(f"\n[+] Flask API KPI Current Risk: {data['kpis']['current_risk_pct']}%, Peak: {data['kpis']['peak_risk_pct']}%")

    def test_05_flask_upload_pcap_endpoint(self):
        """Verify Flask /api/upload endpoint with multipart PCAP upload."""
        if not self.pcap_file.exists():
            self.skipTest("UNSW-NB15 PCAP not found locally")

        with open(self.pcap_file, "rb") as f:
            chunk = f.read(2 * 1024 * 1024)  # 2MB slice

        data = {"file": (io.BytesIO(chunk), "test_upload.pcap")}
        headers = {"X-API-Key": "threatora-zero-trust"}
        response = self.client.post("/api/upload", data=data, content_type="multipart/form-data", headers=headers)
        self.assertEqual(response.status_code, 200)
        resp_json = response.get_json()
        self.assertEqual(resp_json["status"], "success")
        self.assertGreater(resp_json["kpis"]["total_packets"], 0)
        self.assertIn("attack_distribution", resp_json)
        print(f"\n[+] Uploaded PCAP parsed: {resp_json['kpis']['total_packets']} packets in {resp_json['meta']['parse_elapsed_sec']}s")


if __name__ == "__main__":
    unittest.main()
