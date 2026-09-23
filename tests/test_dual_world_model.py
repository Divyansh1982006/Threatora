"""Automated Verification Suite for Threatora Dual-Branch World Model."""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.flow_world_model import FlowLSTMWorldModel, FLOW_FEATURE_NAMES
from src.model.packet_world_model import PacketLSTMWorldModel, PACKET_FEATURE_NAMES
from src.model.fusion import FusionLayer
from src.model.decision import DecisionLayer
from src.features.flow_preprocessor import FlowPreprocessor
from src.features.packet_preprocessor import PacketPreprocessor
from src.inference import InferenceEngine


def test_flow_world_model():
    print("\n[TEST 1] Testing Flow LSTM World Model...")
    model = FlowLSTMWorldModel(input_dim=12, hidden=128, layers=2, latent=64)
    dummy_in = torch.randn(4, 20, 12)
    out = model(dummy_in)
    
    assert "attack_logits" in out
    assert "attack_prob" in out
    assert "next_state" in out
    assert out["attack_prob"].shape == (4,)
    assert out["next_state"].shape == (4, 12)
    print(f"  [PASS] Forward pass OK: attack_prob shape={out['attack_prob'].shape}, next_state={out['next_state'].shape}")

    # Test Rollout .imagine()
    rollout = model.imagine(dummy_in[:1], horizon=10, n_trajectories=8)
    assert len(rollout["forecast_timeline"]) == 10
    assert len(rollout["feature_forecast"]) == 10
    print(f"  [PASS] Rollout .imagine(horizon=10) OK: timeline length={len(rollout['forecast_timeline'])}")


def test_packet_world_model():
    print("\n[TEST 2] Testing Packet LSTM World Model...")
    model = PacketLSTMWorldModel(input_size=20, hidden_size=128, num_layers=2)
    dummy_in = torch.randn(4, 10, 20)
    out = model(dummy_in)
    
    assert "risk_prob" in out
    assert "future_states" in out
    assert "stage_logits" in out
    assert out["risk_prob"].shape == (4,)
    assert out["future_states"].shape == (4, 5, 20)
    assert out["stage_logits"].shape == (4, 6)
    print(f"  [PASS] Forward pass OK: risk_prob shape={out['risk_prob'].shape}, future_states={out['future_states'].shape}")

    # Test Rollout .imagine()
    rollout = model.imagine(dummy_in[:1], horizon=5)
    assert len(rollout["forecast_timeline"]) == 5
    print(f"  [PASS] Rollout .imagine(horizon=5) OK: predicted_stage={rollout['predicted_stage_name']}")


def test_fusion_and_decision():
    print("\n[TEST 3] Testing Fusion & Decision Layers...")
    fusion = FusionLayer(flow_weight=0.5, conflict_threshold=0.40)
    decision = DecisionLayer(threshold=0.50)

    # 1. Flow only
    f_res = fusion.fuse(p_flow=0.92, p_packet=None)
    assert f_res["p_attack"] == 0.92
    assert f_res["fusion_mode"] == "flow_only"
    d_res = decision.decide(f_res["p_attack"])
    assert d_res["is_attack"] is True
    print(f"  [PASS] Flow only fusion & decision OK: P_attack={f_res['p_attack']} -> {d_res['decision']}")

    # 2. Packet only
    p_res = fusion.fuse(p_flow=None, p_packet=0.15)
    assert p_res["p_attack"] == 0.15
    assert p_res["fusion_mode"] == "packet_only"
    d_res = decision.decide(p_res["p_attack"])
    assert d_res["is_attack"] is False
    print(f"  [PASS] Packet only fusion & decision OK: P_attack={p_res['p_attack']} -> {d_res['decision']}")

    # 3. Dual modality
    dual_res = fusion.fuse(p_flow=0.88, p_packet=0.84)
    assert dual_res["fusion_mode"] == "dual_modality"
    assert dual_res["agreement_score"] > 0.90
    d_res = decision.decide(dual_res["p_attack"])
    print(f"  [PASS] Dual modality fusion OK: P_attack={dual_res['p_attack']}, Agreement={dual_res['agreement_score'] * 100:.1f}%")


def test_end_to_end_inference_engine():
    print("\n[TEST 4] Testing End-to-End InferenceEngine...")
    engine = InferenceEngine()

    # Create synthetic test flow DataFrame
    flow_data = {col: np.random.uniform(0, 100, 30) for col in FLOW_FEATURE_NAMES}
    flow_data["label"] = "DoS-SYN_Flood"
    flow_data["saddr"] = "192.168.1.100"
    flow_df = pd.DataFrame(flow_data)

    flow_res = engine.process_flow_csv(flow_df, horizon=10)
    assert flow_res["status"] == "success"
    assert "p_flow" in flow_res
    assert "p_attack" in flow_res
    assert len(flow_res["forecast_timeline"]) == 10
    print(f"  [PASS] Flow CSV pipeline OK: P_flow={flow_res['p_flow']}, P_attack={flow_res['p_attack']}, Decision={flow_res['decision']['decision']}")

    # Create synthetic packet state DataFrame
    pkt_data = {col: np.random.uniform(0, 50, 15) for col in PACKET_FEATURE_NAMES}
    pkt_df = pd.DataFrame(pkt_data)

    pkt_res = engine.process_pcap(pkt_df, horizon=5)
    assert pkt_res["status"] == "success"
    assert "p_packet" in pkt_res
    print(f"  [PASS] Packet pipeline OK: P_packet={pkt_res['p_packet']}, Stage={pkt_res.get('predicted_stage_name')}")

    # Dual modality
    dual_res = engine.process_network_input(flow_input=flow_df, packet_input=pkt_df, horizon=10)
    assert dual_res["status"] == "success"
    assert dual_res["modality"] == "dual_modality"
    assert "agreement_score" in dual_res
    print(f"  [PASS] Dual Input pipeline OK: P_attack={dual_res['p_attack']}, Agreement={dual_res['agreement_score']*100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("THREATORA DUAL WORLD MODEL VERIFICATION TEST SUITE")
    print("=" * 60)
    test_flow_world_model()
    test_packet_world_model()
    test_fusion_and_decision()
    test_end_to_end_inference_engine()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)")
    print("=" * 60)
