"""Command-Line Interface (CLI) for Threatora Dual-Branch World Model.

Provides full operational capability:
  - predict: Runs inference on Flow (CSV), Packet (PCAP), or Dual-Modality inputs
  - benchmark: Evaluates World Model against baselines
  - train: Trains the LSTM World Models
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from pathlib import Path
import pandas as pd

from .config import CHECKPOINT_DIR, SAMPLES_DIR
from .inference import InferenceEngine


def handle_predict(args):
    flow_input = getattr(args, "flow", None) or getattr(args, "input", None)
    packet_input = getattr(args, "packet", None) or getattr(args, "pcap", None)

    if not flow_input and not packet_input:
        print("[!] Error: Please provide --flow (CSV) and/or --packet/--pcap (PCAP) input.")
        sys.exit(1)

    print("[*] Initializing Threatora Dual-Branch Inference Engine...")
    engine = InferenceEngine()

    print(f"[*] Executing multi-modal inference pipeline...")
    res = engine.process_network_input(flow_input=flow_input, packet_input=packet_input)

    if args.format == "json":
        print(json.dumps(res, indent=2))
        return

    # Formatted terminal output
    print("\n" + "=" * 70)
    print(" ⚡ THREATORA // DUAL-BRANCH ATTACK FORECASTING & WORLD MODEL")
    print("=" * 70)
    print(f" Modality Mode       : {res.get('modality', 'unknown').upper()}")
    print(f" Final Score P_attack: {res.get('p_attack', 0.0) * 100:.2f}%")
    if "p_flow" in res:
        print(f" Flow Model P_flow   : {res.get('p_flow', 0.0) * 100:.2f}%")
    if "p_packet" in res:
        print(f" Packet Model P_pkt  : {res.get('p_packet', 0.0) * 100:.2f}%")
    if "agreement_score" in res:
        print(f" Cross-Modal Agreement: {res.get('agreement_score', 1.0) * 100:.2f}%")
    
    decision = res.get("decision", {})
    stg = decision.get("stage", {})
    status_marker = "🚨 [THREAT DETECTED]" if decision.get("is_attack") else "✅ [NORMAL TRAFFIC]"
    print(f" Operational Decision: {decision.get('decision')}  {status_marker}")
    print(f" MITRE ATT&CK Stage  : {stg.get('name', 'N/A')} ({decision.get('technique', 'N/A')})")
    print(f" Recommended Action  : {stg.get('soc_action', 'Routine continuous monitoring.')}")
    print("-" * 70)

    timeline = res.get("forecast_timeline", [])
    if timeline:
        print("\n  --- Forward Simulation Rollout (.imagine) ---")
        print(f"  {'Step':<8} | {'Lookahead':<12} | {'Risk Prob (95% CI)'}")
        print("  " + "-" * 56)
        for item in timeline:
            ci_str = f"{item.get('attack_prob', item.get('risk_prob', 0.0))*100:.1f}%"
            if "lower_ci" in item and "upper_ci" in item:
                ci_str += f" ({item['lower_ci']*100:.1f}%-{item['upper_ci']*100:.1f}%)"
            print(f"  {item.get('step', '-'):<8} | {item.get('lookahead', item.get('minute', '-')):<12} | {ci_str}")
        print("-" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="threatora",
        description="Threatora: Dual-Branch Network Attack Forecasting World Model",
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational Subcommands")

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Runs dual-branch inference on CSV flow or PCAP")
    predict_parser.add_argument("input", nargs="?", default=None, help="Input CSV flow or PCAP file path")
    predict_parser.add_argument("--flow", "-f", default=None, help="Path to Flow CSV file")
    predict_parser.add_argument("--packet", "--pcap", "-p", default=None, help="Path to PCAP packet file")
    predict_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    predict_parser.set_defaults(func=handle_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
