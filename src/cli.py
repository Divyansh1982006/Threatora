"""Command-Line Interface (CLI) for Threatora Dual-Branch World Model.

Provides full operational capability:
  - console: Launches Metasploit-style interactive tactical cyber defense terminal
  - predict: Runs inference on Flow (CSV), Packet (PCAP), or Dual-Modality inputs
  - benchmark: Evaluates World Model against baselines
  - train: Trains the LSTM World Models
  - import-weights: Imports trained weights and feature scalers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def handle_console(args):
    """Launches the interactive tactical defense console."""
    from .interactive_cli import launch_console
    launch_console(
        api_url=args.api_url,
        username=args.username,
        password=args.password,
        require_auth=not args.no_auth,
    )


def handle_predict(args):
    """Runs dual-branch forward rollout inference on CSV or PCAP."""
    flow_input = getattr(args, "flow", None) or getattr(args, "input", None)
    packet_input = getattr(args, "packet", None) or getattr(args, "pcap", None)

    if not flow_input and not packet_input:
        print("[!] Error: Please provide --flow (CSV) and/or --packet/--pcap (PCAP) input.")
        sys.exit(1)

    from .inference import InferenceEngine

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


def handle_benchmark(args):
    """Runs comparative evaluation suite."""
    from .evaluate import run_benchmark
    print("[*] Launching Threatora Benchmark Suite against Baselines...")
    res = run_benchmark()
    wm = res.get("world_model", {})
    lr = res.get("logistic_regression_stacked_8min", {})
    print("\n" + "=" * 65)
    print(" 📊 THREATORA WORLD MODEL BENCHMARK RESULTS")
    print("=" * 65)
    print(f" World Model F1-Score      : {wm.get('f1_score', 'N/A')}")
    print(f" World Model Precision     : {wm.get('precision', 'N/A')}")
    print(f" World Model Recall        : {wm.get('recall', 'N/A')}")
    print(f" World Model False Pos Rate: {wm.get('false_positive_rate', 'N/A')}")
    print(f" World Model ROC-AUC       : {wm.get('roc_auc', 'N/A')}")
    print("-" * 65)
    print(f" Baseline (LR 8-min) F1    : {lr.get('f1_score', 'N/A')}")
    print("-" * 65)
    print(" [+] Benchmark report saved to artifacts/reports/benchmark.json")


def handle_train(args):
    """Runs World Model training."""
    from .train import train_world_model
    print(f"[*] Starting World Model Training ({args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr})...")
    train_world_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    print("[+] Training completed successfully.")


def handle_import_weights(args):
    """Imports trained weights and feature scalers."""
    import shutil
    from .config import CHECKPOINT_DIR
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.weights:
        dst = CHECKPOINT_DIR / "world_model.pt"
        shutil.copy2(args.weights, dst)
        print(f"[+] Installed model weights to: {dst}")
    if args.scaler:
        dst = CHECKPOINT_DIR / "scaler.json"
        shutil.copy2(args.scaler, dst)
        print(f"[+] Installed feature scaler to: {dst}")
    if args.config:
        dst = CHECKPOINT_DIR / "run_config.json"
        shutil.copy2(args.config, dst)
        print(f"[+] Installed model config to: {dst}")
    print("[+] Weights import completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    default_api_url = os.environ.get("THREATORA_API_URL", "http://127.0.0.1:5000")

    parser = argparse.ArgumentParser(
        prog="threatora",
        description="Threatora: Tactical Cyber Defense & Dual-Branch World Model CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational Subcommands")

    # 1. Console Command (Metasploit-Style Interactive Terminal)
    console_parser = subparsers.add_parser(
        "console",
        help="Launches the Metasploit-style interactive tactical cyber defense terminal",
    )
    console_parser.add_argument(
        "--api-url",
        default=default_api_url,
        help=f"Central server API endpoint [default: {default_api_url}]",
    )
    console_parser.add_argument("-u", "--username", default=None, help="Operator username")
    console_parser.add_argument("-p", "--password", default=None, help="Operator password")
    console_parser.add_argument("--no-auth", action="store_true", help="Disable web auth gate")
    console_parser.set_defaults(func=handle_console)

    # 2. Predict Command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Runs dual-branch inference on CSV flow or PCAP",
    )
    predict_parser.add_argument("input", nargs="?", default=None, help="Input CSV flow or PCAP file path")
    predict_parser.add_argument("--flow", "-f", default=None, help="Path to Flow CSV file")
    predict_parser.add_argument("--packet", "--pcap", "-p", default=None, help="Path to PCAP packet file")
    predict_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    predict_parser.set_defaults(func=handle_predict)

    # 3. Benchmark Command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Evaluates World Model against Logistic Regression and Persistence baselines",
    )
    benchmark_parser.set_defaults(func=handle_benchmark)

    # 4. Train Command
    train_parser = subparsers.add_parser(
        "train",
        help="Trains the LSTM World Models on processed network telemetry",
    )
    train_parser.add_argument("--epochs", type=int, default=15, help="Number of epochs [default: 15]")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size [default: 32]")
    train_parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate [default: 8e-4]")
    train_parser.set_defaults(func=handle_train)

    # 5. Import-Weights Command
    import_parser = subparsers.add_parser(
        "import-weights",
        help="Imports trained weights and feature scalers into checkpoints directory",
    )
    import_parser.add_argument("--weights", required=True, help="Path to trained world_model.pt")
    import_parser.add_argument("--scaler", default=None, help="Path to scaler.json")
    import_parser.add_argument("--config", default=None, help="Path to run_config.json")
    import_parser.set_defaults(func=handle_import_weights)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n[*] Quick Start:")
        print("    Interactive Terminal : python cli.py console")
        print("    Attack Forecast      : python cli.py predict --flow data/samples/sample_traffic.csv")
        print("    Run Benchmarks       : python cli.py benchmark\n")
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

