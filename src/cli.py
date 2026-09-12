"""Command-Line Interface (CLI) for Threatora (NTRO PS 26153).

Provides full offline operational capability:
  - predict: Runs inference on PCAP or CSV files, outputs forecast timeline and MITRE stages
  - benchmark: Evaluates World Model against Logistic Regression baseline
  - train: Trains the LSTM World Model
  - extract: Parses raw PCAP / flows into normalized 60-second window matrices
  - import-weights: Imports weights & scaler trained on external laptop
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
from .evaluate import run_benchmark
from .train import train_world_model
from .prepare_data import prepare_dataset, generate_sample_attack_traffic


def handle_predict(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Error: Input file not found at {input_path}")
        sys.exit(1)

    print(f"[*] Initializing Threatora Inference Engine...")
    engine = InferenceEngine(checkpoint_path=args.checkpoint)

    print(f"[*] Processing telemetry from {input_path.name}...")
    if input_path.suffix.lower() in (".pcap", ".pcapng", ".cap"):
        res = engine.process_pcap(str(input_path))
    else:
        df = pd.read_csv(input_path)
        res = engine.process_traffic_dataframe(df)

    if args.format == "json":
        print(json.dumps(res, indent=2))
        return

    # Formatted terminal output
    print("\n" + "=" * 70)
    print(" ⚡ THREATORA // INFILTRATION PREDICTION & ATT&CK FORECAST")
    print("=" * 70)
    print(f" Total Hosts Analyzed: {res.get('total_hosts', 0)}")
    print(f" Flagged Threat Hosts: {res.get('flagged_hosts', 0)}")
    print("-" * 70)

    for host in res.get("hosts", []):
        stg = host["current_stage"]
        status_marker = "[THREAT DETECTED]" if host["is_anomalous"] else "[BENIGN]"
        print(f"\nHost IP: {host['host_ip']}  {status_marker}")
        print(f"Current Infiltration Probability : {host['current_risk_score'] * 100:.1f}%")
        print(f"Current ATT&CK Stage            : {stg['name']} ({stg['metadata'].get('technique', 'N/A')})")
        print(f"Recommended SOC Action          : {stg['metadata'].get('soc_action', 'N/A')}")

        print("\n  --- 10-Minute Forward Simulation (.imagine rollout) ---")
        print(f"  {'Min':<6} | {'Risk Prob (95% CI)':<22} | {'Predicted Attack Stage':<22}")
        print("  " + "-" * 56)
        for step in host.get("forecast_timeline", []):
            ci_str = f"{step['infilt_prob']*100:.1f}% ({step['lower_ci']*100:.1f}%-{step['upper_ci']*100:.1f}%)"
            print(f"  {step['minute']:<6} | {ci_str:<22} | {step['stage_name']:<22}")

        print("\n  --- Top Driving Features (Explainability Saliency) ---")
        top_feats = host.get("explainability", {}).get("top_features", {})
        for feat, score in list(top_feats.items())[:4]:
            bar = "█" * int(score / 3)
            print(f"  {feat:<24} : {score:5.1f}% {bar}")
        print("-" * 70)


def handle_benchmark(args):
    print("[*] Running comparative benchmark: LSTM World Model vs Logistic Regression...")
    run_benchmark()


def handle_train(args):
    print(f"[*] Launching training for {args.epochs} epochs...")
    train_world_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


def handle_import_weights(args):
    weights_src = Path(args.weights)
    if not weights_src.exists():
        print(f"[!] Error: Weights file not found at {weights_src}")
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target_weights = CHECKPOINT_DIR / "world_model.pt"
    shutil.copy2(weights_src, target_weights)
    print(f"[+] Successfully imported model weights to {target_weights}")

    if args.scaler:
        scaler_src = Path(args.scaler)
        if scaler_src.exists():
            target_scaler = CHECKPOINT_DIR / "scaler.json"
            shutil.copy2(scaler_src, target_scaler)
            print(f"[+] Successfully imported scaler to {target_scaler}")
        else:
            print(f"[!] Warning: Scaler file not found at {scaler_src}, skipping.")

    # run_config.json travels together with world_model.pt — it records the
    # exact ModelConfig (hidden_dim, etc.) used during training.
    # Without it, inference.py has no way to know the checkpoint's true
    # architecture and either crashes on load or (previously) silently fell
    # back to an untrained model. Always look for it next to the weights file,
    # with --run-config as an explicit override for non-standard layouts.
    run_config_src = Path(args.run_config) if args.run_config else (weights_src.parent / "run_config.json")
    if run_config_src.exists():
        target_run_config = CHECKPOINT_DIR / "run_config.json"
        shutil.copy2(run_config_src, target_run_config)
        print(f"[+] Successfully imported run_config.json to {target_run_config}")
    else:
        print(
            f"[!] WARNING: No run_config.json found at {run_config_src}. "
            f"Inference will fall back to config.py defaults for the model architecture — "
            f"if this checkpoint was trained with non-default hyperparameters, "
            f"loading will fail or produce meaningless predictions. "
            f"Pass --run-config explicitly if it lives elsewhere."
        )


def handle_console(args):
    from .interactive_cli import launch_console
    launch_console(
        api_url=args.api_url,
        username=getattr(args, "username", None),
        password=getattr(args, "password", None),
        require_auth=not getattr(args, "no_auth", False),
    )


def main():
    parser = argparse.ArgumentParser(
        prog="threatora",
        description="Threatora: Offline Network Attack Forecasting & MITRE ATT&CK Simulation CLI (NTRO PS 26153)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # console (default tactical shell)
    default_api_url = os.environ.get("THREATORA_API_URL", "http://127.0.0.1:5000")
    p_con = subparsers.add_parser("console", help="Launch interactive Metasploit-style tactical terminal")
    p_con.add_argument("--api-url", default=default_api_url, help=f"Threatora Core API URL (default: {default_api_url})")
    p_con.add_argument("-u", "--username", default=None, help="Web portal operator username (e.g. admin)")
    p_con.add_argument("-p", "--password", default=None, help="Web portal operator passphrase")
    p_con.add_argument("--no-auth", action="store_true", help="Run without web authentication gate")
    p_con.set_defaults(func=handle_console)

    # predict
    p_pred = subparsers.add_parser("predict", help="Run forecast simulation on PCAP or CSV file")
    p_pred.add_argument("-i", "--input", required=True, help="Path to input PCAP or CSV flow file")
    p_pred.add_argument("--horizon", type=int, default=10, help="Forecast horizon steps (default: 10)")
    p_pred.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p_pred.add_argument("--checkpoint", default=None, help="Custom weights checkpoint path")
    p_pred.set_defaults(func=handle_predict)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run comparative benchmark vs Logistic Regression")
    p_bench.set_defaults(func=handle_benchmark)

    # train
    p_train = subparsers.add_parser("train", help="Train the LSTM World Model locally")
    p_train.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_train.add_argument("--lr", type=float, default=8e-4, help="Learning rate")
    p_train.set_defaults(func=handle_train)

    # import-weights
    p_imp = subparsers.add_parser("import-weights", help="Integrate trained weights from external laptop")
    p_imp.add_argument("-w", "--weights", required=True, help="Path to external world_model.pt")
    p_imp.add_argument("-s", "--scaler", default=None, help="Path to external scaler.json")
    p_imp.add_argument("-r", "--run-config", default=None,
                        help="Path to external run_config.json (default: looked up next to --weights)")
    p_imp.set_defaults(func=handle_import_weights)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        # If no arguments provided, launch the interactive console directly
        from .interactive_cli import launch_console
        launch_console(api_url=default_api_url)
    else:
        args.func(args)



if __name__ == "__main__":
    main()
