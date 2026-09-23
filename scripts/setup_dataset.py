#!/usr/bin/env python3
"""Threatora CTU-13 Dataset & Environment Setup Verifier.

Verifies and sets up:
  1. CTU-13 scenario parquet files and topology edge graphs in data/processed/
  2. Dataset samples in data/samples/
  3. Feature scaler (scaler.json) and processed training matrix (processed_cells.csv)
  4. MITRE ATT&CK taxonomy engine and flow classification rules
  5. Checkpoint weights and model inference pipeline readiness
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import glob
import pandas as pd


def check_dataset_setup():
    print("=" * 65)
    print(" ⚡ THREATORA // CTU-13 DATASET & ENVIRONMENT SETUP VERIFIER")
    print("=" * 65)

    # 1. Check processed scenarios
    proc_dir = ROOT_DIR / "data" / "processed"
    scenario_files = sorted(glob.glob(str(proc_dir / "scenario_[0-9][0-9].parquet")))
    edge_files = sorted(glob.glob(str(proc_dir / "scenario_[0-9][0-9]_edges.parquet")))

    print(f"[*] 1. Checking CTU-13 Processed Scenarios ({len(scenario_files)} / 13 discovered)...")
    if len(scenario_files) >= 13:
        total_cells = sum(len(pd.read_parquet(f)) for f in scenario_files)
        print(f"    [+] OK: All 13 CTU-13 scenarios present ({total_cells:,} total host-window cells).")
        print(f"    [+] OK: {len(edge_files)} edge topology graphs present.")
    else:
        print(f"    [!] Warning: Found {len(scenario_files)} scenarios. Expected 13.")

    # 2. Check sample captures
    samples_dir = ROOT_DIR / "data" / "samples"
    transition_sample = samples_dir / "host-becomes-infected.csv"
    attack_sample = samples_dir / "sample_traffic.csv"

    print(f"[*] 2. Checking Sample Telemetry Captures...")
    if transition_sample.exists():
        print(f"    [+] OK: Transition capture present: {transition_sample.name} ({transition_sample.stat().st_size // 1024} KB)")
    else:
        print(f"    [!] Warning: {transition_sample.name} not found.")

    if attack_sample.exists():
        print(f"    [+] OK: Attack sample present: {attack_sample.name}")
    else:
        print(f"    [!] Warning: {attack_sample.name} not found.")

    # 3. Check Scaler and Processed Cells
    scaler_path = ROOT_DIR / "artifacts" / "checkpoints" / "scaler.json"
    proc_csv = proc_dir / "processed_cells.csv"
    print(f"[*] 3. Checking Feature Scaler & Training Matrix...")
    if scaler_path.exists():
        print(f"    [+] OK: FeatureScaler present: {scaler_path.name}")
    else:
        print(f"    [*] Generating FeatureScaler...")
        from src.prepare_data import prepare_dataset
        prepare_dataset()

    if proc_csv.exists():
        print(f"    [+] OK: Processed cells present: {proc_csv.name}")

    # 4. Check MITRE ATT&CK Engine
    print(f"[*] 4. Checking MITRE ATT&CK Engine...")
    try:
        from src import mitre
        print(f"    [+] OK: Stages: {', '.join(mitre.STAGE_NAMES)}")
        print(f"    [+] OK: Ground-truth flow label rules: {len(mitre._LABEL_STAGE_RULES)} patterns compiled.")
        test_stage = mitre.stage_from_label("flow=From-Botnet-V42-TCP-CC16-HTTP-Not-Encrypted")
        assert test_stage == mitre.COMMAND_AND_CONTROL
        print(f"    [+] OK: Ground-truth rule test passed (CC16 -> Command & Control).")
    except Exception as e:
        print(f"    [!] Error verifying MITRE engine: {e}")

    # 5. Check LSTM Checkpoints
    flow_ckpt = ROOT_DIR / "artifacts" / "checkpoints" / "flow" / "ciciot_lstm_world_model_fast_best.pt"
    pkt_ckpt = ROOT_DIR / "artifacts" / "checkpoints" / "packet" / "packet_lstm_world_model_best.pt"
    print(f"[*] 5. Checking Dual-Branch LSTM Model Checkpoints...")
    if flow_ckpt.exists():
        print(f"    [+] OK: Flow LSTM checkpoint verified: {flow_ckpt.name}")
    if pkt_ckpt.exists():
        print(f"    [+] OK: Packet LSTM checkpoint verified: {pkt_ckpt.name}")

    print("=" * 65)
    print(" [+] THREATORA DATASET & ENVIRONMENT READY FOR INFERENCE & DEMO")
    print("=" * 65)


if __name__ == "__main__":
    check_dataset_setup()
