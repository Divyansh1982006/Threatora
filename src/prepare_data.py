"""Data Ingestion and Preprocessing Pipeline for NetForecast.

Ingests CTU-13 or CIC-IDS-2018 flows & PCAPs from data/raw/ and generates:
  - data/processed/processed_cells.parquet (or .csv)
  - artifacts/checkpoints/scaler.json
  - data/samples/sample_traffic.csv (synthetic realistic scenario if raw data absent)
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, CHECKPOINT_DIR, ALL_FEATURE_COLS
from .features.windows import build_host_windows_from_flows, FeatureScaler


def generate_sample_attack_traffic(output_csv: Path) -> pd.DataFrame:
    """Generates realistic 20-minute multi-stage attack traffic matching the Infographic scenario:
    12:00 Normal
    12:01 Phishing Click (Port 80/443 web traffic spike)
    12:02 Malware Beacon (Regular heartbeat connections)
    12:03-12:04 C2 Communication (Periodic IRC/HTTP beacons)
    12:05-12:06 Data Exfiltration (Massive outbound byte burst)
    12:07+ Covering Tracks (Interrupted connections, log cleaning)
    """
    np.random.seed(42)
    records = []
    base_epoch = 1628596800.0  # 12:00:00

    # 1. Normal traffic (Minutes 0..3): 100 flows
    for i in range(100):
        t = base_epoch + np.random.uniform(0, 180)
        records.append({
            "StartTime": t,
            "saddr": "192.168.1.105",
            "sport": np.random.randint(49152, 65535),
            "dir": "->",
            "daddr": f"10.0.0.{np.random.randint(1, 10)}",
            "dport": np.random.choice([80, 443, 53]),
            "proto": "tcp" if np.random.rand() > 0.3 else "udp",
            "state": "CON",
            "dur": np.random.exponential(0.5) + 0.01,
            "tot_pkts": np.random.randint(4, 15),
            "tot_bytes": np.random.randint(400, 3000),
            "src_bytes": np.random.randint(200, 1500),
            "flags": "SA",
            "Label": "Normal",
            "is_malicious": 0
        })

    # 2. Reconnaissance (Minutes 3..6): Port Scan (Port 22, 80, 443, 445, 8080)
    for i in range(120):
        t = base_epoch + 180 + np.random.uniform(0, 180)
        records.append({
            "StartTime": t,
            "saddr": "192.168.1.105",
            "sport": np.random.randint(49152, 65535),
            "dir": "->",
            "daddr": f"10.0.0.{np.random.randint(1, 50)}",
            "dport": np.random.randint(1, 1024),
            "proto": "tcp",
            "state": "INT",
            "dur": 0.001,
            "tot_pkts": 2,
            "tot_bytes": 120,
            "src_bytes": 120,
            "flags": "S",
            "Label": "From-Botnet-V42-TCP-Attempt",
            "is_malicious": 1
        })

    # 3. Initial Access / Exploit (Minutes 6..8): Malicious Binary Download
    for i in range(30):
        t = base_epoch + 360 + np.random.uniform(0, 120)
        records.append({
            "StartTime": t,
            "saddr": "192.168.1.105",
            "sport": np.random.randint(49152, 65535),
            "dir": "->",
            "daddr": "198.51.100.45",
            "dport": 80,
            "proto": "tcp",
            "state": "CON",
            "dur": 2.5,
            "tot_pkts": 40,
            "tot_bytes": 85000,
            "src_bytes": 1500,
            "flags": "SPA",
            "Label": "From-Botnet-V49-TCP-HTTP-Binary-Download",
            "is_malicious": 1
        })

    # 4. Command & Control Beacons (Minutes 8..12): Periodic 10-second beaconing
    for i in range(24):
        t = base_epoch + 480 + (i * 10.0) + np.random.normal(0, 0.1)
        records.append({
            "StartTime": t,
            "saddr": "192.168.1.105",
            "sport": 54321,
            "dir": "<->",
            "daddr": "203.0.113.88",
            "dport": 6667,
            "proto": "tcp",
            "state": "CON",
            "dur": 0.05,
            "tot_pkts": 6,
            "tot_bytes": 520,
            "src_bytes": 260,
            "flags": "PA",
            "Label": "From-Botnet-V45-TCP-CC106-IRC-Not-Encrypted",
            "is_malicious": 1
        })

    # 5. Data Exfiltration (Minutes 12..16): Volumetric burst outbound
    for i in range(80):
        t = base_epoch + 720 + np.random.uniform(0, 240)
        records.append({
            "StartTime": t,
            "saddr": "192.168.1.105",
            "sport": np.random.randint(49152, 65535),
            "dir": "->",
            "daddr": "203.0.113.88",
            "dport": 443,
            "proto": "tcp",
            "state": "CON",
            "dur": 4.2,
            "tot_pkts": 250,
            "tot_bytes": 350000,
            "src_bytes": 340000,
            "flags": "PA",
            "Label": "From-Botnet-V42-TCP-Attempt-SPAM",
            "is_malicious": 1
        })

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[+] Created realistic attack scenario with {len(df)} flows at {output_csv}")
    return df


def prepare_dataset():
    """Builds preprocessed 60-second window cells from raw datasets."""
    raw_files = glob.glob(str(RAW_DIR / "*.binetflow")) + glob.glob(str(RAW_DIR / "*.csv"))

    if not raw_files:
        print("[!] No raw captures found in data/raw/. Generating realistic attack scenario dataset...")
        sample_file = SAMPLES_DIR / "sample_traffic.csv"
        df = generate_sample_attack_traffic(sample_file)
    else:
        print(f"[*] Found {len(raw_files)} raw files in data/raw/. Ingesting...")
        dfs = []
        for f in raw_files[:5]:  # Process up to 5 raw captures
            try:
                sub_df = pd.read_csv(f, nrows=100_000)
                dfs.append(sub_df)
            except Exception as e:
                print(f"[!] Error reading {f}: {e}")
        df = pd.concat(dfs, ignore_index=True) if dfs else generate_sample_attack_traffic(SAMPLES_DIR / "sample_traffic.csv")

    X_cells, y_infilt, y_stage, meta = build_host_windows_from_flows(df)
    print(f"[+] Extracted {len(X_cells)} 60-second host-window cells.")

    # Save scaler
    scaler = FeatureScaler()
    scaler.fit(X_cells)
    scaler.save(CHECKPOINT_DIR / "scaler.json")
    print(f"[+] Saved fitted FeatureScaler to {CHECKPOINT_DIR / 'scaler.json'}")

    # Save processed cells
    proc_df = pd.DataFrame(X_cells, columns=ALL_FEATURE_COLS)
    proc_df["is_malicious"] = y_infilt
    proc_df["mitre_stage"] = y_stage
    proc_df["host_ip"] = [m[0] for m in meta]
    proc_df["window_idx"] = [m[1] for m in meta]

    proc_path = PROCESSED_DIR / "processed_cells.csv"
    proc_df.to_csv(proc_path, index=False)
    print(f"[+] Processed cells written to {proc_path}")


if __name__ == "__main__":
    prepare_dataset()
