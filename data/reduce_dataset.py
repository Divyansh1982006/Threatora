"""
NetForecast Dataset Reduction & Preprocessing Pipeline
Processes raw CTU-13 binetflow/PCAP files and generates a normalized, clean dataset.
"""

import os
import glob
import pandas as pd
import numpy as np


RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "processed", "NetForecast_clean_dataset.csv")


def process_binetflow_files(sample_ratio: float = 0.1, max_rows_per_file: int = 200_000):
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.binetflow")) + glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    
    if not raw_files:
        print(f"[!] No .binetflow or .csv files found in {RAW_DATA_DIR}.")
        print("[!] Creating synthetic/starter clean dataset for initial runs...")
        generate_synthetic_clean_dataset(OUTPUT_PATH)
        return

    frames = []
    print(f"[*] Found {len(raw_files)} raw file(s). Processing with sample ratio {sample_ratio}...")

    for file_path in raw_files:
        print(f"[*] Reading {file_path}...")
        try:
            df = pd.read_csv(file_path, nrows=max_rows_per_file)
            # Standardize CTU-13 columns
            df.columns = [c.strip() for c in df.columns]

            # Label extraction (Botnet vs Normal / Background)
            if "Label" in df.columns:
                df["is_malicious"] = df["Label"].astype(str).apply(
                    lambda x: 1 if "botnet" in x.lower() or "malicious" in x.lower() else 0
                )
            else:
                df["is_malicious"] = 0

            # Sample subset to prevent memory overload
            if sample_ratio < 1.0:
                df = df.sample(frac=sample_ratio, random_state=42)

            frames.append(df)
        except Exception as e:
            print(f"[!] Error processing {file_path}: {e}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"[+] Clean dataset successfully written to {OUTPUT_PATH} ({len(combined)} rows)")


def generate_synthetic_clean_dataset(output_path: str, n_samples: int = 1000):
    """Generates synthetic baseline dataset conforming to NetForecast feature specs."""
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "dur": np.random.exponential(scale=2.5, size=n_samples).round(4),
        "proto": np.random.choice(["tcp", "udp", "icmp"], size=n_samples, p=[0.7, 0.25, 0.05]),
        "saddr": [f"192.168.1.{np.random.randint(10, 200)}" for _ in range(n_samples)],
        "sport": np.random.randint(1024, 65535, size=n_samples),
        "dir": np.random.choice(["<->", "->", "<-"], size=n_samples),
        "daddr": [f"10.0.0.{np.random.randint(1, 50)}" for _ in range(n_samples)],
        "dport": np.random.choice([80, 443, 22, 53, 8080, 445], size=n_samples),
        "state": np.random.choice(["CON", "FIN", "INT", "URH"], size=n_samples),
        "s_tos": np.zeros(n_samples, dtype=int),
        "d_tos": np.zeros(n_samples, dtype=int),
        "tot_pkts": np.random.randint(1, 500, size=n_samples),
        "tot_bytes": np.random.randint(64, 500000, size=n_samples),
        "src_bytes": np.random.randint(32, 250000, size=n_samples),
        "is_malicious": np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
    }

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[+] Generated starter clean dataset at {output_path} with {n_samples} samples.")


if __name__ == "__main__":
    process_binetflow_files()
