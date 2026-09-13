
"""
REPAIR CICIoT2023 V2 FILE SPLIT

This script fixes the previous bad file-level split.

Expected:
    169 source CSV files
    118 train
     25 validation
     26 test

Important:
- No CTU-13 files are touched.
- No row from a source file is split across train/val/test.
- File assignment is deterministic.
- We optimize split class proportions and target file counts.
- We do NOT use the test split to fit the scaler.
- Output goes to ciciot2023_prepared_v3.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT / "data" / "CIC IoT dataset 2023"
OUTPUT_DIR = PROJECT / "data" / "ciciot2023_prepared_v3"

SEED = 42
WINDOW_SIZE = 20
STRIDE = 5
CHUNKSIZE = 100_000

TRAIN_FILES = 118
VAL_FILES = 25
TEST_FILES = 26

BENIGN_LABEL = "BenignTraffic"

FEATURES = [
    "flow_duration",
    "Duration",
    "Rate",
    "Srate",
    "Drate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "ack_flag_number",
    "Tot size",
    "IAT",
    "Number",
]

REQUIRED_COLUMNS = FEATURES + ["label"]


def discover_files():
    files = sorted(INPUT_DIR.glob("*.csv"))
    if len(files) != 169:
        raise RuntimeError(
            f"Expected 169 CSV files, found {len(files)}"
        )
    return files


def profile_file(path):
    df = pd.read_csv(
        path,
        usecols=["label"],
        dtype={"label": "string"},
        low_memory=False,
    )
    labels = df["label"].str.strip()
    benign = int(labels.eq(BENIGN_LABEL).sum())
    attack = int((~labels.eq(BENIGN_LABEL)).sum())
    rows = len(labels)

    return {
        "file": path.name,
        "path": str(path),
        "rows": rows,
        "benign": benign,
        "attack": attack,
        "benign_rate": benign / rows if rows else 0.0,
        "attack_rate": attack / rows if rows else 0.0,
    }


def get_profiles(files):
    rows = []
    for i, f in enumerate(files, 1):
        p = profile_file(f)
        rows.append(p)
        print(
            f"{i:03d}/169 {f.name} | "
            f"rows={p['rows']:,} | "
            f"benign={p['benign']:,} | attack={p['attack']:,}"
        )
    return pd.DataFrame(rows)


def make_balanced_split(profile):
    """
    Assign exactly 118/25/26 whole files.

    Since almost every file is attack-heavy, optimize:
      1. file counts exactly
      2. total rows near target
      3. benign/attack proportions near global target

    Greedy assignment uses largest files first. Each candidate is scored
    against its target split, while respecting remaining file capacity.
    """
    rng = random.Random(SEED)

    records = profile.to_dict("records")
    rng.shuffle(records)
    records.sort(key=lambda r: r["rows"], reverse=True)

    targets = {
        "train": TRAIN_FILES,
        "validation": VAL_FILES,
        "test": TEST_FILES,
    }

    total_rows = sum(r["rows"] for r in records)
    total_benign = sum(r["benign"] for r in records)
    total_attack = sum(r["attack"] for r in records)

    desired = {
        "train": {
            "rows": total_rows * TRAIN_FILES / 169,
            "benign": total_benign * TRAIN_FILES / 169,
            "attack": total_attack * TRAIN_FILES / 169,
        },
        "validation": {
            "rows": total_rows * VAL_FILES / 169,
            "benign": total_benign * VAL_FILES / 169,
            "attack": total_attack * VAL_FILES / 169,
        },
        "test": {
            "rows": total_rows * TEST_FILES / 169,
            "benign": total_benign * TEST_FILES / 169,
            "attack": total_attack * TEST_FILES / 169,
        },
    }

    assigned = {k: [] for k in targets}
    rows = {k: 0 for k in targets}
    benign = {k: 0 for k in targets}
    attack = {k: 0 for k in targets}

    def capacity_left(split):
        return targets[split] - len(assigned[split])

    def score(split, rec):
        nr = rows[split] + rec["rows"]
        nb = benign[split] + rec["benign"]
        na = attack[split] + rec["attack"]

        d = desired[split]

        row_err = abs(nr - d["rows"]) / max(d["rows"], 1)
        benign_err = abs(nb - d["benign"]) / max(d["benign"], 1)
        attack_err = abs(na - d["attack"]) / max(d["attack"], 1)

        return (
            3.0 * row_err
            + 1.0 * benign_err
            + 1.0 * attack_err
        )

    for rec in records:
        available = [
            s for s in targets
            if capacity_left(s) > 0
        ]

        chosen = min(
            available,
            key=lambda s: (
                score(s, rec),
                len(assigned[s]) / targets[s],
            ),
        )

        assigned[chosen].append(rec["file"])
        rows[chosen] += rec["rows"]
        benign[chosen] += rec["benign"]
        attack[chosen] += rec["attack"]

    if (
        len(assigned["train"]) != 118
        or len(assigned["validation"]) != 25
        or len(assigned["test"]) != 26
    ):
        raise RuntimeError(
            "Split file counts are incorrect: "
            f"{ {k: len(v) for k,v in assigned.items()} }"
        )

    all_names = (
        assigned["train"]
        + assigned["validation"]
        + assigned["test"]
    )

    if len(set(all_names)) != 169:
        raise RuntimeError("Duplicate/missing source files detected.")

    return assigned


def validate_schema(files):
    for f in files:
        cols = list(pd.read_csv(f, nrows=0).columns)
        if "label" not in cols:
            raise RuntimeError(f"{f.name}: label column missing")

        missing = [c for c in FEATURES if c not in cols]
        if missing:
            raise RuntimeError(
                f"{f.name}: missing required features {missing}"
            )


def fit_train_scaler(train_paths):
    """
    Fit clipping + StandardScaler only from TRAIN source files.
    """
    rng = np.random.default_rng(SEED)
    reservoir_max = 300_000
    reservoir = []
    total_seen = 0

    for i, path in enumerate(train_paths, 1):
        print(f"Stats {i:03d}/{len(train_paths)} {path.name}")

        for chunk in pd.read_csv(
            path,
            usecols=FEATURES,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            for col in FEATURES:
                chunk[col] = pd.to_numeric(
                    chunk[col], errors="coerce"
                )

            arr = chunk[FEATURES].to_numpy(
                dtype=np.float64, copy=True
            )
            arr[~np.isfinite(arr)] = np.nan
            total_seen += len(arr)

            if len(arr):
                take = min(
                    len(arr),
                    max(1000, reservoir_max // max(len(train_paths), 1)),
                )
                idx = rng.choice(
                    len(arr),
                    size=take,
                    replace=False,
                )
                reservoir.append(arr[idx])

            current = sum(a.shape[0] for a in reservoir)

            if current > reservoir_max:
                joined = np.concatenate(reservoir, axis=0)
                keep = rng.choice(
                    len(joined),
                    size=reservoir_max,
                    replace=False,
                )
                reservoir = [joined[keep]]

    sample = np.concatenate(reservoir, axis=0)

    med = np.nanmedian(sample, axis=0)
    lo = np.nanquantile(sample, 0.001, axis=0)
    hi = np.nanquantile(sample, 0.999, axis=0)

    medians = {}
    clips = {}

    for i, f in enumerate(FEATURES):
        m = float(med[i]) if np.isfinite(med[i]) else 0.0
        l = float(lo[i]) if np.isfinite(lo[i]) else m
        h = float(hi[i]) if np.isfinite(hi[i]) else m
        if h < l:
            l, h = h, l

        medians[f] = m
        clips[f] = {"low": l, "high": h}

    scaler = StandardScaler()

    for i, path in enumerate(train_paths, 1):
        print(f"Scaler {i:03d}/{len(train_paths)} {path.name}")

        for chunk in pd.read_csv(
            path,
            usecols=FEATURES,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            for col in FEATURES:
                chunk[col] = pd.to_numeric(
                    chunk[col], errors="coerce"
                )

            arr = chunk[FEATURES].to_numpy(
                dtype=np.float64, copy=True
            )
            arr[~np.isfinite(arr)] = np.nan

            for j, f in enumerate(FEATURES):
                arr[:, j] = np.nan_to_num(
                    arr[:, j],
                    nan=medians[f],
                    posinf=clips[f]["high"],
                    neginf=clips[f]["low"],
                )
                arr[:, j] = np.clip(
                    arr[:, j],
                    clips[f]["low"],
                    clips[f]["high"],
                )

            scaler.partial_fit(arr)

    if not hasattr(scaler, "mean_"):
        raise RuntimeError("Scaler was not fitted.")

    return medians, clips, scaler


def transform_and_window(path, medians, clips, scaler):
    df = pd.read_csv(
        path,
        usecols=FEATURES + ["label"],
        low_memory=False,
    )

    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    labels = df["label"].astype("string").str.strip()

    X = df[FEATURES].to_numpy(
        dtype=np.float64, copy=True
    )
    X[~np.isfinite(X)] = np.nan

    for j, f in enumerate(FEATURES):
        X[:, j] = np.nan_to_num(
            X[:, j],
            nan=medians[f],
            posinf=clips[f]["high"],
            neginf=clips[f]["low"],
        )
        X[:, j] = np.clip(
            X[:, j],
            clips[f]["low"],
            clips[f]["high"],
        )

    X = scaler.transform(X).astype(np.float32)

    y = (~labels.eq(BENIGN_LABEL)).astype(np.int8).to_numpy()

    if len(X) < WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, len(FEATURES)), np.float32),
            np.empty((0,), np.int8),
        )

    starts = range(
        0,
        len(X) - WINDOW_SIZE + 1,
        STRIDE,
    )

    W = []
    Y = []

    for s in starts:
        e = s + WINDOW_SIZE
        W.append(X[s:e])

        # FINAL row label.
        # This prevents the previous "any attack in 20 rows" inflation.
        Y.append(y[e - 1])

    return (
        np.stack(W).astype(np.float32),
        np.asarray(Y, dtype=np.int8),
    )


def process_split(
    split_name,
    file_names,
    lookup,
    medians,
    clips,
    scaler,
):
    X_parts = []
    y_parts = []

    for i, name in enumerate(file_names, 1):
        path = lookup[name]

        X, y = transform_and_window(
            path,
            medians,
            clips,
            scaler,
        )

        if len(y) == 0:
            continue

        X_parts.append(X)
        y_parts.append(y)

        print(
            f"{split_name:10s} "
            f"{i:03d}/{len(file_names):03d} "
            f"{name} | "
            f"windows={len(y):,} | "
            f"normal={(y==0).sum():,} | "
            f"attack={(y==1).sum():,}"
        )

    if not X_parts:
        raise RuntimeError(
            f"No windows produced for {split_name}"
        )

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    if not np.isfinite(X_all).all():
        raise RuntimeError(
            f"{split_name}: NaN/Inf detected after scaling."
        )

    return X_all, y_all


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CICIoT2023 PREPARATION V3 — CORRECT FILE SPLIT")
    print("=" * 78)

    files = discover_files()
    lookup = {f.name: f for f in files}

    print("Schema validation...")
    validate_schema(files)
    print("Schema validation: PASS")

    print("\nProfiling files...")
    profile = get_profiles(files)
    profile.to_csv(
        OUTPUT_DIR / "file_label_profile.csv",
        index=False,
    )

    print("\nCreating exact 118/25/26 split...")
    split_map = make_balanced_split(profile)

    split_rows = []
    for split, names in split_map.items():
        sub = profile[profile["file"].isin(names)]

        split_rows.append(
            {
                "split": split,
                "files": len(names),
                "rows": int(sub["rows"].sum()),
                "benign": int(sub["benign"].sum()),
                "attack": int(sub["attack"].sum()),
                "benign_rate": float(
                    sub["benign"].sum() / sub["rows"].sum()
                ),
                "attack_rate": float(
                    sub["attack"].sum() / sub["rows"].sum()
                ),
            }
        )

    summary = pd.DataFrame(split_rows)
    print(summary.to_string(index=False))

    split_df = pd.DataFrame(
        [
            {"split": s, "file": f}
            for s, names in split_map.items()
            for f in names
        ]
    ).sort_values(["split", "file"])

    split_df.to_csv(
        OUTPUT_DIR / "file_split.csv",
        index=False,
    )

    print("\nFitting TRAIN-ONLY scaler...")
    train_paths = [
        lookup[name]
        for name in split_map["train"]
    ]

    medians, clips, scaler = fit_train_scaler(train_paths)

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.pkl",
    )

    print("\nGenerating windows...")
    summaries = {}

    for split in ("train", "validation", "test"):
        X, y = process_split(
            split,
            split_map[split],
            lookup,
            medians,
            clips,
            scaler,
        )

        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "X.npy", X)
        np.save(split_dir / "y.npy", y)

        if split == "train":
            # Useful for capture-balanced sampling later.
            source_ids = np.concatenate(
                [
                    np.full(
                        len(transform_and_window(
                            lookup[name],
                            medians,
                            clips,
                            scaler,
                        )[1]),
                        i,
                        dtype=np.int16,
                    )
                    for i, name in enumerate(split_map[split])
                ]
            )
            np.save(
                split_dir / "source_file_id.npy",
                source_ids,
            )

        summaries[split] = {
            "files": len(split_map[split]),
            "windows": int(len(y)),
            "normal": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
            "attack_rate": float(y.mean()),
            "X_shape": list(X.shape),
        }

        print(
            f"\n{split.upper()} SUMMARY: "
            f"windows={len(y):,} | "
            f"normal={(y==0).sum():,} | "
            f"attack={(y==1).sum():,} | "
            f"attack_rate={y.mean():.4%} | "
            f"shape={X.shape}"
        )

    metadata = {
        "dataset": "CICIoT2023",
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "seed": SEED,
        "features": FEATURES,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "window_label_rule": "final row label",
        "benign_label": BENIGN_LABEL,
        "train_files": split_map["train"],
        "validation_files": split_map["validation"],
        "test_files": split_map["test"],
        "split_summaries": summaries,
        "train_only_scaling": True,
        "cross_file_windows": False,
        "row_random_split": False,
    }

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "feature_statistics.json").write_text(
        json.dumps(
            {
                "medians": medians,
                "clip_stats": clips,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("CICIoT2023 PREPARATION V3 COMPLETE")
    print("=" * 78)

    for split in ("train", "validation", "test"):
        s = summaries[split]
        print(
            f"{split:10s}: "
            f"files={s['files']:3d} | "
            f"windows={s['windows']:,} | "
            f"normal={s['normal']:,} | "
            f"attack={s['attack']:,} | "
            f"attack_rate={s['attack_rate']:.4%} | "
            f"shape={s['X_shape']}"
        )

    print(f"\nOutput: {OUTPUT_DIR}")
    print("CTU-13 modified: NO")
    print("Test scaler fitting: NO")
    print("Cross-file windows: NO")


if __name__ == "__main__":
    main()
