"""
Build a 50:50 CICIoT2023 TRAINING VECTOR with attack-category coverage.

Uses ONLY the 118 files already assigned to the CICIoT2023 TRAIN split in:
    data/ciciot2023_prepared_v3/file_split.csv

Does NOT touch validation/test.

Output:
    data/ciciot2023_balanced_train/
        X.npy                  [N,20,12]
        y.npy                  [N]
        attack_category.npy    [N]
        source_file.npy        [N]
        metadata.json

Strategy:
- Reuses the already-fitted TRAIN-ONLY scaler from prepared_v3.
- Builds the same 20-row / stride-5 temporal windows.
- Window label = final row label.
- Uses all available NORMAL windows up to a configurable cap.
- Samples an equal number of ATTACK windows.
- Attack samples are distributed as evenly as possible across every
  attack category actually present in the selected TRAIN files.
- No validation/test rows are used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT / "data" / "CIC IoT dataset 2023"
PREPARED_DIR = PROJECT / "data" / "ciciot2023_prepared_v3"
SPLIT_FILE = PREPARED_DIR / "file_split.csv"
SCALER_FILE = PREPARED_DIR / "scaler.pkl"
STATS_FILE = PREPARED_DIR / "feature_statistics.json"

OUTPUT_DIR = PROJECT / "data" / "ciciot2023_balanced_train"

WINDOW_SIZE = 20
STRIDE = 5
CHUNKSIZE = 100_000
SEED = 42

BENIGN = "BenignTraffic"

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


def load_train_files():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(SPLIT_FILE)

    split = pd.read_csv(SPLIT_FILE)
    if not {"split", "file"}.issubset(split.columns):
        raise RuntimeError("file_split.csv must contain split,file columns.")

    train = split.loc[split["split"] == "train", "file"].astype(str).tolist()

    if len(train) != 118:
        raise RuntimeError(
            f"Expected 118 train files, found {len(train)}."
        )

    return [INPUT_DIR / f for f in train]


def load_stats_and_scaler():
    scaler = joblib.load(SCALER_FILE)

    meta = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    medians = meta["medians"]
    clips = meta["clip_stats"]

    return scaler, medians, clips


def process_file(path, scaler, medians, clips):
    df = pd.read_csv(
        path,
        usecols=FEATURES + ["label"],
        low_memory=False,
    )

    labels = df["label"].astype("string").str.strip()

    X = df[FEATURES].copy()

    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    arr = X.to_numpy(dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = np.nan

    for j, col in enumerate(FEATURES):
        arr[:, j] = np.nan_to_num(
            arr[:, j],
            nan=medians[col],
            posinf=clips[col]["high"],
            neginf=clips[col]["low"],
        )
        arr[:, j] = np.clip(
            arr[:, j],
            clips[col]["low"],
            clips[col]["high"],
        )

    arr = scaler.transform(arr).astype(np.float32)

    y = labels.ne(BENIGN).to_numpy(dtype=np.int8)

    if len(arr) < WINDOW_SIZE:
        return (
            np.empty((0, WINDOW_SIZE, len(FEATURES)), np.float32),
            np.empty((0,), np.int8),
            np.empty((0,), dtype=object),
        )

    starts = np.arange(
        0,
        len(arr) - WINDOW_SIZE + 1,
        STRIDE,
        dtype=np.int64,
    )

    # Final-row supervision for the temporal window.
    y_win = y[starts + WINDOW_SIZE - 1]
    cat_win = labels.to_numpy(dtype=object)[
        starts + WINDOW_SIZE - 1
    ]

    X_win = np.stack(
        [arr[s:s + WINDOW_SIZE] for s in starts],
        axis=0,
    ).astype(np.float32)

    return X_win, y_win, cat_win


def pass1_count(files, scaler, medians, clips):
    """
    Count available windows per exact attack category and benign.
    """
    counts = {}

    total_windows = 0
    total_normal = 0
    total_attack = 0

    for i, path in enumerate(files, 1):
        X, y, cats = process_file(
            path, scaler, medians, clips
        )

        total_windows += len(y)
        total_normal += int((y == 0).sum())
        total_attack += int((y == 1).sum())

        vc = pd.Series(cats, dtype="string").value_counts()
        for cat, n in vc.items():
            counts[str(cat)] = counts.get(str(cat), 0) + int(n)

        print(
            f"PASS1 {i:03d}/118 {path.name} | "
            f"windows={len(y):,} | "
            f"normal={(y==0).sum():,} | "
            f"attack={(y==1).sum():,}"
        )

    return counts, total_windows, total_normal, total_attack


def build_target_counts(counts, normal_available):
    """
    50/50 target:
      use every normal window available
      sample the same number of attack windows

    Attack target is distributed approximately equally across attack
    categories that actually exist in the selected training files.
    """
    normal_target = int(normal_available)
    attack_target = normal_target

    attack_categories = sorted(
        c for c in counts
        if c != BENIGN and counts[c] > 0
    )

    if not attack_categories:
        raise RuntimeError("No attack categories found.")

    base = attack_target // len(attack_categories)
    remainder = attack_target % len(attack_categories)

    targets = {
        BENIGN: normal_target
    }

    for i, cat in enumerate(attack_categories):
        targets[cat] = base + (1 if i < remainder else 0)

    # If a rare attack class has fewer available windows than its target,
    # reduce to availability and redistribute the deficit iteratively.
    while True:
        deficit = 0
        active = []

        for cat in attack_categories:
            if targets[cat] > counts[cat]:
                deficit += targets[cat] - counts[cat]
                targets[cat] = counts[cat]
            else:
                active.append(cat)

        if deficit == 0:
            break

        if not active:
            break

        q, r = divmod(deficit, len(active))

        for i, cat in enumerate(active):
            targets[cat] += q + (1 if i < r else 0)

    actual_attack_target = sum(
        targets[c] for c in attack_categories
    )

    if actual_attack_target != normal_target:
        raise RuntimeError(
            "Could not construct exact 50:50 attack target. "
            f"normal={normal_target}, attack={actual_attack_target}"
        )

    return targets


def pass2_collect(files, scaler, medians, clips, targets):
    """
    Collect exactly the target number of windows per class.

    To avoid holding the complete 5.5M-window corpus in memory:
    process each file and take only the required windows.
    """
    rng = np.random.default_rng(SEED)

    X_parts = []
    y_parts = []
    cat_parts = []
    file_parts = []

    remaining = dict(targets)
    file_names = [p.name for p in files]

    # We shuffle source-file processing order so selection isn't biased
    # toward the lexicographically first files.
    order = np.arange(len(files))
    rng.shuffle(order)

    for processed, file_index in enumerate(order, 1):
        path = files[file_index]

        X, y, cats = process_file(
            path, scaler, medians, clips
        )

        if len(y) == 0:
            continue

        cats_str = np.asarray(cats, dtype=object)

        selected_global = []

        for cat, target_left in list(remaining.items()):
            if target_left <= 0:
                continue

            if cat == BENIGN:
                mask = (y == 0)
            else:
                mask = cats_str == cat

            idx = np.flatnonzero(mask)

            if len(idx) == 0:
                continue

            take = min(int(target_left), len(idx))

            if take < len(idx):
                chosen = rng.choice(
                    idx,
                    size=take,
                    replace=False,
                )
            else:
                chosen = idx

            selected_global.extend(
                (int(i), cat) for i in chosen
            )
            remaining[cat] -= take

        if selected_global:
            sel = np.array(
                [x[0] for x in selected_global],
                dtype=np.int64,
            )

            X_parts.append(X[sel])
            y_parts.append(y[sel])
            cat_parts.append(cats_str[sel])
            file_parts.append(
                np.asarray(
                    [file_names[file_index]] * len(sel),
                    dtype=object,
                )
            )

        print(
            f"PASS2 {processed:03d}/{len(files)} {path.name} | "
            f"selected={len(selected_global):,} | "
            f"remaining={sum(remaining.values()):,}"
        )

        if sum(remaining.values()) == 0:
            break

    if sum(remaining.values()) != 0:
        raise RuntimeError(
            f"Could not collect all target windows. Remaining: {remaining}"
        )

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    cat_all = np.concatenate(cat_parts, axis=0)
    file_all = np.concatenate(file_parts, axis=0)

    # Shuffle final balanced training vector.
    perm = rng.permutation(len(y_all))

    X_all = X_all[perm]
    y_all = y_all[perm]
    cat_all = cat_all[perm]
    file_all = file_all[perm]

    return X_all, y_all, cat_all, file_all


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CICIoT2023 — 50:50 BALANCED LSTM TRAINING VECTOR")
    print("=" * 78)

    files = load_train_files()

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing training files: {missing[:5]}"
        )

    print(f"Training files: {len(files)}")
    print(f"Window shape : ({WINDOW_SIZE}, {len(FEATURES)})")
    print("Normal/attack target: 50/50")
    print("Attack-category balancing: ENABLED")
    print("Validation/test used: NO")
    print()

    scaler, medians, clips = load_stats_and_scaler()

    print("PASS 1 — counting temporal windows by class/category")
    counts, total_windows, total_normal, total_attack = pass1_count(
        files,
        scaler,
        medians,
        clips,
    )

    print()
    print(
        f"Available windows: {total_windows:,} | "
        f"normal={total_normal:,} | attack={total_attack:,}"
    )

    attack_categories = sorted(
        c for c in counts
        if c != BENIGN and counts[c] > 0
    )

    print(f"Attack categories present: {len(attack_categories)}")
    for cat in attack_categories:
        print(f"  {cat:35s} {counts[cat]:,}")

    targets = build_target_counts(counts, total_normal)

    print()
    print("TARGET WINDOWS")
    print(f"  {BENIGN:35s} {targets[BENIGN]:,}")

    for cat in attack_categories:
        print(f"  {cat:35s} {targets[cat]:,}")

    print()
    print("PASS 2 — collecting balanced windows")

    X, y, categories, source_file = pass2_collect(
        files,
        scaler,
        medians,
        clips,
        targets,
    )

    if X.ndim != 3 or X.shape[1:] != (WINDOW_SIZE, len(FEATURES)):
        raise RuntimeError(f"Bad X shape: {X.shape}")

    if len(X) != len(y) or len(y) != len(categories):
        raise RuntimeError("Output arrays have inconsistent lengths.")

    if not np.isfinite(X).all():
        raise RuntimeError("X contains NaN/Inf.")

    normal = int((y == 0).sum())
    attack = int((y == 1).sum())

    if normal != attack:
        raise RuntimeError(
            f"Not 50:50: normal={normal}, attack={attack}"
        )

    np.save(OUTPUT_DIR / "X.npy", X.astype(np.float32))
    np.save(OUTPUT_DIR / "y.npy", y.astype(np.int8))
    np.save(
        OUTPUT_DIR / "attack_category.npy",
        categories.astype(object),
    )
    np.save(
        OUTPUT_DIR / "source_file.npy",
        source_file.astype(object),
    )

    category_counts = (
        pd.Series(categories, dtype="string")
        .value_counts()
        .rename_axis("label")
        .reset_index(name="windows")
    )
    category_counts.to_csv(
        OUTPUT_DIR / "category_counts.csv",
        index=False,
    )

    metadata = {
        "dataset": "CICIoT2023",
        "source_training_files": 118,
        "output": str(OUTPUT_DIR),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "feature_names": FEATURES,
        "normal_label": BENIGN,
        "window_label_rule": "final row label",
        "normal_windows": normal,
        "attack_windows": attack,
        "attack_ratio": attack / len(y),
        "balance": "50:50",
        "attack_category_balancing": True,
        "attack_categories_in_output": attack_categories,
        "targets_by_category": targets,
        "available_windows_by_category": counts,
        "scaler_source": str(SCALER_FILE),
        "scaler_fitted_on_train_only": True,
        "validation_used": False,
        "test_used": False,
        "ctu13_used": False,
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("BALANCED TRAINING VECTOR COMPLETE")
    print("=" * 78)
    print(f"X shape       : {X.shape}")
    print(f"Normal        : {normal:,}")
    print(f"Attack        : {attack:,}")
    print(f"Attack ratio  : {attack / len(y):.4%}")
    print(f"Categories    : {len(attack_categories)}")
    print(f"Output        : {OUTPUT_DIR}")
    print()
    print("50:50 CHECK    : PASS")
    print("TRAIN ONLY     : PASS")
    print("VAL USED       : NO")
    print("TEST USED      : NO")
    print("CTU-13 MODIFIED: NO")


if __name__ == "__main__":
    main()
