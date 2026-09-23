
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ciciot_lstm_world_model_fast_best.pt"
TEST_DIR = ROOT / "data" / "ciciot2023_prepared_v3" / "test"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1024
USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArrayDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.X.ndim != 3:
            raise ValueError(f"Expected X [N,T,F], got {self.X.shape}")
        if len(self.X) != len(self.y):
            raise ValueError(f"X/y mismatch: {len(self.X)} vs {len(self.y)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32).copy()
        y = int(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


class LSTMWorldModel(nn.Module):
    def __init__(self, input_dim=12, hidden=128, layers=2, latent=64, dropout=0.30):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(hidden, latent),
            nn.LayerNorm(latent),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attack_head = nn.Sequential(
            nn.Linear(latent, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.next_state_head = nn.Sequential(
            nn.Linear(latent, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        x = self.input_norm(x)
        seq, _ = self.encoder(x)
        z = self.to_latent(seq[:, -1, :])
        return self.attack_head(z).squeeze(-1), self.next_state_head(z)


def binary_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int8)
    probs = np.asarray(probs)
    pred = (probs >= threshold).astype(np.int8)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def rank_auc(y_true, scores):
    y = np.asarray(y_true).astype(np.int8)
    s = np.asarray(scores, dtype=np.float64)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)

    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    sum_pos_ranks = ranks[y == 1].sum()
    return float(
        (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def average_precision(y_true, scores):
    y = np.asarray(y_true).astype(np.int8)
    s = np.asarray(scores, dtype=np.float64)

    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]

    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, len(y_sorted) + 1)

    return float((precision[y_sorted == 1]).sum() / n_pos)


def main():
    print("=" * 72)
    print("CICIoT2023 TEST — FAST LSTM WORLD MODEL")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x_path = TEST_DIR / "X.npy"
    y_path = TEST_DIR / "y.npy"

    for p in (MODEL_PATH, x_path, y_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    test_y = np.load(y_path, mmap_mode="r")
    print(
        f"Test: {len(test_y):,} | "
        f"normal={(test_y == 0).sum():,} | "
        f"attack={(test_y == 1).sum():,} | "
        f"attack_rate={(test_y == 1).mean():.4%}",
        flush=True,
    )

    ds = ArrayDataset(x_path, y_path)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = LSTMWorldModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}",
        flush=True,
    )
    print("-" * 72, flush=True)

    y_true_chunks = []
    prob_chunks = []
    total = len(loader)

    start = time.time()
    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(loader, 1):
            x = x.to(DEVICE, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            ):
                logits, _ = model(x)

            probs = torch.sigmoid(logits)

            y_true_chunks.append(y.numpy())
            prob_chunks.append(probs.cpu().numpy())

            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total:
                print(
                    f"Test batch {batch_idx}/{total} | "
                    f"samples={(batch_idx * BATCH_SIZE):,} "
                    f"| elapsed={time.time()-start:.1f}s",
                    flush=True,
                )

    y_true = np.concatenate(y_true_chunks)
    probs = np.concatenate(prob_chunks)

    metrics = binary_metrics(y_true, probs, threshold=0.5)
    metrics["roc_auc"] = rank_auc(y_true, probs)
    metrics["pr_auc"] = average_precision(y_true, probs)
    metrics["threshold"] = 0.5
    metrics["samples"] = len(y_true)
    metrics["attack_rate"] = float(y_true.mean())

    output = OUT_DIR / "ciciot_lstm_world_model_test_results.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("-" * 72)
    print("TEST COMPLETE")
    print(f"ROC-AUC   : {metrics['roc_auc']:.6f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")
    print(f"F1        : {metrics['f1']:.6f}")
    print(f"FPR       : {metrics['fpr']:.6f}")
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(
        f"Confusion : TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
