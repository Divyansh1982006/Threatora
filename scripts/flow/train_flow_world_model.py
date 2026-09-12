
import json
import math
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "ciciot2023_balanced_train"
VAL_DIR = ROOT / "data" / "ciciot2023_prepared_v3" / "validation"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 1024
EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.30
HIDDEN = 128
LAYERS = 2
LATENT = 64
VAL_LIMIT = 50000
PRINT_EVERY = 20
USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


class ArrayDataset(Dataset):
    def __init__(self, x_path, y_path, indices=None):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.X.ndim != 3:
            raise ValueError(f"Expected X [N,T,F], got {self.X.shape}")
        if len(self.X) != len(self.y):
            raise ValueError(f"X/y mismatch: {len(self.X)} vs {len(self.y)}")
        self.indices = indices

    def __len__(self):
        return len(self.X) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        i = idx if self.indices is None else int(self.indices[idx])
        x = np.asarray(self.X[i], dtype=np.float32).copy()
        y = int(self.y[i])
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


def make_validation_indices(y_path, limit, seed=42):
    y = np.load(y_path, mmap_mode="r")
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64) if n <= limit else rng.choice(
        n, size=limit, replace=False
    ).astype(np.int64)

    labels = np.asarray(y[idx], dtype=np.int8)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise RuntimeError("Validation subset contains only one class.")

    per_class = min(len(pos), len(neg), limit // 2)
    pos_sel = rng.choice(pos, size=per_class, replace=False)
    neg_sel = rng.choice(neg, size=per_class, replace=False)
    final = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(final)
    return idx[final]


def binary_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int8)
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
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
        "accuracy": acc, "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def rank_auc(y_true, scores):
    y = np.asarray(y_true).astype(np.int8)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
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
    rpos = ranks[y == 1].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def evaluate(model, loader, criterion):
    model.eval()
    losses, yt, pp = [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(USE_AMP and DEVICE.type == "cuda")
            ):
                logits, _ = model(x)
                loss = criterion(logits, y)
            losses.append(float(loss.item()))
            yt.append(y.cpu().numpy())
            pp.append(torch.sigmoid(logits).cpu().numpy())

    yt = np.concatenate(yt)
    pp = np.concatenate(pp)
    m = binary_metrics(yt, pp)
    m["roc_auc"] = rank_auc(yt, pp)
    m["loss"] = float(np.mean(losses))
    return m


def main():
    print("=" * 70)
    print("FAST CICIoT2023 LSTM WORLD MODEL")
    print("=" * 70)
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_x, train_y = TRAIN_DIR / "X.npy", TRAIN_DIR / "y.npy"
    val_x, val_y = VAL_DIR / "X.npy", VAL_DIR / "y.npy"

    for p in (train_x, train_y, val_x, val_y):
        if not p.exists():
            raise FileNotFoundError(p)

    train_y_mm = np.load(train_y, mmap_mode="r")
    val_y_mm = np.load(val_y, mmap_mode="r")
    print(
        f"Train: {len(train_y_mm):,} | "
        f"normal={(train_y_mm == 0).sum():,} | attack={(train_y_mm == 1).sum():,}",
        flush=True
    )
    print(
        f"Full val: {len(val_y_mm):,} | "
        f"normal={(val_y_mm == 0).sum():,} | attack={(val_y_mm == 1).sum():,}",
        flush=True
    )

    val_indices = make_validation_indices(val_y, VAL_LIMIT, SEED)
    train_ds = ArrayDataset(train_x, train_y)
    val_ds = ArrayDataset(val_x, val_y, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    sx, _ = train_ds[0]
    if tuple(sx.shape) != (20, 12):
        raise ValueError(f"Expected window [20,12], got {tuple(sx.shape)}")

    model = LSTMWorldModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(USE_AMP and DEVICE.type == "cuda")
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Train batches/epoch: {math.ceil(len(train_ds)/BATCH_SIZE):,}", flush=True)
    print(f"Validation samples: {len(val_ds):,}", flush=True)
    print("-" * 70, flush=True)

    best_f1 = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=(USE_AMP and DEVICE.type == "cuda")
            ):
                logits, _ = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())

            if batch_idx == 1 or batch_idx % PRINT_EVERY == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | batch {batch_idx}/{len(train_loader)} | "
                    f"loss={loss.item():.5f} | elapsed={time.time()-start:.1f}s",
                    flush=True
                )

        train_loss = total_loss / len(train_loader)
        vm = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch} DONE | train_loss={train_loss:.5f} | "
            f"val_loss={vm['loss']:.5f} | ROC-AUC={vm['roc_auc']:.4f} | "
            f"F1={vm['f1']:.4f} | Precision={vm['precision']:.4f} | "
            f"Recall={vm['recall']:.4f} | FPR={vm['fpr']:.4f} | "
            f"time={time.time()-start:.1f}s",
            flush=True
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **vm})

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": vm,
                "config": {
                    "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
                    "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
                    "hidden": HIDDEN, "layers": LAYERS, "latent": LATENT,
                    "validation_subset": len(val_ds)
                }
            }, OUT_DIR / "ciciot_lstm_world_model_fast_best.pt")
            print(f"*** NEW BEST: epoch={epoch}, val_F1={best_f1:.4f}", flush=True)

    with (OUT_DIR / "ciciot_lstm_world_model_fast_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 70)
    print("TRAINING COMPLETE", flush=True)
    print(f"Checkpoint: {OUT_DIR / 'ciciot_lstm_world_model_fast_best.pt'}", flush=True)
    print(f"History: {OUT_DIR / 'ciciot_lstm_world_model_fast_history.json'}", flush=True)
    print("NOTE: next-state head is retained architecturally but is NOT trained as a genuine chronological S(t)->S(t+1) target in this fast classification run.", flush=True)


if __name__ == "__main__":
    main()
