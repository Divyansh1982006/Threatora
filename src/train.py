"""Training Pipeline for NetForecast LSTM World Model.

Optimizes the multi-task objective:
  L = L_recon + w_cls * L_infilt + w_stage * L_stage

Includes:
  - Checkpoint saving & loading compatible with external training laptop
  - Validation tracking & early stopping
  - Scaler export
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import (
    CHECKPOINT_DIR, PROCESSED_DIR, SAMPLES_DIR,
    ModelConfig, default_model_config
)
from .model.world_model import NetworkWorldModel
from .dataset import get_dataloaders
from .features.windows import FeatureScaler
from .prepare_data import prepare_dataset, generate_sample_attack_traffic


def train_world_model(
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 8e-4,
    cfg: Optional[ModelConfig] = None
) -> NetworkWorldModel:
    """Trains the LSTM-based World Model."""
    cfg = cfg or default_model_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training NetForecast LSTM World Model on device: {device}")

    # 1. Load or prepare processed dataset
    proc_csv = PROCESSED_DIR / "processed_cells.csv"
    if not proc_csv.exists():
        print("[!] Processed cells not found. Running data preparation pipeline...")
        prepare_dataset()

    df = pd.read_csv(proc_csv)
    feat_cols = [c for c in df.columns if c not in ("is_malicious", "mitre_stage", "host_ip", "window_idx")]
    X_cells = df[feat_cols].values.astype(np.float32)
    y_infilt = df["is_malicious"].values.astype(np.float32)
    y_stage = df["mitre_stage"].values.astype(np.int64)

    # 2. Dataloaders with temporal splits & guard bands
    scaler = FeatureScaler.load(CHECKPOINT_DIR / "scaler.json")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        X_cells, y_infilt, y_stage, batch_size=batch_size, scaler=scaler
    )

    # 3. Model & Optimizer
    model = NetworkWorldModel(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    criterion_ce = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    ckpt_path = CHECKPOINT_DIR / "world_model.pt"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        recon_loss_sum = 0.0
        cls_loss_sum = 0.0

        for batch_x, batch_y_inf, batch_y_stg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_inf = batch_y_inf.to(device)
            batch_y_stg = batch_y_stg.to(device)

            optimizer.zero_grad()
            out = model(batch_x)

            # Losses
            l_recon = criterion_mse(out["reconstruction"], batch_x)
            l_infilt = criterion_bce(out["infiltration_prob"], batch_y_inf)
            l_stage = criterion_ce(out["stage_logits"].view(-1, cfg.num_stages + 1), batch_y_stg.view(-1))

            total_loss = (
                cfg.recon_weight * l_recon +
                cfg.cls_weight * l_infilt +
                cfg.stage_weight * l_stage
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += total_loss.item()
            recon_loss_sum += l_recon.item()
            cls_loss_sum += l_infilt.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy_inf, vy_stg in val_loader:
                vx, vy_inf, vy_stg = vx.to(device), vy_inf.to(device), vy_stg.to(device)
                vout = model(vx)
                vl_rec = criterion_mse(vout["reconstruction"], vx)
                vl_inf = criterion_bce(vout["infiltration_prob"], vy_inf)
                vl_stg = criterion_ce(vout["stage_logits"].view(-1, cfg.num_stages + 1), vy_stg.view(-1))
                val_loss += (vl_rec + cfg.cls_weight * vl_inf + cfg.stage_weight * vl_stg).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train:.4f} (Recon: {recon_loss_sum/len(train_loader):.3f}) | Val Loss: {avg_val:.4f}")

        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [+] Saved new best model checkpoint to {ckpt_path}")

    # Save run config
    config_path = CHECKPOINT_DIR / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_config": cfg.__dict__,
            "best_val_loss": best_val_loss,
            "trained_epochs": epochs
        }, f, indent=2)

    print(f"[+] Training complete. Model weights saved to {ckpt_path}")
    return model


if __name__ == "__main__":
    import pandas as pd
    train_world_model(epochs=10)
