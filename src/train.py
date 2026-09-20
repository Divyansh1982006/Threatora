"""Training Pipeline for Threatora Temporal Transformer World Model.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Optimizes the multi-task objective:
  L = 0.5 * L_state + 1.0 * L_attack + 0.5 * L_mitre
Includes:
  - RAM-safe and fast Parquet Dataset ingestion (16 canonical slots)
  - Multi-Horizon future target supervision across all 5 lookahead steps (Y_t in R^5)
  - Neural MITRE ATT&CK 6-class classification head training via CrossEntropyLoss
  - FP16 AMP mixed precision training optimized for NVIDIA GPUs with CPU fallback
  - Model checkpointing to models/threatora_transformer.pt
  - ONNX export with dynamic batch axis to models/threatora_transformer.onnx
  - Execution summary and metrics exported to data/processed/train_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .model import ThreatoraTemporalTransformerWorldModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainWorldModel")


class ParquetWindowDataset(Dataset):
    """PyTorch Dataset for paired temporal state transition windows loaded from Parquet."""

    def __init__(self, parquet_path: Union[str, Path]):
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        t0 = time.time()
        logger.info(f"Loading parquet dataset from: {parquet_path}...")
        df = pl.read_parquet(parquet_path)

        # Ingest list columns into contiguous numpy arrays
        s_t_raw = np.array(df["s_t"].to_list(), dtype=np.float32)
        s_next_raw = np.array(df["s_next"].to_list(), dtype=np.float32)
        labels_raw = np.array(df["label"].to_list(), dtype=np.float32)

        # Determine feature dimension: (N, 20 * num_features) -> (N, 20, num_features)
        num_feats = s_t_raw.shape[1] // 20
        self.s_t = torch.from_numpy(s_t_raw).view(-1, 20, num_feats)
        self.s_next = torch.from_numpy(s_next_raw).view(-1, 20, num_feats)
        self.label = torch.from_numpy(labels_raw)

        # Multi-horizon ground truth targets: Y_t in R^5
        if "y_horizon" in df.columns:
            y_h_raw = np.array(df["y_horizon"].to_list(), dtype=np.float32)
            self.y_horizon = torch.from_numpy(y_h_raw).view(-1, 5)
        else:
            self.y_horizon = self.label.unsqueeze(-1).repeat(1, 5)

        # MITRE ATT&CK stage targets (0..5)
        if "mitre_stage" in df.columns:
            m_raw = np.array(df["mitre_stage"].to_list(), dtype=np.int64)
            self.mitre_stage = torch.from_numpy(m_raw)
        else:
            self.mitre_stage = (self.label.long() * 2)

        self.num_samples = len(self.label)

        logger.info(
            f"Loaded {self.num_samples} samples in {time.time() - t0:.2f}s "
            f"(s_t: {tuple(self.s_t.shape)}, attacks: {int(torch.sum(self.label))})"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.s_t[idx], self.s_next[idx], self.label[idx], self.y_horizon[idx], self.mitre_stage[idx]


def get_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Creates PyTorch DataLoaders for training and validation splits."""
    train_dataset = ParquetWindowDataset(train_path)
    val_dataset = ParquetWindowDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def export_onnx(
    model: ThreatoraTemporalTransformerWorldModel,
    output_onnx_path: Path,
    device: torch.device,
    num_features: int = 16,
) -> None:
    """Exports the trained model to ONNX with dynamic batch axis for CPU deployment."""
    logger.info(f"Exporting model to ONNX: {output_onnx_path}...")
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, 20, num_features, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        input_names=["input_s_t"],
        output_names=["pred_s_next", "primary_attack_logit", "latent_embedding", "mitre_logits", "attn_weights"],
        dynamic_axes={
            "input_s_t": {0: "batch_size"},
            "pred_s_next": {0: "batch_size"},
            "primary_attack_logit": {0: "batch_size"},
            "latent_embedding": {0: "batch_size"},
            "mitre_logits": {0: "batch_size"},
            "attn_weights": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model verified and exported successfully ({file_size_mb:.2f} MB)")


def train(
    train_parquet: Optional[Union[str, Path]] = None,
    val_parquet: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    num_workers: int = 0,
) -> Dict[str, object]:
    """Runs the end-to-end training, validation, checkpointing, and ONNX export pipeline."""
    repo_root = Path(__file__).resolve().parent.parent
    train_path = Path(train_parquet) if train_parquet else repo_root / "data" / "processed" / "train_windows.parquet"
    val_path = Path(val_parquet) if val_parquet else repo_root / "data" / "processed" / "val_windows.parquet"
    save_dir = Path(models_dir) if models_dir else repo_root / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_out_path = repo_root / "data" / "processed" / "train_metrics.json"

    # Hardware detection & acceleration configuration
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {device_name} ({vram_total_gb:.2f} GB VRAM) with cuDNN benchmark & AMP FP16")
    else:
        logger.warning("CUDA not available, running on CPU.")

    # DataLoaders (num_workers=0 on Windows avoids spawn overhead)
    train_loader, val_loader = get_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=is_cuda,
    )

    num_features = train_loader.dataset.s_t.shape[-1]
    logger.info(f"Detected {num_features} canonical continuous feature slots in dataset.")

    # Model instantiation
    model = ThreatoraTemporalTransformerWorldModel(
        seq_len=20,
        num_features=num_features,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        dropout=0.1,
        horizon_k=5,
        num_mitre_classes=6,
    ).to(device)

    # Multi-task loss functions
    criterion_state = nn.SmoothL1Loss()
    criterion_attack = nn.BCEWithLogitsLoss()
    criterion_mitre = nn.CrossEntropyLoss()

    # Optimizer and Cosine Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # AMP Mixed Precision Scaler
    scaler = GradScaler("cuda", enabled=is_cuda)

    best_val_loss_state = float("inf")
    best_weights_path = save_dir / "threatora_transformer.pt"
    onnx_path = save_dir / "threatora_transformer.onnx"

    epoch_metrics: List[Dict[str, float]] = []
    start_time_all = time.time()
    peak_vram_mb = 0.0

    logger.info("=" * 75)
    logger.info(f"Starting Training: {epochs} Epochs | Batch Size: {batch_size} | LR: {lr}")
    logger.info("Multi-task Loss: 0.5 * SmoothL1(S_{t+1}) + 1.0 * BCE(5-Horizon) + 0.5 * CE(MITRE)")
    logger.info("=" * 75)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ------------------ TRAINING PHASE ------------------
        model.train()
        train_loss_total = 0.0
        train_loss_state = 0.0
        train_loss_attack = 0.0
        train_loss_mitre = 0.0

        train_preds: List[np.ndarray] = []
        train_targets: List[np.ndarray] = []

        for batch_s_t, batch_s_next, batch_label, batch_y_horizon, batch_mitre in train_loader:
            batch_s_t = batch_s_t.to(device, non_blocking=True)
            batch_s_next = batch_s_next.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            batch_y_horizon = batch_y_horizon.to(device, non_blocking=True)
            batch_mitre = batch_mitre.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=is_cuda):
                pred_s_next, primary_attack_logit, latent_embedding, mitre_logits, _ = model(batch_s_t)
                l_state = criterion_state(pred_s_next, batch_s_next)
                future_risk_logits = model.forecasting_head(latent_embedding)
                l_attack = criterion_attack(future_risk_logits, batch_y_horizon)
                l_mitre = criterion_mitre(mitre_logits, batch_mitre)
                total_loss = 0.5 * l_state + 1.0 * l_attack + 0.5 * l_mitre

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = len(batch_label)
            train_loss_total += total_loss.item() * bs
            train_loss_state += l_state.item() * bs
            train_loss_attack += l_attack.item() * bs
            train_loss_mitre += l_mitre.item() * bs

            # Store predictions for train ROC-AUC
            with torch.no_grad():
                probs = torch.sigmoid(primary_attack_logit.squeeze(-1)).detach().cpu().numpy()
                train_preds.append(probs)
                train_targets.append(batch_label.detach().cpu().numpy())

        scheduler.step()

        n_train = len(train_loader.dataset)
        avg_train_loss = train_loss_total / n_train
        avg_train_loss_state = train_loss_state / n_train
        avg_train_loss_attack = train_loss_attack / n_train
        avg_train_loss_mitre = train_loss_mitre / n_train

        # Compute Train ROC-AUC
        train_preds_arr = np.concatenate(train_preds)
        train_targets_arr = np.concatenate(train_targets)
        try:
            train_roc_auc = float(roc_auc_score(train_targets_arr, train_preds_arr))
        except Exception:
            train_roc_auc = 0.0

        # ------------------ VALIDATION PHASE ------------------
        model.eval()
        val_loss_total = 0.0
        val_loss_state = 0.0
        val_loss_attack = 0.0
        val_loss_mitre = 0.0

        with torch.no_grad():
            for batch_s_t, batch_s_next, batch_label, batch_y_horizon, batch_mitre in val_loader:
                batch_s_t = batch_s_t.to(device, non_blocking=True)
                batch_s_next = batch_s_next.to(device, non_blocking=True)
                batch_label = batch_label.to(device, non_blocking=True)
                batch_y_horizon = batch_y_horizon.to(device, non_blocking=True)
                batch_mitre = batch_mitre.to(device, non_blocking=True)

                with autocast("cuda", enabled=is_cuda):
                    pred_s_next, primary_attack_logit, latent_embedding, mitre_logits, _ = model(batch_s_t)
                    l_state = criterion_state(pred_s_next, batch_s_next)
                    future_risk_logits = model.forecasting_head(latent_embedding)
                    l_attack = criterion_attack(future_risk_logits, batch_y_horizon)
                    l_mitre = criterion_mitre(mitre_logits, batch_mitre)
                    total_loss = 0.5 * l_state + 1.0 * l_attack + 0.5 * l_mitre

                bs = len(batch_label)
                val_loss_total += total_loss.item() * bs
                val_loss_state += l_state.item() * bs
                val_loss_attack += l_attack.item() * bs
                val_loss_mitre += l_mitre.item() * bs

        n_val = len(val_loader.dataset)
        avg_val_loss = val_loss_total / n_val
        avg_val_loss_state = val_loss_state / n_val
        avg_val_loss_attack = val_loss_attack / n_val
        avg_val_loss_mitre = val_loss_mitre / n_val

        # Peak VRAM monitoring
        if is_cuda:
            current_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            peak_vram_mb = max(peak_vram_mb, current_vram_mb)

        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        # Checkpoint best model strictly based on state reconstruction loss (val_loss_state)
        is_best = avg_val_loss_state < best_val_loss_state
        if is_best:
            best_val_loss_state = avg_val_loss_state
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss_state": best_val_loss_state,
                    "train_roc_auc": train_roc_auc,
                    "num_features": num_features,
                },
                best_weights_path,
            )

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} (State: {avg_train_loss_state:.4f}, Atk: {avg_train_loss_attack:.4f}, MITRE: {avg_train_loss_mitre:.4f}) | "
            f"Train ROC-AUC: {train_roc_auc:.4f} | "
            f"Val State Loss: {avg_val_loss_state:.4f} | "
            f"LR: {current_lr:.2e}"
            f"{' [BEST SAVED]' if is_best else ''}"
        )

        epoch_metrics.append({
            "epoch": epoch,
            "train_loss_total": round(avg_train_loss, 5),
            "train_loss_state": round(avg_train_loss_state, 5),
            "train_loss_attack": round(avg_train_loss_attack, 5),
            "train_loss_mitre": round(avg_train_loss_mitre, 5),
            "train_roc_auc": round(train_roc_auc, 5),
            "val_loss_total": round(avg_val_loss, 5),
            "val_loss_state": round(avg_val_loss_state, 5),
            "lr": round(current_lr, 7),
            "duration_sec": round(epoch_duration, 2),
        })

    total_training_time = time.time() - start_time_all
    logger.info("=" * 75)
    logger.info(f"Training Complete in {total_training_time:.2f}s! Best Val State Loss: {best_val_loss_state:.5f}")
    logger.info("=" * 75)

    # Load best checkpoint before ONNX export
    logger.info(f"Loading best weights from {best_weights_path} for ONNX export...")
    best_checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Export to ONNX (deploy on CPU with dynamic batch size)
    export_onnx(model=model, output_onnx_path=onnx_path, device=device, num_features=num_features)

    # Save training metrics JSON
    train_summary = {
        "task": "Threatora Temporal Transformer World Model Training",
        "dataset_train": str(train_path),
        "dataset_val": str(val_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_features": num_features,
        "best_val_state_loss": round(best_val_loss_state, 5),
        "final_train_roc_auc": round(epoch_metrics[-1]["train_roc_auc"], 5),
        "total_training_time_sec": round(total_training_time, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "checkpoint_path": str(best_weights_path),
        "onnx_path": str(onnx_path),
        "epoch_metrics": epoch_metrics,
    }

    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2)

    logger.info(f"Saved training metrics to: {metrics_out_path}")
    return train_summary


def main():
    parser = argparse.ArgumentParser(
        description="Train Threatora Temporal Transformer World Model (UNSW-NB15)"
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default=None,
        help="Path to train_windows.parquet (default: data/processed/train_windows.parquet)",
    )
    parser.add_argument(
        "--val-parquet",
        type=str,
        default=None,
        help="Path to val_windows.parquet (default: data/processed/val_windows.parquet)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Output directory for model weights (default: models)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count (default: 0 for Windows)",
    )

    args = parser.parse_args()

    try:
        train(
            train_parquet=args.train_parquet,
            val_parquet=args.val_parquet,
            models_dir=args.models_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
        )
    except Exception as exc:
        logger.exception(f"Training pipeline execution failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
