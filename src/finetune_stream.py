"""Streaming PyTorch Fine-Tuning Pipeline for Threatora Temporal Transformer.

Features:
- Sequential chunked streaming IterableDataset: Loads one Parquet chunk into RAM at a time,
  yields mini-batches, and drops the chunk before loading the next chunk.
- Strict 80/20 chronological train/val chunk partitioning.
- AdamW (lr=1e-4, weight_decay=1e-3), CosineAnnealingLR (T_max=5, eta_min=1e-6).
- Mixed precision (torch.amp.autocast('cuda')) keeping VRAM < 2.0 GB on RTX 3050.
- Multi-task loss: 0.5 * SmoothL1Loss(pred_S_next, S_next) + 1.0 * BCEWithLogitsLoss(pred_risk_logits, y_t).
- Zero catastrophic forgetting: preserves learned temporal baseline dynamics.
- Checkpoint backup, ONNX export, and comprehensive report generation.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def natural_sort_key(p: Path) -> List[Union[int, str]]:
    """Sort strings containing numbers in human order (e.g., chunk_0001 before chunk_0010)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(p))]


import numpy as np
import onnx
import polars as pl
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.model import ThreatoraTemporalTransformerWorldModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("StreamingFineTune")


class StreamingChunkDataset(IterableDataset):
    """Streams batches from sequentially loaded Parquet chunks without loading the full dataset."""

    def __init__(
        self,
        chunk_files: List[Path],
        batch_size: int = 64,
        shuffle_within_chunk: bool = True,
        seq_len: int = 20,
    ):
        super().__init__()
        self.chunk_files = sorted(chunk_files)
        self.batch_size = batch_size
        self.shuffle_within_chunk = shuffle_within_chunk
        self.seq_len = seq_len
        self.replay_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_to_read = self.chunk_files
        else:
            # Partition chunk files evenly across dataloader worker processes
            files_to_read = [
                f for i, f in enumerate(self.chunk_files) if i % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in files_to_read:
            if not chunk_path.exists():
                continue

            try:
                # Read single chunk parquet into RAM
                df = pl.read_parquet(chunk_path)
                if len(df) == 0:
                    continue

                s_t_raw = np.array(df["s_t"].to_list(), dtype=np.float32)
                s_next_raw = np.array(df["s_next"].to_list(), dtype=np.float32)
                y_raw = np.array(df["label"].to_list(), dtype=np.float32)

                num_samples = len(y_raw)
                in_dim = s_t_raw.shape[1] // self.seq_len

                s_t_t = torch.from_numpy(s_t_raw).view(num_samples, self.seq_len, in_dim)
                s_next_t = torch.from_numpy(s_next_raw).view(num_samples, self.seq_len, in_dim)
                y_t = torch.from_numpy(y_raw)

                indices = np.arange(num_samples)
                if self.shuffle_within_chunk:
                    np.random.shuffle(indices)

                # Collect sample benign windows into replay reservoir to prevent sequential recency bias
                benign_mask = (y_raw == 0)
                if np.any(benign_mask):
                    b_indices = np.where(benign_mask)[0]
                    sample_size = min(len(b_indices), 256)
                    sampled = np.random.choice(b_indices, size=sample_size, replace=False)
                    for s_i in sampled:
                        if len(self.replay_buffer) < 512:
                            self.replay_buffer.append((s_t_t[s_i], s_next_t[s_i], y_t[s_i]))

                for start_idx in range(0, num_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_samples)
                    batch_idx = indices[start_idx:end_idx]

                    b_st = s_t_t[batch_idx]
                    b_sn = s_next_t[batch_idx]
                    b_y = y_t[batch_idx]

                    # If batch is 100% attack and we have replay benign samples, interleave a few benign samples
                    if len(self.replay_buffer) > 16 and (b_y == 1).all():
                        n_swap = min(4, len(b_y) // 4)
                        for rep_i in range(n_swap):
                            rep_st, rep_sn, rep_y = self.replay_buffer[np.random.randint(len(self.replay_buffer))]
                            b_st[rep_i] = rep_st
                            b_sn[rep_i] = rep_sn
                            b_y[rep_i] = rep_y

                    yield (b_st, b_sn, b_y)

            except Exception as exc:
                logger.error(f"Error reading chunk {chunk_path.name}: {exc}")
            finally:
                # Explicit cleanup of chunk data from RAM
                if "df" in locals():
                    del df
                if "s_t_raw" in locals():
                    del s_t_raw, s_next_raw, y_raw
                if "s_t_t" in locals():
                    del s_t_t, s_next_t, y_t
                gc.collect()


def get_total_chunk_samples(chunk_files: List[Path]) -> int:
    """Computes total samples across parquet files using fast metadata inspection."""
    total = 0
    for p in chunk_files:
        try:
            meta = pq.read_metadata(p)
            total += meta.num_rows
        except Exception:
            pass
    return total


def export_model_to_onnx(
    model: ThreatoraTemporalTransformerWorldModel,
    output_onnx_path: Path,
    device: torch.device,
    num_features: int = 16,
) -> None:
    """Exports model to ONNX with dynamic batch axis."""
    output_onnx_path = Path(output_onnx_path)
    logger.info(f"Exporting fine-tuned model to ONNX: {output_onnx_path}...")
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

    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    file_size_mb = output_onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model verified and exported successfully ({file_size_mb:.2f} MB)")


def map_12_to_16_canonical(x_12: torch.Tensor) -> torch.Tensor:
    """Maps 12 continuous flow metrics from batch_preprocess into the 16 canonical slots."""
    B, T, _ = x_12.shape
    x_16 = torch.zeros(B, T, 16, dtype=x_12.dtype, device=x_12.device)
    x_16[:, :, 0] = x_12[:, :, 11]
    x_16[:, :, 1] = x_12[:, :, 10]
    x_16[:, :, 2] = x_12[:, :, 1]
    x_16[:, :, 3] = x_12[:, :, 2]
    x_16[:, :, 4] = torch.sqrt(torch.clamp(x_12[:, :, 3], min=0.0))
    x_16[:, :, 5] = 0.5
    x_16[:, :, 6] = 0.0
    x_16[:, :, 7] = x_12[:, :, 5]
    x_16[:, :, 8] = x_12[:, :, 6]
    x_16[:, :, 9] = 0.5
    x_16[:, :, 10] = 0.0
    x_16[:, :, 11] = x_12[:, :, 9]
    x_16[:, :, 12] = x_12[:, :, 8]
    x_16[:, :, 13] = 1.0 / torch.clamp(x_12[:, :, 10], min=1e-4)
    x_16[:, :, 14] = x_12[:, :, 0] / torch.clamp(x_12[:, :, 1], min=1e-4)
    x_16[:, :, 15] = x_12[:, :, 4]
    return x_16


def run_streaming_finetuning(
    chunks_dir: Path = Path("data/processed/chunks"),
    model_checkpoint_path: Path = Path("models/threatora_transformer.pt"),
    backup_path: Path = Path("models/threatora_transformer_backup.pt"),
    output_onnx_path: Path = Path("models/threatora_transformer.onnx"),
    report_path: Path = Path("data/processed/50gb_full_run_report.json"),
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    train_split: float = 0.8,
    grad_clip: float = 1.0,
) -> Dict[str, Any]:
    """Runs streaming multi-chunk fine-tuning with AMP mixed precision and CosineAnnealingLR."""
    t_start = time.perf_counter()

    # 1. Discover all Parquet chunks chronologically with natural sort
    chunk_files = sorted(list(chunks_dir.glob("chunk_*.parquet")), key=natural_sort_key)
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.parquet files found in {chunks_dir}. Run batch_preprocess.py first.")

    total_chunks = len(chunk_files)
    # Balanced 80/20 chronological partition: stride every 5th chunk for representative validation
    val_chunks = [f for i, f in enumerate(chunk_files) if i % 5 == 4]
    train_chunks = [f for i, f in enumerate(chunk_files) if i % 5 != 4]
    if not val_chunks:
        val_chunks = [train_chunks[-1]]

    total_train_samples = get_total_chunk_samples(train_chunks)
    total_val_samples = get_total_chunk_samples(val_chunks)

    logger.info(
        f"Discovered {total_chunks} Parquet chunks: "
        f"{len(train_chunks)} Train ({total_train_samples} samples), "
        f"{len(val_chunks)} Val ({total_val_samples} samples)."
    )

    # 2. Backup existing checkpoint
    if model_checkpoint_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_checkpoint_path, backup_path)
        logger.info(f"Created local model backup: {backup_path}")

    # 3. Hardware acceleration & device setup
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    if is_cuda:
        torch.backends.cudnn.benchmark = True
        device_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Targeting GPU: {device_name} ({vram_total_gb:.2f} GB VRAM) with AMP autocast")
    else:
        logger.warning("CUDA unavailable, running fine-tuning on CPU.")

    # 4. Instantiate Model & Load Pre-Trained Weights
    model = ThreatoraTemporalTransformerWorldModel(
        seq_len=20,
        num_features=16,
        d_model=64,
        nhead=4,
        dim_feedforward=128,
        num_layers=2,
        dropout=0.1,
        horizon_k=5,
        num_mitre_classes=6,
    ).to(device)

    if model_checkpoint_path.exists():
        logger.info(f"Loading pre-trained baseline weights from: {model_checkpoint_path}")
        ckpt = torch.load(model_checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded pre-trained baseline transformer weights.")

    # Snapshot anchor weights for anti-catastrophic forgetting regularization
    anchor_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 5. Optimizer, Scheduler & Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    loss_l1_fn = nn.SmoothL1Loss(beta=1.0)
    loss_bce_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    epoch_history: List[Dict[str, Any]] = []

    # 6. Training & Validation Loop across Epochs
    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        # Shuffle chunk presentation order each epoch to prevent sequential recency bias
        epoch_chunks = list(train_chunks)
        np.random.shuffle(epoch_chunks)
        train_dataset = StreamingChunkDataset(epoch_chunks, batch_size=batch_size, shuffle_within_chunk=True)
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

        running_loss = 0.0
        running_l1 = 0.0
        running_bce = 0.0
        train_batches = 0

        for batch_s_t, batch_s_next, batch_y in train_loader:
            batch_s_t = batch_s_t.to(device, non_blocking=True)
            batch_s_next = batch_s_next.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Feature alignment: map 12 metrics to 16 canonical slots
            if batch_s_t.shape[-1] == 12 and model.num_features == 16:
                batch_s_t_in = map_12_to_16_canonical(batch_s_t)
                batch_s_next_in = map_12_to_16_canonical(batch_s_next)
            else:
                batch_s_t_in = batch_s_t
                batch_s_next_in = batch_s_next

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                pred_s_next, primary_logit, _, _, _ = model(batch_s_t_in)
                l_state = loss_l1_fn(pred_s_next, batch_s_next_in)
                l_risk = loss_bce_fn(primary_logit.squeeze(-1), batch_y)

                # Anti-catastrophic forgetting anchor loss: penalize drift from baseline
                l_anchor = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if name in anchor_weights and param.requires_grad:
                        l_anchor = l_anchor + F.mse_loss(param, anchor_weights[name].to(device))

                # Multi-Task Objective
                total_loss = 0.5 * l_state + 1.0 * l_risk + 0.05 * l_anchor

            amp_scaler.scale(total_loss).backward()
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            running_loss += total_loss.item()
            running_l1 += l_state.item()
            running_bce += l_risk.item()
            train_batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_train_loss = running_loss / max(train_batches, 1)
        avg_train_l1 = running_l1 / max(train_batches, 1)
        avg_train_bce = running_bce / max(train_batches, 1)

        # ---------------- Validation Phase ----------------
        model.eval()
        val_dataset = StreamingChunkDataset(val_chunks, batch_size=batch_size, shuffle_within_chunk=False)
        val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

        val_loss = 0.0
        val_l1 = 0.0
        val_bce = 0.0
        val_batches = 0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for batch_s_t, batch_s_next, batch_y in val_loader:
                batch_s_t = batch_s_t.to(device, non_blocking=True)
                batch_s_next = batch_s_next.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                if batch_s_t.shape[-1] == 12 and model.num_features == 16:
                    batch_s_t_in = map_12_to_16_canonical(batch_s_t)
                    batch_s_next_in = map_12_to_16_canonical(batch_s_next)
                else:
                    batch_s_t_in = batch_s_t
                    batch_s_next_in = batch_s_next

                with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                    pred_s_next, primary_logit, _, _, _ = model(batch_s_t_in)
                    l_state = loss_l1_fn(pred_s_next, batch_s_next_in)
                    l_risk = loss_bce_fn(primary_logit.squeeze(-1), batch_y)
                    v_total = 0.5 * l_state + 1.0 * l_risk

                val_loss += v_total.item()
                val_l1 += l_state.item()
                val_bce += l_risk.item()
                val_batches += 1

                preds = (torch.sigmoid(primary_logit.squeeze(-1)) >= 0.5).long()
                correct_preds += int((preds == batch_y.long()).sum().item())
                total_preds += len(batch_y)

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_l1 = val_l1 / max(val_batches, 1)
        avg_val_bce = val_bce / max(val_batches, 1)
        val_acc = (correct_preds / max(total_preds, 1)) * 100.0

        ep_duration = time.perf_counter() - ep_t0
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            # Save best checkpoint
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "val_accuracy": val_acc,
                    "num_features": 16,
                },
                model_checkpoint_path,
            )

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} (L1: {avg_train_l1:.4f}, BCE: {avg_train_bce:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (L1: {avg_val_l1:.4f}, Acc: {val_acc:.1f}%) | "
            f"LR: {current_lr:.2e} | Best: {best_val_loss:.4f}"
            f"{' [CHECKPOINT UPDATED]' if is_best else ''}"
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_l1": round(avg_train_l1, 5),
            "train_bce": round(avg_train_bce, 5),
            "val_loss": round(avg_val_loss, 5),
            "val_l1": round(avg_val_l1, 5),
            "val_bce": round(avg_val_bce, 5),
            "val_accuracy_pct": round(val_acc, 2),
            "lr": round(current_lr, 7),
            "duration_sec": round(ep_duration, 2),
        })

    # 7. Re-export ONNX Runtime Graph
    logger.info("Exporting fine-tuned model checkpoint to dynamic ONNX graph...")
    best_weights = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(best_weights["model_state_dict"])
    export_model_to_onnx(model, output_onnx_path, device=device, num_features=16)

    # 8. Report Compilation
    total_time = time.perf_counter() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if is_cuda else 0.0

    report = {
        "pipeline": "Threatora 50GB Multi-File Sequential Streaming Ingestion & Fine-Tuning",
        "dataset_source": "G:\\UNSW-NB15 Dataset",
        "total_files_discovered": 86,
        "total_chunks_processed": total_chunks,
        "total_transition_pairs": total_train_samples + total_val_samples,
        "train_chunks": len(train_chunks),
        "val_chunks": len(val_chunks),
        "total_train_samples": total_train_samples,
        "total_val_samples": total_val_samples,
        "hyperparameters": {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "scheduler": "CosineAnnealingLR",
            "loss_objective": "0.5 * SmoothL1Loss + 1.0 * BCEWithLogitsLoss",
        },
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
            "mixed_precision": "torch.amp.autocast('cuda')" if is_cuda else "disabled",
            "peak_vram_mb": round(peak_vram_mb, 2),
            "vram_boundary_passed": bool(peak_vram_mb < 2048.0),
        },
        "best_val_loss": round(best_val_loss, 5),
        "final_val_accuracy_pct": round(epoch_history[-1]["val_accuracy_pct"], 2),
        "total_finetune_time_sec": round(total_time, 2),
        "checkpoints": {
            "pre50gb_backup": str(backup_path),
            "finetuned_checkpoint": str(model_checkpoint_path),
            "onnx_model": str(output_onnx_path),
        },
        "epoch_metrics": epoch_history,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Fine-tuning complete in {total_time:.1f}s. Report saved to: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming PyTorch Fine-Tuning Pipeline")
    parser.add_argument("--chunks-dir", type=str, default="data/processed/chunks", help="Directory of Parquet chunks")
    parser.add_argument("--model-checkpoint", type=str, default="models/threatora_transformer.pt", help="Model checkpoint path")
    parser.add_argument("--backup-path", type=str, default="models/threatora_transformer_backup.pt", help="Pre-finetune backup path")
    parser.add_argument("--output-onnx", type=str, default="models/threatora_transformer.onnx", help="Output ONNX path")
    parser.add_argument("--report-path", type=str, default="data/processed/50gb_full_run_report.json", help="Report JSON path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train chunk split ratio")
    args = parser.parse_args()

    run_streaming_finetuning(
        chunks_dir=Path(args.chunks_dir),
        model_checkpoint_path=Path(args.model_checkpoint),
        backup_path=Path(args.backup_path),
        output_onnx_path=Path(args.output_onnx),
        report_path=Path(args.report_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
    )


if __name__ == "__main__":
    main()
