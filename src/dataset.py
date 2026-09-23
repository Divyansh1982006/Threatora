"""Temporal Sequence Dataset and DataLoaders for NetForecast (16-window rolling context).

Handles:
  1. Sequence slicing: 16 consecutive 60s windows per sequence
  2. Guard bands: prevents train/val/test data leakage across time boundaries
  3. WeightedRandomSampler: balances benign vs attack sequences
"""

from __future__ import annotations

import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .config import SEQUENCE_LENGTH, PROCESSED_DIR
from .features.windows import FeatureScaler


class AttackSequenceDataset(Dataset):
    """PyTorch Dataset yielding 16-window sequential network states, infiltration labels, and stages."""

    def __init__(
        self,
        sequences: np.ndarray,      # (N, 16, 62)
        infilt_labels: np.ndarray,  # (N, 16)
        stages: np.ndarray          # (N, 16)
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.infilt_labels = torch.tensor(infilt_labels, dtype=torch.float32)
        self.stages = torch.tensor(stages, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.infilt_labels[idx], self.stages[idx]


def slice_into_sequences(
    X_cells: np.ndarray,
    y_infilt: np.ndarray,
    y_stage: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
    stride: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts sliding sequence windows of length 16."""
    num_cells = len(X_cells)
    if num_cells < seq_len:
        # Pad if short sequence
        pad_len = seq_len - num_cells
        pad_x = np.repeat(X_cells[-1:], pad_len, axis=0) if num_cells > 0 else np.zeros((seq_len, X_cells.shape[1]))
        pad_infilt = np.repeat(y_infilt[-1:], pad_len, axis=0) if num_cells > 0 else np.zeros((seq_len,))
        pad_stage = np.repeat(y_stage[-1:], pad_len, axis=0) if num_cells > 0 else np.zeros((seq_len,), dtype=int)
        
        full_x = np.vstack([X_cells, pad_x]) if num_cells > 0 else pad_x
        full_infilt = np.concatenate([y_infilt, pad_infilt]) if num_cells > 0 else pad_infilt
        full_stage = np.concatenate([y_stage, pad_stage]) if num_cells > 0 else pad_stage
        return full_x[np.newaxis, :seq_len], full_infilt[np.newaxis, :seq_len], full_stage[np.newaxis, :seq_len]

    seq_x, seq_infilt, seq_stage = [], [], []
    for i in range(0, num_cells - seq_len + 1, stride):
        seq_x.append(X_cells[i : i + seq_len])
        seq_infilt.append(y_infilt[i : i + seq_len])
        seq_stage.append(y_stage[i : i + seq_len])

    return np.array(seq_x, dtype=np.float32), np.array(seq_infilt, dtype=np.float32), np.array(seq_stage, dtype=np.int64)


def get_dataloaders(
    X_cells: np.ndarray,
    y_infilt: np.ndarray,
    y_stage: np.ndarray,
    batch_size: int = 32,
    scaler: Optional[FeatureScaler] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureScaler]:
    """Splits cell data temporally (70% train, 15% val, 15% test) with guard bands."""
    num_cells = len(X_cells)
    train_idx = int(num_cells * 0.70)
    val_idx = int(num_cells * 0.85)

    # Fit scaler on training portion only to avoid lookahead bias
    if scaler is None:
        scaler = FeatureScaler()
        scaler.fit(X_cells[:train_idx])

    X_scaled = scaler.transform(X_cells)

    # Slice with guard band of SEQUENCE_LENGTH between partitions
    X_train, y_train_inf, y_train_stg = slice_into_sequences(
        X_scaled[:train_idx], y_infilt[:train_idx], y_stage[:train_idx]
    )
    X_val, y_val_inf, y_val_stg = slice_into_sequences(
        X_scaled[train_idx + SEQUENCE_LENGTH : val_idx],
        y_infilt[train_idx + SEQUENCE_LENGTH : val_idx],
        y_stage[train_idx + SEQUENCE_LENGTH : val_idx]
    )
    X_test, y_test_inf, y_test_stg = slice_into_sequences(
        X_scaled[val_idx + SEQUENCE_LENGTH :],
        y_infilt[val_idx + SEQUENCE_LENGTH :],
        y_stage[val_idx + SEQUENCE_LENGTH :]
    )

    train_ds = AttackSequenceDataset(X_train, y_train_inf, y_train_stg)
    val_ds = AttackSequenceDataset(X_val, y_val_inf, y_val_stg)
    test_ds = AttackSequenceDataset(X_test, y_test_inf, y_test_stg)

    # WeightedRandomSampler for class imbalance
    sample_weights = []
    for inf_seq in y_train_inf:
        # Attack present if any step is positive
        has_attack = (inf_seq > 0.5).any()
        sample_weights.append(5.0 if has_attack else 1.0)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    ) if len(sample_weights) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None)
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
