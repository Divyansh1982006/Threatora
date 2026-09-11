"""Baseline Models for Benchmark Comparison (NTRO PS 26153).

Provides:
  1. Single-window Logistic Regression baseline
  2. Stacked-history Logistic Regression baseline (past 8 windows)
  3. Persistence baseline ("status quo / nothing changes")
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix


class LogisticRegressionBaseline:
    """Logistic Regression baseline for point-in-time and stacked history detection."""

    def __init__(self, history_len: int = 1):
        self.history_len = history_len
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        )
        self.is_fitted = False

    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Flattens sequence windows if 3D (B, T, D)."""
        if X.ndim == 3:
            b, t, d = X.shape
            if self.history_len == 1:
                # Last window only
                return X[:, -1, :]
            else:
                # Stack past history_len windows
                hist = min(self.history_len, t)
                return X[:, -hist:, :].reshape(b, -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_flat = self._prepare_features(X)
        self.model.fit(X_flat, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._prepare_features(X)
        if not self.is_fitted:
            return np.full(len(X_flat), 0.5)
        return self.model.predict_proba(X_flat)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        
        # False Positive Rate: FP / (FP + TN)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        fpr = float(fp / max(fp + tn, 1))

        try:
            auc = float(roc_auc_score(y, probs))
        except Exception:
            auc = 0.5

        return {
            "precision": float(round(prec, 4)),
            "recall": float(round(rec, 4)),
            "f1_score": float(round(f1, 4)),
            "false_positive_rate": float(round(fpr, 4)),
            "roc_auc": float(round(auc, 4))
        }


class PersistenceBaseline:
    """Persistence baseline assuming no change in attack state over future horizons."""

    def __init__(self):
        pass

    def forecast_stage(self, current_stages: np.ndarray, horizon: int) -> np.ndarray:
        """Predicts stage at t+k is identical to stage at t."""
        # current_stages shape: (B,)
        return np.tile(current_stages[:, None], (1, horizon))
