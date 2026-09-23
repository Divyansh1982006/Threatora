"""Baseline Models for Benchmark Comparison (NTRO PS 26153).

Provides:
  1. Single-window Logistic Regression baseline
  2. Stacked-history Logistic Regression baseline (past 8 windows)
  3. Persistence baseline ("status quo / nothing changes")
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

# Defensive import: Windows Application Control or minimal envs may block sklearn C-extensions
try:
    from sklearn.linear_model import LogisticRegression as _SklearnLR
    from sklearn.metrics import precision_recall_fscore_support as _sk_prf, roc_auc_score as _sk_auc, confusion_matrix as _sk_cm
    HAS_SKLEARN = True
except Exception:
    _SklearnLR = None
    _sk_prf = None
    _sk_auc = None
    _sk_cm = None
    HAS_SKLEARN = False


def compute_metrics_fallback(y_true: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Computes precision, recall, f1, fpr, and roc_auc using sklearn or pure numpy fallback."""
    if HAS_SKLEARN:
        try:
            prec, rec, f1, _ = _sk_prf(y_true, preds, average="binary", zero_division=0)
            tn, fp, fn, tp = _sk_cm(y_true, preds, labels=[0, 1]).ravel()
            fpr = float(fp / max(fp + tn, 1))
            try:
                auc = float(_sk_auc(y_true, probs))
            except Exception:
                auc = 0.5
            return {
                "precision": float(round(prec, 4)),
                "recall": float(round(rec, 4)),
                "f1_score": float(round(f1, 4)),
                "false_positive_rate": float(round(fpr, 4)),
                "roc_auc": float(round(auc, 4)),
            }
        except Exception:
            pass

    # Pure NumPy fallback
    y_true_arr = np.asarray(y_true, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)
    probs_arr = np.asarray(probs, dtype=float)

    tp = int(np.sum((y_true_arr == 1) & (preds_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (preds_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (preds_arr == 0)))
    tn = int(np.sum((y_true_arr == 0) & (preds_arr == 0)))

    prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0

    # Mann-Whitney / Wilcoxon rank sum for ROC AUC approximation
    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(probs_arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(probs_arr) + 1)
        rank_sum = np.sum(ranks[y_true_arr == 1])
        auc = float((rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))
        auc = max(0.0, min(1.0, auc))
    else:
        auc = 0.5

    return {
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "false_positive_rate": float(round(fpr, 4)),
        "roc_auc": float(round(auc, 4)),
    }


class LogisticRegressionBaseline:
    """Logistic Regression baseline for point-in-time and stacked history detection."""

    def __init__(self, history_len: int = 1):
        self.history_len = history_len
        self.is_fitted = False
        self._use_sklearn = False
        self.model = None
        self.weights = None
        self.bias = 0.0

        if HAS_SKLEARN and _SklearnLR is not None:
            try:
                self.model = _SklearnLR(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42
                )
                self._use_sklearn = True
            except Exception:
                self._use_sklearn = False

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
        if self._use_sklearn and self.model is not None:
            try:
                self.model.fit(X_flat, y)
                self.is_fitted = True
                return
            except Exception:
                self._use_sklearn = False

        # Pure NumPy SGD Logistic Regression Fallback
        b, d = X_flat.shape
        self.weights = np.zeros(d, dtype=np.float32)
        self.bias = 0.0
        lr = 0.05
        y_vec = np.asarray(y, dtype=np.float32)

        for _ in range(80):
            logits = np.dot(X_flat, self.weights) + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            err = probs - y_vec
            grad_w = np.dot(X_flat.T, err) / max(b, 1)
            grad_b = float(np.mean(err))
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._prepare_features(X)
        if not self.is_fitted:
            return np.full(len(X_flat), 0.5)

        if self._use_sklearn and self.model is not None:
            try:
                return self.model.predict_proba(X_flat)[:, 1]
            except Exception:
                pass

        if self.weights is not None:
            logits = np.dot(X_flat, self.weights) + self.bias
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

        return np.full(len(X_flat), 0.5)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)
        return compute_metrics_fallback(y, preds, probs)


class PersistenceBaseline:
    """Persistence baseline assuming no change in attack state over future horizons."""

    def __init__(self):
        pass

    def forecast_stage(self, current_stages: np.ndarray, horizon: int) -> np.ndarray:
        """Predicts stage at t+k is identical to stage at t."""
        # current_stages shape: (B,)
        return np.tile(current_stages[:, None], (1, horizon))

