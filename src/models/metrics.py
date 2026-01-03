"""Metric utilities for model evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_classification_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, F1, and accuracy from logits."""

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    y_probs = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred)
    accuracy = float((y_pred == y_true).mean())

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "accuracy": accuracy,
    }
