"""Metric utilities for model evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, F1, and accuracy from logits."""

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    if y_logits.ndim > 1 and y_logits.shape[1] > 1:
        return compute_multiclass_metrics(y_true, y_logits)
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


def compute_multiclass_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, float]:
    """Compute macro ROC-AUC, PR-AUC, F1, and accuracy for multi-class logits."""

    y_true = np.asarray(y_true)
    y_logits = np.asarray(y_logits)
    num_classes = y_logits.shape[1]
    y_probs = np.exp(y_logits - y_logits.max(axis=1, keepdims=True))
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_probs, axis=1)

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    roc_auc = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
    pr_auc = average_precision_score(y_true_bin, y_probs, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    accuracy = float((y_pred == y_true).mean())

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "accuracy": accuracy,
    }
