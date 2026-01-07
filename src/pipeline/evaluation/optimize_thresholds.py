"""
Threshold Optimization for Multi-label Superclass.

Optimizes per-class thresholds on validation set (ensemble probabilities).
MI uses F_beta (beta=2) for recall-focused optimization.
Other classes use Youden J statistic (AUROC-based).

Usage:
    python -m src.pipeline.optimize_thresholds --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_curve, fbeta_score


# Multi-label class order
SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]


def find_threshold_fbeta(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Find optimal threshold maximizing F_beta score.
    
    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities
        beta: F_beta parameter (beta=2 favors recall)
        thresholds: Thresholds to search
        
    Returns:
        (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)
    
    best_threshold = 0.5
    best_score = 0.0
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)
    
    return best_threshold, best_score


def find_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float]:
    """
    Find optimal threshold using Youden's J statistic (sensitivity + specificity - 1).
    
    Returns:
        (best_threshold, best_j_score)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Youden's J = TPR - FPR = sensitivity + specificity - 1
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def optimize_thresholds(
    y_true: np.ndarray,
    y_probs: Dict[str, np.ndarray],
    mi_beta: float = 2.0,
) -> Dict[str, Dict]:
    """
    Optimize thresholds for all classes.
    
    Args:
        y_true: Multi-hot labels (n_samples, 4)
        y_probs: Dict of class -> probabilities
        mi_beta: F_beta parameter for MI threshold
        
    Returns:
        Dict with threshold info per class
    """
    results = {}
    
    for i, cls in enumerate(SUPERCLASS_LABELS):
        y_true_cls = y_true[:, i]
        y_prob_cls = y_probs[cls]
        
        if cls == "MI":
            # MI: recall-focused with F_beta
            threshold, score = find_threshold_fbeta(y_true_cls, y_prob_cls, beta=mi_beta)
            method = f"F_beta (beta={mi_beta})"
        else:
            # Others: Youden's J
            threshold, score = find_threshold_youden(y_true_cls, y_prob_cls)
            method = "Youden_J"
        
        # Also compute F1 at this threshold
        y_pred = (y_prob_cls >= threshold).astype(int)
        f1 = f1_score(y_true_cls, y_pred, zero_division=0)
        
        # Compute recall at this threshold
        tp = ((y_pred == 1) & (y_true_cls == 1)).sum()
        fn = ((y_pred == 0) & (y_true_cls == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        results[cls] = {
            "threshold": threshold,
            "method": method,
            "score": float(score),
            "f1_at_threshold": float(f1),
            "recall_at_threshold": float(recall),
            "support": int(y_true_cls.sum()),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Optimize Multi-label Thresholds")
    parser.add_argument("--cnn-probs", type=Path, required=True,
                        help="Path to CNN probabilities .npz")
    parser.add_argument("--xgb-probs", type=Path, required=True,
                        help="Path to calibrated XGBoost probabilities .npz")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Path to ground truth labels .npz")
    parser.add_argument("--output", type=Path, default=Path("artifacts/thresholds_superclass.json"),
                        help="Output path for thresholds")
    parser.add_argument("--ensemble-weight", type=float, default=0.5,
                        help="Weight for CNN (1-weight for XGB)")
    parser.add_argument("--mi-beta", type=float, default=2.0,
                        help="Beta for MI F_beta optimization")
    parser.add_argument("--mi-recall-min", type=float, default=0.0,
                        help="Minimum recall constraint for MI (0 = no constraint)")
    
    args = parser.parse_args()
    
    print("Loading data...")
    
    # Load CNN probabilities
    cnn_data = np.load(args.cnn_probs)
    cnn_probs = {cls: cnn_data[cls] for cls in SUPERCLASS_LABELS}
    
    # Load XGBoost probabilities
    xgb_data = np.load(args.xgb_probs)
    xgb_probs = {cls: xgb_data[cls] for cls in SUPERCLASS_LABELS}
    
    # Load labels
    labels_data = np.load(args.labels)
    y_true = labels_data["y_multi"]
    
    # Compute ensemble probabilities
    print(f"Computing ensemble (CNN weight={args.ensemble_weight})...")
    w = args.ensemble_weight
    ensemble_probs = {
        cls: w * cnn_probs[cls] + (1 - w) * xgb_probs[cls]
        for cls in SUPERCLASS_LABELS
    }
    
    # Optimize thresholds
    print("Optimizing thresholds...")
    results = optimize_thresholds(y_true, ensemble_probs, mi_beta=args.mi_beta)
    
    # Apply MI recall constraint if specified
    if args.mi_recall_min > 0:
        current_recall = results["MI"]["recall_at_threshold"]
        if current_recall < args.mi_recall_min:
            print(f"MI recall {current_recall:.3f} < min {args.mi_recall_min}")
            print("Searching for threshold meeting recall constraint...")
            
            y_true_mi = y_true[:, 0]
            y_prob_mi = ensemble_probs["MI"]
            
            # Search for threshold that meets recall constraint
            for t in np.linspace(0.01, 0.95, 95):
                y_pred = (y_prob_mi >= t).astype(int)
                tp = ((y_pred == 1) & (y_true_mi == 1)).sum()
                fn = ((y_pred == 0) & (y_true_mi == 1)).sum()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if recall >= args.mi_recall_min:
                    f1 = f1_score(y_true_mi, y_pred, zero_division=0)
                    results["MI"]["threshold"] = float(t)
                    results["MI"]["f1_at_threshold"] = float(f1)
                    results["MI"]["recall_at_threshold"] = float(recall)
                    results["MI"]["method"] += f" + recall_min={args.mi_recall_min}"
                    break
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZED THRESHOLDS")
    print("=" * 60)
    for cls, info in results.items():
        print(f"{cls:6}: threshold={info['threshold']:.3f}, "
              f"F1={info['f1_at_threshold']:.3f}, "
              f"recall={info['recall_at_threshold']:.3f}, "
              f"method={info['method']}")
    
    # Save results
    output = {
        "thresholds": {cls: info["threshold"] for cls, info in results.items()},
        "details": results,
        "ensemble_weight": args.ensemble_weight,
        "mi_beta": args.mi_beta,
        "class_order": SUPERCLASS_LABELS,
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
