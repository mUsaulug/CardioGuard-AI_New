"""
Evaluate Ensemble vs CNN vs XGBoost.

Generates a comparative report (metrics table) using the validation predictions.

Usage:
    python -m src.pipeline.evaluate_ensemble
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]

def compute_metrics(y_true, y_prob):
    """Compute AUROC and AUPRC (Macro)."""
    try:
        auroc = roc_auc_score(y_true, y_prob, average="macro")
    except:
        auroc = 0.0
    
    try:
        auprc = average_precision_score(y_true, y_prob, average="macro")
    except:
        auprc = 0.0
        
    return auroc, auprc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--ensemble-weight", type=float, default=0.5)
    args = parser.parse_args()
    
    # Load data
    print("Loading predictions...")
    cnn_data = np.load(args.pred_dir / "val_cnn_probs.npz")
    xgb_data = np.load(args.pred_dir / "val_xgb_probs.npz")
    lbl_data = np.load(args.pred_dir / "val_labels.npz")
    
    y_true = lbl_data["y_multi"]
    
    # Prepare arrays
    cnn_probs = np.stack([cnn_data[cls] for cls in SUPERCLASS_LABELS], axis=1)
    xgb_probs = np.stack([xgb_data[cls] for cls in SUPERCLASS_LABELS], axis=1)
    
    # Ensemble
    w = args.ensemble_weight
    ens_probs = w * cnn_probs + (1 - w) * xgb_probs
    
    # Compute metrics
    metrics = []
    
    # CNN
    roc, pr = compute_metrics(y_true, cnn_probs)
    metrics.append({"Model": "CNN (Baseline)", "Macro AUROC": roc, "Macro AUPRC": pr})
    
    # XGB
    roc, pr = compute_metrics(y_true, xgb_probs)
    metrics.append({"Model": "XGBoost (OVR)", "Macro AUROC": roc, "Macro AUPRC": pr})
    
    # Ensemble
    roc, pr = compute_metrics(y_true, ens_probs)
    metrics.append({"Model": f"Ensemble (w={w})", "Macro AUROC": roc, "Macro AUPRC": pr})
    
    df = pd.DataFrame(metrics)
    print("\n" + "="*50)
    print("COMPARATIVE PERFORMANCE REPORT (Validation Set)")
    print("="*50)
    print(df.to_string(index=False, float_format="%.4f"))
    print("="*50)
    
    # Per-class analysis for Ensemble
    print("\n[Ensemble Per-Class Breakdown]")
    for i, cls in enumerate(SUPERCLASS_LABELS):
        auc = roc_auc_score(y_true[:, i], ens_probs[:, i])
        ap = average_precision_score(y_true[:, i], ens_probs[:, i])
        print(f"{cls:4} -> AUROC: {auc:.4f}, AUPRC: {ap:.4f}")

if __name__ == "__main__":
    main()
