"""
XGBoost One-vs-Rest Training for Multi-label Superclass.

Trains 4 separate binary XGBoost models (MI, STTC, CD, HYP).
Each model uses CNN embeddings as features.
Platt calibration applied per model on validation set.

Usage:
    python -m src.pipeline.train_superclass_xgb_ovr
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# Multi-label class order (NORM is derived)
SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load features and multi-label targets from npz."""
    data = np.load(path, allow_pickle=True)
    
    # Features
    if "X" in data:
        X = data["X"]
    elif "features" in data:
        X = data["features"]
    else:
        raise ValueError(f"Missing features in {path}")
    
    # Multi-label targets (shape: n_samples, 4)
    if "y_multi" in data:
        y = data["y_multi"]
    elif "y" in data:
        y = data["y"]
    else:
        raise ValueError(f"Missing labels in {path}")
    
    return X, y


def train_binary_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_name: str,
    seed: int = 42,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train a single binary XGBoost for one class."""
    # Compute scale_pos_weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Evaluate
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    
    metrics = {
        "class": class_name,
        "train_samples": len(y_train),
        "train_positive": int(y_train.sum()),
        "val_samples": len(y_val),
        "val_positive": int(y_val.sum()),
        "scale_pos_weight": float(scale_pos_weight),
    }
    
    # Handle case where validation has only one class
    if len(np.unique(y_val)) > 1:
        metrics["val_auroc"] = float(roc_auc_score(y_val, val_proba))
        metrics["val_auprc"] = float(average_precision_score(y_val, val_proba))
    else:
        metrics["val_auroc"] = None
        metrics["val_auprc"] = None
    
    metrics["val_f1"] = float(f1_score(y_val, val_pred, zero_division=0))
    
    return model, metrics


def calibrate_model(
    model: XGBClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> LogisticRegression:
    """
    Platt calibration (sigmoid) or Isotonic calibration using validation set.
    Returns a calibrator that transforms raw probabilities.
    """
    # Get uncalibrated probabilities
    raw_proba = model.predict_proba(X_val)[:, 1]
    
    if method == "isotonic":
        # Use IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_proba, y_val)
    else:
        # Default: Sigmoid (Platt) via LogisticRegression
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(raw_proba.reshape(-1, 1), y_val)
    
    return calibrator


class CalibratedOVRModel:
    """Wrapper for calibrated OVR predictions."""
    
    def __init__(
        self,
        models: Dict[str, XGBClassifier],
        calibrators: Dict[str, LogisticRegression],
        scaler: StandardScaler,
        class_order: list = None,
    ):
        self.models = models
        self.calibrators = calibrators
        self.scaler = scaler
        self.class_order = class_order or SUPERCLASS_LABELS
    
    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> Dict[str, np.ndarray]:
        """Predict calibrated probabilities for each class."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        probs = {}
        for cls in self.class_order:
            model = self.models[cls]
            raw_proba = model.predict_proba(X)[:, 1]
            
            if calibrated and cls in self.calibrators:
                calibrator = self.calibrators[cls]
                # Isotonic expects 1D array, LogisticRegression expects 2D
                if isinstance(calibrator, IsotonicRegression):
                     proba = calibrator.predict(raw_proba)
                else:
                     proba = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
            else:
                proba = raw_proba
            
            probs[cls] = proba
        
        return probs
    
    def predict_proba_array(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Return probabilities as (n_samples, n_classes) array."""
        probs = self.predict_proba(X, calibrated)
        return np.column_stack([probs[cls] for cls in self.class_order])


def evaluate_ovr(
    ovr_model: CalibratedOVRModel,
    X: np.ndarray,
    y: np.ndarray,
    calibrated: bool = True,
) -> Dict[str, Any]:
    """Evaluate OVR model on a dataset."""
    probs = ovr_model.predict_proba(X, calibrated=calibrated)
    
    results = {"per_class": {}}
    valid_aurocs = []
    valid_auprcs = []
    
    for i, cls in enumerate(ovr_model.class_order):
        y_true = y[:, i]
        y_prob = probs[cls]
        y_pred = (y_prob >= 0.5).astype(int)
        
        cls_metrics = {
            "support": int(y_true.sum()),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        if len(np.unique(y_true)) > 1:
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            cls_metrics["auroc"] = float(auroc)
            cls_metrics["auprc"] = float(auprc)
            valid_aurocs.append(auroc)
            valid_auprcs.append(auprc)
        else:
            cls_metrics["auroc"] = None
            cls_metrics["auprc"] = None
        
        results["per_class"][cls] = cls_metrics
    
    results["macro_auroc"] = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0
    results["macro_auprc"] = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0
    
    return results


def load_xgb_ovr_models(output_dir: Path) -> CalibratedOVRModel:
    """
    Load trained OVR models and return CalibratedOVRModel.
    
    Args:
        output_dir: Directory containing trained models
        
    Returns:
        CalibratedOVRModel instance
    """
    models = {}
    calibrators = {}
    
    # Load scaler
    scaler_path = output_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    for cls in SUPERCLASS_LABELS:
        model_dir = output_dir / cls
        
        # Load model
        model = XGBClassifier()
        try:
            model.load_model(model_dir / "xgb_model.json")
            models[cls] = model
        except Exception as e:
            print(f"Warning: Could not load model for {cls}: {e}")
            continue
        
        # Load calibrator
        cal_path = model_dir / "calibrator.joblib"
        if cal_path.exists():
            calibrators[cls] = joblib.load(cal_path)
            
    return CalibratedOVRModel(models, calibrators, scaler, SUPERCLASS_LABELS)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost OVR for Superclass")
    parser.add_argument("--train", default="features_out/train_superclass.npz",
                        help="Path to train features")
    parser.add_argument("--val", default="features_out/val_superclass.npz",
                        help="Path to val features")
    parser.add_argument("--test", default="features_out/test_superclass.npz",
                        help="Path to test features")
    parser.add_argument("--output-dir", default="logs/xgb_superclass",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration", default="sigmoid",
                        choices=["sigmoid", "isotonic", "none"])
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print("Loading features...")
    X_train, y_train = load_features(Path(args.train))
    X_val, y_val = load_features(Path(args.val))
    X_test, y_test = load_features(Path(args.test))
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train OVR models
    models = {}
    calibrators = {}
    all_metrics = []
    
    print("\nTraining OVR models...")
    print("=" * 60)
    
    for i, cls in enumerate(SUPERCLASS_LABELS):
        print(f"\n[{i+1}/4] Training {cls} model...")
        
        # Extract binary labels for this class
        y_train_cls = y_train[:, i]
        y_val_cls = y_val[:, i]
        
        # Train model
        model, metrics = train_binary_xgb(
            X_train_scaled, y_train_cls,
            X_val_scaled, y_val_cls,
            class_name=cls,
            seed=args.seed,
        )
        models[cls] = model
        all_metrics.append(metrics)
        
        # Calibrate
        if args.calibration != "none" and len(np.unique(y_val_cls)) > 1:
            print(f"  Calibrating with {args.calibration}...")
            calibrator = calibrate_model(model, X_val_scaled, y_val_cls, args.calibration)
            calibrators[cls] = calibrator
        
        # Print metrics
        auroc = metrics.get("val_auroc", "N/A")
        auprc = metrics.get("val_auprc", "N/A")
        if isinstance(auroc, float):
            print(f"  Val AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {metrics['val_f1']:.4f}")
        else:
            print(f"  Val F1: {metrics['val_f1']:.4f} (single class in val)")
        
        # Save model
        model_dir = output_dir / cls
        model_dir.mkdir(parents=True, exist_ok=True)
        # Use get_booster().save_model() to avoid sklearn metadata issues
        model.get_booster().save_model(model_dir / "xgb_model.json")
        
        if cls in calibrators:
            joblib.dump(calibrators[cls], model_dir / "calibrator.joblib")
    
    # Save scaler
    joblib.dump(scaler, output_dir / "scaler.joblib")
    
    # Create combined model
    ovr_model = CalibratedOVRModel(models, calibrators, scaler, SUPERCLASS_LABELS)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    
    test_metrics = evaluate_ovr(ovr_model, X_test, y_test, calibrated=True)
    
    print(f"\nTest Results:")
    print(f"  Macro AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"  Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
    print("\nPer-class metrics:")
    for cls, cls_metrics in test_metrics["per_class"].items():
        auroc = cls_metrics.get("auroc", "N/A")
        auprc = cls_metrics.get("auprc", "N/A")
        if isinstance(auroc, float):
            print(f"  {cls:6}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, "
                  f"F1={cls_metrics['f1']:.4f}, support={cls_metrics['support']}")
        else:
            print(f"  {cls:6}: F1={cls_metrics['f1']:.4f}, support={cls_metrics['support']}")
    
    # Save results
    results = {
        "class_order": SUPERCLASS_LABELS,
        "calibration_method": args.calibration,
        "training_metrics": all_metrics,
        "test_metrics": test_metrics,
        "seed": args.seed,
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
