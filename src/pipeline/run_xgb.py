"""Train and evaluate an XGBoost classifier on saved CNN features.

Note: XGBoost requires labels to be present in the saved .npz features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from src.models.xgb import (
    XGBConfig,
    calibrate_xgb,
    compute_binary_metrics,
    find_best_threshold,
    predict_xgb,
    train_xgb,
)


def load_features(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load features, labels, and optional ids from a .npz file."""

    data = np.load(path)
    features = data["features"]
    if "labels" not in data:
        raise ValueError(
            f"Missing labels in features file: {path}. "
            "XGBoost training requires labels in the .npz archive."
        )
    labels = data["labels"]
    if labels.size == 0:
        raise ValueError(
            f"Empty labels in features file: {path}. "
            "XGBoost training requires non-empty labels."
        )
    ids = data["ids"] if "ids" in data else None
    return features, labels, ids


def evaluate_split(
    name: str,
    model,
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    proba, _ = predict_xgb(model, features)
    metrics = compute_binary_metrics(labels, proba, threshold=threshold)
    return {"split": name, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on CNN embeddings")
    parser.add_argument("--train", required=True, help="Path to train .npz features")
    parser.add_argument("--val", required=True, help="Path to val .npz features")
    parser.add_argument("--test", required=True, help="Path to test .npz features")
    parser.add_argument("--output-dir", default="logs/xgb", help="Directory for metrics.json")
    parser.add_argument(
        "--no-scale-features",
        action="store_true",
        help="Disable StandardScaler normalization for CNN embeddings.",
    )
    parser.add_argument(
        "--calibration",
        choices=["sigmoid", "isotonic"],
        default=None,
        help="Optional probability calibration method (uses validation set).",
    )
    args = parser.parse_args()

    X_train, y_train, _ = load_features(args.train)
    X_val, y_val, _ = load_features(args.val)
    X_test, y_test, _ = load_features(args.test)

    scaler = None
    if not args.no_scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    model, val_metrics = train_xgb(X_train, y_train, X_val, y_val, XGBConfig())
    calibrated_model = None
    model_for_eval = model
    if args.calibration is not None:
        calibrated_model = calibrate_xgb(model, X_val, y_val, method=args.calibration)
        model_for_eval = calibrated_model

    val_proba, _ = predict_xgb(model_for_eval, X_val)
    best_threshold, best_threshold_f1 = find_best_threshold(y_val, val_proba)

    results = {
        "val": {
            **val_metrics,
            "best_threshold": best_threshold,
            "best_threshold_f1": best_threshold_f1,
            "calibration_method": args.calibration,
        },
        "train": evaluate_split(
            "train",
            model_for_eval,
            X_train,
            y_train,
            threshold=best_threshold,
        )["metrics"],
        "test": evaluate_split(
            "test",
            model_for_eval,
            X_test,
            y_test,
            threshold=best_threshold,
        )["metrics"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    model_path = output_dir / "xgb_model.json"
    model.save_model(model_path)
    if calibrated_model is not None:
        calibrated_path = output_dir / "xgb_calibrated.joblib"
        joblib.dump(calibrated_model, calibrated_path)
        print(f"Saved calibrated XGBoost model to: {calibrated_path}")
    if scaler is not None:
        scaler_path = output_dir / "xgb_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Saved XGBoost scaler to: {scaler_path}")

    print(json.dumps(results, indent=2))
    print(f"Saved XGBoost model to: {model_path}")


if __name__ == "__main__":
    main()
