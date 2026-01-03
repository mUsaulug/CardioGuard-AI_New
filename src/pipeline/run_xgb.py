"""Train and evaluate an XGBoost classifier on saved CNN features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.models.xgb import XGBConfig, compute_binary_metrics, predict_xgb, train_xgb


def load_features(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load features, labels, and optional ids from a .npz file."""

    data = np.load(path)
    features = data["features"]
    labels = data["labels"]
    ids = data["ids"] if "ids" in data else None
    return features, labels, ids


def evaluate_split(
    name: str,
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    proba, _ = predict_xgb(model, features)
    metrics = compute_binary_metrics(labels, proba)
    return {"split": name, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on CNN embeddings")
    parser.add_argument("--train", required=True, help="Path to train .npz features")
    parser.add_argument("--val", required=True, help="Path to val .npz features")
    parser.add_argument("--test", required=True, help="Path to test .npz features")
    parser.add_argument("--output-dir", default="logs/xgb", help="Directory for metrics.json")
    args = parser.parse_args()

    X_train, y_train, _ = load_features(args.train)
    X_val, y_val, _ = load_features(args.val)
    X_test, y_test, _ = load_features(args.test)

    model, val_metrics = train_xgb(X_train, y_train, X_val, y_val, XGBConfig())

    results = {
        "val": val_metrics,
        "train": evaluate_split("train", model, X_train, y_train)["metrics"],
        "test": evaluate_split("test", model, X_test, y_test)["metrics"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
