"""Tests for XGBoost pipeline utilities."""

from __future__ import annotations

import numpy as np

from src.models.xgb import compute_binary_metrics
from src.pipeline.run_xgb import load_features


def test_load_features_accepts_X_y(tmp_path) -> None:
    path = tmp_path / "features_xy.npz"
    X = np.random.randn(4, 8)
    y = np.array([0, 1, 0, 1])
    np.savez_compressed(path, X=X, y=y)

    features, labels, ids = load_features(path)

    assert np.allclose(features, X)
    assert np.allclose(labels, y)
    assert ids is None


def test_load_features_accepts_features_labels(tmp_path) -> None:
    path = tmp_path / "features_legacy.npz"
    features = np.random.randn(3, 5)
    labels = np.array([1, 0, 1])
    np.savez_compressed(path, features=features, labels=labels)

    loaded_features, loaded_labels, _ = load_features(path)

    assert np.allclose(loaded_features, features)
    assert np.allclose(loaded_labels, labels)


def test_compute_binary_metrics_includes_f1_and_confusion() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.6, 0.4])

    metrics = compute_binary_metrics(y_true, y_proba, threshold=0.5)

    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
