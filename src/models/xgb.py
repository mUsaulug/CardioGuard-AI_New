"""XGBoost training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import inspect
import warnings

from xgboost import Booster, DMatrix, XGBClassifier


@dataclass
class XGBConfig:
    """Configuration for XGBoost classifier."""

    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "binary:logistic"
    eval_metric: str = "aucpr"
    random_state: int = 42
    early_stopping_rounds: int = 30
    scale_pos_weight: float | None = None


class ManualCalibratedModel:
    """Manual probability calibrator wrapper for models without prefit support."""

    def __init__(self, base_model, calibrator) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        base_proba = self.base_model.predict_proba(features)[:, 1]
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba(base_proba.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(base_proba.reshape(-1, 1))
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1 - calibrated, calibrated])


def _parse_version(version: str) -> Tuple[int, int, int]:
    parts = version.split(".")
    numbers = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        numbers.append(int(digits) if digits else 0)
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers)  # type: ignore[return-value]


def _prefit_calibration_supported() -> bool:
    major, minor, _ = _parse_version(sklearn_version)
    return (major, minor) < (1, 4)


def _manual_calibration(
    model: XGBClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    method: str,
) -> ManualCalibratedModel:
    base_proba = model.predict_proba(features)[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(base_proba, labels)
        return ManualCalibratedModel(model, iso)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(base_proba.reshape(-1, 1), labels)
    return ManualCalibratedModel(model, lr)


def _predict_booster_proba(model: Booster, features: np.ndarray) -> np.ndarray:
    dmatrix = DMatrix(features)
    proba = model.predict(dmatrix)
    return np.asarray(proba)


def predict_xgb(
    model: Union[XGBClassifier, Booster, CalibratedClassifierCV, ManualCalibratedModel],
    features: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and binary labels with an XGBoost model."""
    if isinstance(model, Booster):
        proba = _predict_booster_proba(model, features)
    else:
        proba = model.predict_proba(features)[:, 1]
    preds = (proba >= threshold).astype(int)
    return proba, preds


def compute_binary_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute accuracy, ROC-AUC, PR-AUC, and a classification report."""

    preds = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, preds)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "report": classification_report(y_true, preds, output_dict=True),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Tuple[float, float]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def calibrate_xgb(
    model: XGBClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "sigmoid",
) -> Union[CalibratedClassifierCV, ManualCalibratedModel]:
    """Calibrate XGBoost probabilities using a validation split."""

    if not _prefit_calibration_supported():
        return _manual_calibration(model, features, labels, method)
    try:
        calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
        calibrator.fit(features, labels)
        return calibrator
    except Exception:
        return _manual_calibration(model, features, labels, method)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: XGBConfig | None = None,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Fit an XGBoost classifier and return metrics."""

    config = config or XGBConfig()
    scale_pos_weight = config.scale_pos_weight
    if scale_pos_weight is None:
        positives = float((y_train == 1).sum())
        negatives = float((y_train == 0).sum())
        if positives > 0:
            scale_pos_weight = negatives / positives
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=config.objective,
        eval_metric=config.eval_metric,
        random_state=config.random_state,
        scale_pos_weight=scale_pos_weight,
    )
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    fit_params = set(inspect.signature(model.fit).parameters)
    if "eval_set" not in fit_params:
        fit_kwargs.pop("eval_set", None)
    if "verbose" not in fit_params:
        fit_kwargs.pop("verbose", None)
    if "early_stopping_rounds" in fit_params and config.early_stopping_rounds:
        fit_kwargs["early_stopping_rounds"] = config.early_stopping_rounds
    elif config.early_stopping_rounds:
        warnings.warn(
            "XGBClassifier.fit does not support early_stopping_rounds; continuing without early stopping.",
            RuntimeWarning,
        )
    model.fit(**fit_kwargs)
    val_proba, _ = predict_xgb(model, X_val)
    metrics = compute_binary_metrics(y_val, val_proba)
    metrics["best_iteration"] = int(getattr(model, "best_iteration", -1))
    metrics["best_score"] = float(getattr(model, "best_score", 0.0))
    return model, metrics


def save_xgb(model: XGBClassifier, path: str | Path) -> None:
    """Save XGBoost model to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save_model(path)
    except TypeError:
        booster = model.get_booster()
        booster.save_model(path)


def load_xgb(path: str | Path) -> Union[XGBClassifier, Booster]:
    """Load XGBoost model from JSON."""
    model = XGBClassifier()
    try:
        model.load_model(path)
        return model
    except TypeError:
        booster = Booster()
        booster.load_model(path)
        return booster
