"""XGBoost training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
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
    eval_metric: str = "logloss"
    random_state: int = 42


def _predict_booster_proba(model: Booster, features: np.ndarray) -> np.ndarray:
    dmatrix = DMatrix(features)
    proba = model.predict(dmatrix)
    return np.asarray(proba)


def predict_xgb(
    model: Union[XGBClassifier, Booster],
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
        "report": classification_report(y_true, preds, output_dict=True),
    }


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: XGBConfig | None = None,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Fit an XGBoost classifier and return metrics."""

    config = config or XGBConfig()
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=config.objective,
        eval_metric=config.eval_metric,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_proba, _ = predict_xgb(model, X_val)
    metrics = compute_binary_metrics(y_val, val_proba)
    return model, metrics


def save_xgb(model: XGBClassifier, path: str | Path) -> None:
    """Save XGBoost model to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


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
