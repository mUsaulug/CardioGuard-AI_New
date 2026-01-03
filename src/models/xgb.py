"""XGBoost training and evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


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
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "report": classification_report(y_val, preds, output_dict=True),
    }
    return model, metrics
