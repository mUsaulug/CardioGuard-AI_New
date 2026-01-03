"""SHAP explanations for XGBoost models."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import shap
from xgboost import XGBClassifier


def explain_xgb(
    model: XGBClassifier,
    X: np.ndarray,
    max_samples: int = 1000,
) -> Dict[str, Any]:
    """Compute SHAP values for the given samples."""

    samples = X[:max_samples]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(samples)
    base_value = explainer.expected_value
    return {
        "shap_values": shap_values,
        "base_value": base_value,
        "samples": samples,
    }
