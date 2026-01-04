"""SHAP explanations for XGBoost models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import Booster, XGBClassifier


def explain_xgb(
    model: XGBClassifier | Booster,
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


def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    max_display: int = 20,
    plot_type: str = "bar",
    title: str = "Feature Importance (SHAP)",
) -> plt.Figure:
    """
    Generate SHAP summary plot.
    
    Args:
        shap_values: SHAP values array of shape (n_samples, n_features)
        features: Feature matrix of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        save_path: Optional path to save figure
        max_display: Maximum number of features to display
        plot_type: "bar" for mean importance, "beeswarm" for distribution
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Generate default feature names if not provided
    if feature_names is None:
        n_features = features.shape[1] if len(features.shape) > 1 else 1
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if plot_type == "bar":
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1][:max_display]
        
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, mean_shap[sorted_idx], color="#1f77b4", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        
    else:  # beeswarm style using matplotlib
        mean_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1][:max_display]
        
        for i, idx in enumerate(sorted_idx):
            y = np.full(len(shap_values), i) + np.random.uniform(-0.2, 0.2, len(shap_values))
            colors = plt.cm.RdBu_r((features[:, idx] - features[:, idx].min()) / 
                                    (features[:, idx].max() - features[:, idx].min() + 1e-8))
            ax.scatter(shap_values[:, idx], y, c=colors, s=10, alpha=0.5)
        
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel("SHAP value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP summary plot to {save_path}")
    
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    base_value: float,
    sample_idx: int,
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    max_display: int = 15,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Generate SHAP waterfall plot for a single prediction.
    
    Args:
        shap_values: SHAP values array of shape (n_samples, n_features)
        base_value: Expected value (baseline prediction)
        sample_idx: Index of sample to explain
        features: Feature matrix of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        save_path: Optional path to save figure
        max_display: Maximum number of features to display
        title: Optional plot title
        
    Returns:
        matplotlib Figure object
    """
    sample_shap = shap_values[sample_idx]
    sample_features = features[sample_idx]
    
    if feature_names is None:
        n_features = len(sample_shap)
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:max_display]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Starting point
    cumsum = base_value
    y_positions = []
    widths = []
    starts = []
    colors = []
    labels = []
    
    for i, idx in enumerate(sorted_idx):
        y_positions.append(i)
        shap_val = sample_shap[idx]
        widths.append(abs(shap_val))
        starts.append(min(cumsum, cumsum + shap_val))
        colors.append("#d62728" if shap_val > 0 else "#2ca02c")  # Red positive, Green negative
        labels.append(f"{feature_names[idx]} = {sample_features[idx]:.3f}")
        cumsum += shap_val
    
    # Plot bars
    ax.barh(y_positions, widths, left=starts, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add annotations
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    
    # Add baseline and prediction lines
    ax.axvline(x=base_value, color="gray", linewidth=2, linestyle="--", label=f"Base: {base_value:.3f}")
    final_pred = base_value + sample_shap.sum()
    ax.axvline(x=final_pred, color="black", linewidth=2, label=f"Prediction: {final_pred:.3f}")
    
    ax.set_xlabel("SHAP value contribution", fontsize=12)
    ax.legend(loc="lower right")
    
    if title is None:
        title = f"SHAP Waterfall (Sample {sample_idx})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved SHAP waterfall plot to {save_path}")
    
    return fig


def get_top_features(
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get top-k most important features based on mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values array of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        top_k: Number of top features to return
        
    Returns:
        List of dicts with feature name and importance
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(mean_shap))]
    
    sorted_idx = np.argsort(mean_shap)[::-1][:top_k]
    
    return [
        {"feature": feature_names[idx], "importance": float(mean_shap[idx])}
        for idx in sorted_idx
    ]
