"""CardioGuard-AI XAI (Explainable AI) Module."""

from src.xai.gradcam import GradCAM
from src.xai.shap_xgb import (
    explain_xgb,
    get_top_features,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.xai.visualize import (
    LEAD_NAMES,
    plot_ecg_with_localization,
    plot_ecg_with_prediction,
    plot_gradcam_heatmap,
    plot_lead_attention,
)
from src.xai.summary import (
    compute_lead_attention,
    compute_top_shap_feature,
    summarize_visual_explanations,
)

__all__ = [
    "GradCAM",
    "explain_xgb",
    "get_top_features",
    "plot_shap_summary",
    "plot_shap_waterfall",
    "LEAD_NAMES",
    "plot_ecg_with_localization",
    "plot_ecg_with_prediction",
    "plot_gradcam_heatmap",
    "plot_lead_attention",
    "compute_lead_attention",
    "compute_top_shap_feature",
    "summarize_visual_explanations",
]
