"""
XAI Visualization Module for CardioGuard-AI.

Provides visualization functions for:
- Grad-CAM heatmaps on ECG signals
- Lead-wise attention analysis
- SHAP summary and waterfall plots
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Standard 12-lead ECG names
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def plot_gradcam_heatmap(
    signal: np.ndarray,
    cam: np.ndarray,
    lead_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Grad-CAM Attention on ECG",
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = "hot",
    localization_bounds: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Plot 12-lead ECG with Grad-CAM attention overlay.
    
    Args:
        signal: ECG signal array of shape (12, T) or (T, 12)
        cam: Grad-CAM attention array of shape (T,)
        lead_names: List of lead names (default: standard 12-lead)
        save_path: Optional path to save figure
        title: Plot title
        figsize: Figure size
        cmap: Colormap for attention overlay
        localization_bounds: Optional (start, end) indices for localization overlay
        
    Returns:
        matplotlib Figure object
    """
    if lead_names is None:
        lead_names = LEAD_NAMES
    
    # Ensure signal is (12, T)
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
    
    n_leads, n_samples = signal.shape
    
    # Normalize CAM to [0, 1]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(n_leads, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Time axis
    time = np.arange(n_samples)
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        # Plot ECG signal
        ax.plot(time, signal[i], color="black", linewidth=0.8, alpha=0.8)

        # Overlay attention as colored background
        for j in range(n_samples - 1):
            ax.axvspan(
                time[j], time[j + 1],
                alpha=cam_norm[j] * 0.6,
                color=plt.cm.get_cmap(cmap)(cam_norm[j]),
                linewidth=0,
            )

        if localization_bounds is not None:
            start, end = localization_bounds
            ax.axvspan(
                start,
                end,
                color="#1f77b4",
                alpha=0.15,
                linestyle="--",
                linewidth=1.0,
                edgecolor="#1f77b4",
            )
        
        # Lead label
        ax.set_ylabel(lead_name, fontsize=10, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    
    axes[-1].set_xlabel("Sample", fontsize=12)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Attention", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved Grad-CAM heatmap to {save_path}")
    
    return fig


def plot_ecg_with_localization(
    signal: np.ndarray,
    localization_bounds: Tuple[int, int],
    lead_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Localization Overlay",
    figsize: Tuple[int, int] = (14, 10),
    line_color: str = "#222222",
) -> plt.Figure:
    """
    Plot 12-lead ECG with localization bounds overlayed on the time axis.
    """
    if lead_names is None:
        lead_names = LEAD_NAMES

    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T

    n_leads, n_samples = signal.shape
    time = np.arange(n_samples)

    fig, axes = plt.subplots(n_leads, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    start, end = localization_bounds

    for ax, lead_name, lead_signal in zip(axes, lead_names, signal):
        ax.plot(time, lead_signal, color=line_color, linewidth=0.8)
        ax.axvspan(
            start,
            end,
            color="#1f77b4",
            alpha=0.2,
            linewidth=0,
        )
        ax.set_ylabel(lead_name, fontsize=10, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Sample", fontsize=12)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved localization overlay to {save_path}")

    return fig


def plot_lead_attention(
    cam: np.ndarray,
    signal: Optional[np.ndarray] = None,
    lead_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Per-Lead Attention",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Bar chart showing average attention per lead.
    
    If signal is provided, computes weighted attention per lead.
    Otherwise, uses uniform distribution across leads.
    
    Args:
        cam: Grad-CAM attention array of shape (T,)
        signal: Optional ECG signal of shape (12, T) for weighted attention
        lead_names: List of lead names
        save_path: Optional path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if lead_names is None:
        lead_names = LEAD_NAMES
    
    n_leads = len(lead_names)
    
    if signal is not None:
        # Ensure signal is (12, T)
        if signal.shape[0] != 12 and signal.shape[1] == 12:
            signal = signal.T
        
        # Compute per-lead attention weighted by signal magnitude
        signal_magnitude = np.abs(signal)
        weighted_attention = signal_magnitude * cam[np.newaxis, :]
        lead_attention = weighted_attention.mean(axis=1)
    else:
        # Equal attention across leads (just replicate mean CAM)
        lead_attention = np.full(n_leads, cam.mean())
    
    # Normalize
    lead_attention = lead_attention / (lead_attention.sum() + 1e-8)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn_r(lead_attention / lead_attention.max())
    bars = ax.bar(lead_names, lead_attention, color=colors, edgecolor="black", linewidth=0.5)
    
    ax.set_xlabel("Lead", fontsize=12)
    ax.set_ylabel("Relative Attention", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Add value labels on bars
    for bar, val in zip(bars, lead_attention):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    ax.set_ylim(0, lead_attention.max() * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved lead attention chart to {save_path}")
    
    return fig


def plot_ecg_with_prediction(
    signal: np.ndarray,
    prediction: float,
    true_label: Optional[int] = None,
    lead_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Plot 12-lead ECG with prediction result.
    
    Args:
        signal: ECG signal array of shape (12, T) or (T, 12)
        prediction: Model prediction probability (0-1 for MI)
        true_label: Optional ground truth (0=NORM, 1=MI)
        lead_names: List of lead names
        save_path: Optional path to save figure
        title: Optional plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if lead_names is None:
        lead_names = LEAD_NAMES
    
    # Ensure signal is (12, T)
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
    
    n_leads, n_samples = signal.shape
    time = np.arange(n_samples)
    
    # Create figure
    fig, axes = plt.subplots(n_leads, 1, figsize=figsize, sharex=True)
    
    # Build title
    pred_class = "MI" if prediction >= 0.5 else "NORM"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    
    if title is None:
        title = f"Prediction: {pred_class} ({confidence:.1%} confidence)"
        if true_label is not None:
            true_class = "MI" if true_label == 1 else "NORM"
            correct = "✓" if (prediction >= 0.5) == true_label else "✗"
            title += f" | True: {true_class} {correct}"
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Plot color based on prediction
    line_color = "#d62728" if prediction >= 0.5 else "#2ca02c"  # Red for MI, Green for NORM
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(time, signal[i], color=line_color, linewidth=0.8)
        ax.set_ylabel(lead_name, fontsize=10, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    
    axes[-1].set_xlabel("Sample", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ECG plot to {save_path}")
    
    return fig
