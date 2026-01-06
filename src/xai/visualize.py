"""
XAI Visualization Module.

Utilities to plot ECG signals with Grad-CAM overlays and explainability annotations.

Usage:
    plot_explanation(signal, gradcam_map, output_path, title)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_12lead_gradcam(
    signal: np.ndarray,
    gradcam_maps: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "ECG Explanation",
    sampling_rate: int = 100
):
    """
    Plot 12-lead ECG with Grad-CAM heatmap overlay for multiple classes.
    
    Args:
        signal: (12, 1000) ECG signal
        gradcam_maps: Dict of {class_name: heatmap_array (1000,)}
        output_path: Path to save the image
        title: Chart title
    """
    
    # Setup plot layout (6 rows x 2 cols standard or 12 rows 1 col)
    # Let's do 12 rows for clarity in MVP
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    
    # Create time axis
    timesteps = signal.shape[1]
    time = np.arange(timesteps) / sampling_rate
    
    # Determine colors for classes
    colors = ['r', 'g', 'b', 'purple']
    class_names = list(gradcam_maps.keys())
    
    for i, ax in enumerate(axes):
        # Plot signal
        ax.plot(time, signal[i], 'k', linewidth=0.8, alpha=0.8)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Overlay Heatmaps
        # We fill background or overlay line color based on activation
        for idx, cls in enumerate(class_names):
            heatmap = gradcam_maps[cls]
            if heatmap.ndim > 1:
                heatmap = heatmap.squeeze()
            
            # Normalize for visualization if not already
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)
            
            # Mask low values to avoid clutter
            mask = heatmap > 0.2
            if mask.any():
                # Overlay distinct color for each class
                # Using fill_between for emphasis
                ax.fill_between(
                    time, 
                    signal[i].min(), 
                    signal[i].max(), 
                    where=mask, 
                    color=colors[idx % len(colors)], 
                    alpha=0.3, 
                    label=cls if i == 0 else ""
                )

    axes[-1].set_xlabel("Time (s)", fontsize=14)
    if class_names:
        fig.legend(loc='upper right', fontsize=12, bbox_to_anchor=(0.95, 0.95))
        
    plt.suptitle(title, fontsize=16, y=0.92)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Explanation plot saved to {output_path}")

def plot_ecg_with_localization(
    signal: np.ndarray,
    localization_probs: Dict[str, float],
    output_path: Path,
    title: str = "MI Localization",
    sampling_rate: int = 100
):
    """
    Plot 12-lead ECG with visual markers for predicted MI regions (mockup).
    Ideally, this would highlight leads relevant to the region, but for now
    it plots the signal and adds a text box with localization probabilities.
    """
    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    timesteps = signal.shape[1]
    time = np.arange(timesteps) / sampling_rate
    
    for i, ax in enumerate(axes):
        ax.plot(time, signal[i], 'k', linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        
    axes[-1].set_xlabel("Time (s)", fontsize=14)
    
    # Add text box with probabilities
    # Filter to only include numeric values (skips "predicted_regions" list)
    text_str = "\n".join([f"{loc}: {prob:.2f}" for loc, prob in localization_probs.items() if isinstance(prob, (int, float))])
    fig.text(0.75, 0.85, text_str, fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, y=0.92)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_ecg_with_prediction(
    signal: np.ndarray,
    prediction: Dict[str, float],
    output_path: Path,
    title: str = "ECG Prediction"
):
    """
    Plot 12-lead ECG with prediction results.
    """
    # Simply reuse localization plot structure for now as it's generic enough
    plot_ecg_with_localization(signal, prediction, output_path, title)

def plot_gradcam_heatmap(
    heatmap: np.ndarray,
    output_path: Path,
    title: str = "Grad-CAM Heatmap"
):
    """
    Plot a single 1D heatmap.
    """
    if heatmap.ndim > 1:
        heatmap = heatmap.squeeze()
        
    plt.figure(figsize=(10, 2))
    plt.imshow(heatmap[np.newaxis, :], aspect="auto", cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.yticks([])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_lead_attention(
    attention_scores: np.ndarray, # (12,)
    output_path: Path,
    title: str = "Lead Attention"
):
    """
    Bar chart of lead importance.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(LEAD_NAMES, attention_scores, color='skyblue')
    plt.title(title)
    plt.ylabel("Relevance Score")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
