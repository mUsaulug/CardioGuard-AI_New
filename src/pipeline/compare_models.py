"""
Compare CNN, XGBoost, and Ensemble models on the test set.

Generates a Markdown report and console output comparing metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata
from src.data.signals import SignalDataset, compute_channel_stats_streaming, normalize_with_stats
from src.data.splits import get_standard_split
from src.models.cnn import ECGCNN, ECGCNNConfig
from src.models.xgb import load_xgb, predict_xgb


# ============================================================================
# Ensemble Optimization
# ============================================================================

def optimize_ensemble_weight(
    y_true: np.ndarray,
    p_cnn: np.ndarray,
    p_xgb: np.ndarray,
    metric: str = "roc_auc",
    alpha_range: Optional[np.ndarray] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Find optimal ensemble weight α via grid search.
    
    ensemble_prob = α * p_cnn + (1-α) * p_xgb
    
    Args:
        y_true: Ground truth labels
        p_cnn: CNN predicted probabilities
        p_xgb: XGBoost predicted probabilities
        metric: Optimization metric ("roc_auc", "pr_auc", or "f1")
        alpha_range: Array of α values to try (default: 0 to 1, step 0.05)
        
    Returns:
        Tuple of (best_alpha, best_score, all_scores_dict)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    if alpha_range is None:
        alpha_range = np.linspace(0, 1, 21)  # 0.00, 0.05, ..., 1.00
    
    metric_funcs: Dict[str, Callable] = {
        "roc_auc": lambda y, p: roc_auc_score(y, p),
        "pr_auc": lambda y, p: average_precision_score(y, p),
        "f1": lambda y, p: f1_score(y, (p >= 0.5).astype(int)),
    }
    
    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}")
    
    score_func = metric_funcs[metric]
    
    best_alpha = 0.5
    best_score = 0.0
    all_scores = {}
    
    for alpha in alpha_range:
        p_ens = alpha * p_cnn + (1 - alpha) * p_xgb
        score = score_func(y_true, p_ens)
        all_scores[f"alpha_{alpha:.2f}"] = float(score)
        
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    return float(best_alpha), float(best_score), all_scores


def get_cnn_probs(
    model: ECGCNN,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CNN inference to get probabilities and true labels."""
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    probs_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
            
            # Ensure float32
            inputs = inputs.to(device_obj).float()
            # Forward pass returns logits (or whatever the head returns)
            # BinaryHead returns logits squeezed to (batch,)
            logits = model(inputs)
            
            # Apply sigmoid for binary classification
            probs = torch.sigmoid(logits).cpu().numpy()
            
            probs_list.append(probs)
            labels_list.append(labels.numpy())

    return np.concatenate(probs_list), np.concatenate(labels_list)


def get_cnn_embeddings(
    model: ECGCNN,
    dataloader: DataLoader,
    device: str
) -> np.ndarray:
    """Extract embeddings from CNN backbone for XGBoost."""
    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    embeddings_list = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch

            inputs = inputs.to(device_obj).float()
            # Use backbone directly
            features = model.backbone(inputs).cpu().numpy()
            embeddings_list.append(features)

    return np.concatenate(embeddings_list)


def build_loader(
    df: pd.DataFrame,
    indices: np.ndarray,
    config,
    batch_size: int,
    transform: Optional[callable] = None,
) -> DataLoader:
    dataset = SignalDataset(
        df=df.loc[indices],
        base_path=config.data_root,
        filename_column=config.filename_column,
        label_column="label_mi_norm",
        transform=transform,
        expected_channels=12,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid metrics JSON format: {path}")
    return payload


def _extract_test_metrics(payload: Dict[str, object], label: str) -> Dict[str, float]:
    if "test" in payload and isinstance(payload["test"], dict):
        return payload["test"]
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    raise KeyError(
        f"Unable to find test metrics for {label}. Expected 'test' or 'metrics' keys."
    )


def _format_metrics(name: str, metrics: Dict[str, float]) -> Dict[str, object]:
    return {
        "Model": name,
        "AUC": metrics.get("roc_auc"),
        "PR_AUC": metrics.get("pr_auc"),
        "F1": metrics.get("f1"),
        "Accuracy": metrics.get("accuracy"),
    }


def _save_ensemble_config(
    output_dir: Path,
    best_alpha: float,
    best_score: float,
    metric: str,
    alpha_scores: Dict[str, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_alpha": float(best_alpha),
        "best_score": float(best_score),
        "metric": metric,
        "alpha_scores": alpha_scores,
    }
    config_path = output_dir / "ensemble_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return config_path


def compare_from_metrics(
    cnn_metrics_path: Path,
    xgb_metrics_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    cnn_payload = _load_json(cnn_metrics_path)
    xgb_payload = _load_json(xgb_metrics_path)
    cnn_metrics = _extract_test_metrics(cnn_payload, "CNN")
    xgb_metrics = _extract_test_metrics(xgb_payload, "XGBoost")

    results = [
        _format_metrics("CNN", cnn_metrics),
        _format_metrics("XGBoost", xgb_metrics),
    ]
    results_df = pd.DataFrame(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"
    results_df.to_markdown(report_path, index=False, floatfmt=".4f")
    results_df.to_csv(output_dir / "comparison_report.csv", index=False)
    return results_df


def _write_markdown_table(results_df: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_markdown(report_path, index=False, floatfmt=".4f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN, XGB, and Ensemble")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compare models using existing metrics.json files instead of rerunning inference.",
    )
    parser.add_argument(
        "--cnn-metrics",
        type=Path,
        default=Path("logs/cnn/metrics.json"),
        help="Path to CNN metrics.json (default: logs/cnn/metrics.json)",
    )
    parser.add_argument(
        "--xgb-metrics",
        type=Path,
        default=Path("logs/xgb/metrics.json"),
        help="Path to XGBoost metrics.json (default: logs/xgb/metrics.json)",
    )
    parser.add_argument("--cnn-path", type=Path, default=Path("checkpoints/ecgcnn.pt"), help="Path to CNN checkpoint")
    parser.add_argument("--xgb-path", type=Path, default=Path("logs/xgb/xgb_model.json"), help="Path to XGB model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Output directory")
    args = parser.parse_args()

    if args.metrics_only:
        print("Loading metrics from JSON files...")
        results_df = compare_from_metrics(
            cnn_metrics_path=args.cnn_metrics,
            xgb_metrics_path=args.xgb_metrics,
            output_dir=args.output_dir,
        )
        print("\n=== Model Comparison Report (from metrics) ===")
        print(results_df.to_markdown(index=False, floatfmt=".4f"))
        report_path = args.output_dir / "comparison_report.md"
        print(f"\nReport saved to {report_path}")
        return

    # 1. Setup & Data Loading
    print(f"Loading data... (Device: {args.device})")
    config = get_default_config()
    df = load_ptbxl_metadata(config.metadata_path)
    
    # Load SCP statements and generate labels
    from src.data.loader import load_scp_statements
    from src.data.labels import add_binary_mi_labels
    
    print("Generating labels...")
    scp_df = load_scp_statements(config.scp_statements_path)
    df = add_binary_mi_labels(df, scp_df)
    
    # Filter valid labels and test split
    df = df[df["label_mi_norm"] != -1].copy()
    
    # Get train/val/test split
    train_indices, val_indices, test_indices = get_standard_split(df)
    
    # Intersect valid labels with splits
    valid_train_indices = np.intersect1d(train_indices, df.index)
    valid_val_indices = np.intersect1d(val_indices, df.index)
    valid_test_indices = np.intersect1d(test_indices, df.index)
    print(f"Train Set Size: {len(valid_train_indices)}")
    print(f"Validation Set Size: {len(valid_val_indices)}")
    print(f"Test Set Size: {len(valid_test_indices)}")

    stats_batch_size = 128
    mean, std = compute_channel_stats_streaming(
        df.loc[valid_train_indices],
        base_path=config.data_root,
        filename_column=config.filename_column,
        batch_size=stats_batch_size,
        progress=False,
        expected_channels=12,
    )

    def normalize(signal: np.ndarray) -> np.ndarray:
        mean_flat = mean.reshape(-1)
        std_flat = std.reshape(-1)
        normalized = normalize_with_stats(signal, mean_flat, std_flat)
        return np.transpose(normalized, (1, 0))

    val_loader = build_loader(df, valid_val_indices, config, args.batch_size, transform=normalize)
    test_loader = build_loader(df, valid_test_indices, config, args.batch_size, transform=normalize)
    
    # 2. CNN Inference
    print("Loading CNN model...")
    cnn_config = ECGCNNConfig()
    cnn_model = ECGCNN(cnn_config)
    
    if args.cnn_path.exists():
        checkpoint = torch.load(args.cnn_path, map_location=args.device)
        # Handle checkpoint dict if present
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        # Remap keys if needed (0. -> backbone., 1. -> head.)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("0."):
                new_state_dict[k.replace("0.", "backbone.", 1)] = v
            elif k.startswith("1."):
                new_state_dict[k.replace("1.", "head.", 1)] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
            
        cnn_model.load_state_dict(state_dict)
    else:
        print(f"Warning: CNN checkpoint not found at {args.cnn_path}")
        sys.exit(1)
        
    print("Running CNN inference...")
    cnn_probs_val, y_val = get_cnn_probs(cnn_model, val_loader, args.device)
    cnn_probs_test, y_test = get_cnn_probs(cnn_model, test_loader, args.device)
    
    # 3. XGB Inference
    print("Loading XGB model...")
    if not args.xgb_path.exists():
        print(f"Error: XGB model not found at {args.xgb_path}")
        sys.exit(1)
        
    xgb_model = load_xgb(args.xgb_path)
    
    print("Extracting features for XGB...")
    xgb_features_val = get_cnn_embeddings(cnn_model, val_loader, args.device)
    xgb_features_test = get_cnn_embeddings(cnn_model, test_loader, args.device)

    print("Running XGB inference...")
    xgb_probs_val, _ = predict_xgb(xgb_model, xgb_features_val)
    xgb_probs_test, _ = predict_xgb(xgb_model, xgb_features_test)

    # 4. Ensemble with Optimized Alpha (validation)
    print("\nOptimizing ensemble weight α (validation)...")
    metric_name = "roc_auc"
    best_alpha, best_score, alpha_scores = optimize_ensemble_weight(
        y_val, cnn_probs_val, xgb_probs_val, metric=metric_name
    )
    print(f"Best α = {best_alpha:.2f} (AUC = {best_score:.4f})")
    config_path = _save_ensemble_config(
        args.output_dir,
        best_alpha,
        best_score,
        metric_name,
        alpha_scores,
    )
    print(f"Saved ensemble config to {config_path}")
    
    # Compute ensemble with optimal α on test
    ensemble_probs_opt = best_alpha * cnn_probs_test + (1 - best_alpha) * xgb_probs_test
    # Also keep naive average for comparison
    ensemble_probs_avg = (cnn_probs_test + xgb_probs_test) / 2
    
    # 5. Metrics
    print("\nCalculating metrics...")
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
    
    def calc_metrics(y_t, y_p, name):
        y_pred = (y_p >= 0.5).astype(int)
        return {
            "Model": name,
            "AUC": roc_auc_score(y_t, y_p),
            "PR_AUC": average_precision_score(y_t, y_p),
            "F1": f1_score(y_t, y_pred),
            "Accuracy": accuracy_score(y_t, y_pred)
        }

    results = []
    results.append(calc_metrics(y_test, cnn_probs_test, "CNN"))
    results.append(calc_metrics(y_test, xgb_probs_test, "XGBoost"))
    results.append(calc_metrics(y_test, ensemble_probs_avg, "Ensemble (α=0.5)"))
    results.append(calc_metrics(y_test, ensemble_probs_opt, f"Ensemble (α={best_alpha:.2f})"))
    
    results_df = pd.DataFrame(results)
    
    # 6. Output
    print("\n=== Model Comparison Report ===")
    print(results_df.to_string(index=False))
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison_report.md"
    _write_markdown_table(results_df, report_path)
    print(f"\nReport saved to {report_path}")
    
    # Save CSV too
    results_df.to_csv(args.output_dir / "comparison_report.csv", index=False)


if __name__ == "__main__":
    main()
