"""
Extract CNN embeddings for multi-label superclass training.

Uses the trained multi-label CNN backbone to extract embeddings,
saving them with multi-hot labels for XGBoost OVR training.

Usage:
    python -m src.pipeline.extract_superclass_features --checkpoint checkpoints/ecgcnn_superclass.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_default_config, PTBXLConfig
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import add_superclass_labels_derived, PATHOLOGY_CLASSES
from src.data.splits import get_standard_split, verify_no_patient_leakage
from src.data.signals import SignalDataset, compute_channel_stats_streaming, normalize_with_stats
from src.pipeline.training.train_superclass_cnn import (
    MultiLabelECGCNN, 
    MultiLabelECGDataset, 
    SUPERCLASS_LABELS,
    filter_missing_files  # Import the robust filter
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_embeddings(
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from dataloader."""
    backbone.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            embeddings = backbone(signals)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Extract CNN features for multi-label XGBoost")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained superclass CNN checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("features_out"),
                        help="Output directory for features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-likelihood", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load data
    config = get_default_config()
    print("Loading data...")
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    # Use derived labels (FIX: This adds y_multi4 column which Dataset expects)
    df = add_superclass_labels_derived(df, scp_df, args.min_likelihood)
    
    # Filter missing files (CRITICAL FIX for FileNotFoundError)
    df = filter_missing_files(df, config.data_root, config.filename_column)
    
    # Get splits
    train_idx, val_idx, test_idx = get_standard_split(df)
    verify_no_patient_leakage(df, train_idx, val_idx, test_idx)
    
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]
    
    # Compute normalization stats from train
    print("Computing normalization stats...")
    mean, std = compute_channel_stats_streaming(
        train_df,
        base_path=config.data_root,
        filename_column=config.filename_column,
        batch_size=128,
        progress=True,
        expected_channels=12,
    )
    
    def normalize(signal: np.ndarray) -> np.ndarray:
        normalized = normalize_with_stats(signal, mean.flatten(), std.flatten())
        return np.transpose(normalized, (1, 0))
    
    # Build datasets
    datasets = {
        "train": MultiLabelECGDataset(train_df, config.data_root, config.filename_column, normalize),
        "val": MultiLabelECGDataset(val_df, config.data_root, config.filename_column, normalize),
        "test": MultiLabelECGDataset(test_df, config.data_root, config.filename_column, normalize),
    }
    
    loaders = {
        split: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for split, ds in datasets.items()
    }
    
    # Build model and load weights
    from src.models.cnn import ECGCNNConfig, ECGBackbone
    
    cnn_config = ECGCNNConfig()
    model = MultiLabelECGCNN(cnn_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Extract embeddings using backbone only
    backbone = model.backbone
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_paths = {}
    
    print("\nExtracting embeddings...")
    for split in ["train", "val", "test"]:
        print(f"  {split}...")
        X, y = extract_embeddings(backbone, loaders[split], device)
        
        output_path = output_dir / f"{split}_superclass.npz"
        np.savez(
            output_path,
            X=X,
            y_multi=y,  # Multi-hot labels (n_samples, 4)
            class_order=SUPERCLASS_LABELS,
        )
        output_paths[split] = output_path
        print(f"    Saved {X.shape[0]} samples with {X.shape[1]} features to {output_path}")
    
    # Save config
    config_data = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(output_dir),
        "class_order": SUPERCLASS_LABELS,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "splits": {k: str(v) for k, v in output_paths.items()},
    }
    
    config_path = output_dir / "superclass_feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\nDone! Config saved to {config_path}")


if __name__ == "__main__":
    main()
