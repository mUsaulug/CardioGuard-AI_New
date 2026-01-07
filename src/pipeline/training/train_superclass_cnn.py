"""
Multi-label Superclass CNN Training Script.

Trains a 4-output CNN (MI, STTC, CD, HYP) with BCEWithLogitsLoss.
NORM is derived as: NORM = 1 iff all pathology predictions < threshold.

Usage:
    python -m src.pipeline.train_superclass_cnn --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import get_default_config, PTBXLConfig, DIAGNOSTIC_SUPERCLASSES
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import (
    add_superclass_labels_derived,
    extract_y_multi4,
    compute_pos_weight_train,
    PATHOLOGY_CLASSES,
)
from src.data.splits import get_standard_split, verify_no_patient_leakage
from src.data.signals import SignalDataset, compute_channel_stats_streaming, normalize_with_stats
from src.models.cnn import ECGCNNConfig, ECGBackbone, MultiClassHead
from src.models.metrics import compute_multilabel_metrics


# Multi-label class order (NORM is derived, not trained)
SUPERCLASS_LABELS = ["MI", "STTC", "CD", "HYP"]
NUM_CLASSES = len(SUPERCLASS_LABELS)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiLabelECGCNN(nn.Module):
    """4-output multi-label CNN for superclass prediction."""
    
    def __init__(self, config: ECGCNNConfig):
        super().__init__()
        self.backbone = ECGBackbone(config)
        self.head = nn.Linear(config.num_filters, NUM_CLASSES)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning 4 logits (MI, STTC, CD, HYP)."""
        embeddings = self.backbone(x)
        return self.head(embeddings)


class MultiLabelECGDataset(Dataset):
    """Dataset that returns multi-hot labels for superclass."""
    
    def __init__(
        self,
        df,
        base_path: Path,
        filename_column: str,
        transform=None,
        class_order: List[str] = None,
    ):
        self.df = df.reset_index()
        self.base_path = Path(base_path)
        self.filename_column = filename_column
        self.transform = transform
        self.class_order = class_order or SUPERCLASS_LABELS
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        import wfdb
        
        row = self.df.iloc[idx]
        record_path = self.base_path / row[self.filename_column]
        
        # Load signal
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal  # (timesteps, channels)
        
        if self.transform:
            signal = self.transform(signal)
        
        # Create multi-hot label (4 classes) - use pre-computed y_multi4 from df
        if "y_multi4" in row.index:
            label = np.array(row["y_multi4"], dtype=np.float32)
        else:
            # Fallback: extract from superclass_pathologies
            pathologies = row.get("superclass_pathologies", [])
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            for i, cls in enumerate(self.class_order):
                if cls in pathologies:
                    label[i] = 1.0
        
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32)
        label_tensor = torch.as_tensor(label, dtype=torch.float32)
        
        return signal_tensor, label_tensor


def compute_pos_weight(train_df, y_train: np.ndarray) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss from training data only.
    Uses the new labels_superclass module.
    """
    pos_weight = compute_pos_weight_train(y_train)
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    running_loss = 0.0
    
    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * signals.size(0)
    
    return running_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Validate model, return loss and metrics."""
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            
            logits = model(signals)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * signals.size(0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader.dataset)
    
    # Compute multi-label metrics
    y_true = np.concatenate(all_labels, axis=0)
    y_logits = np.concatenate(all_logits, axis=0)
    metrics = compute_multilabel_metrics(y_true, y_logits, class_names=SUPERCLASS_LABELS)
    
    return avg_loss, metrics



def filter_missing_files(df: pd.DataFrame, data_root: Path, filename_col: str) -> pd.DataFrame:
    """Filter out records where the physical files are missing."""
    valid_indices = []
    missing_count = 0
    
    print(f"Checking file existence for {len(df)} records...")
    for idx, row in df.iterrows():
        # PTB-XL records have .hea and .dat files
        base_name = data_root / row[filename_col]
        header_path = base_name.with_suffix(".hea")
        dat_path = base_name.with_suffix(".dat")
        
        if header_path.exists() and dat_path.exists():
            valid_indices.append(idx)
        else:
            missing_count += 1
            if missing_count <= 5:  # Log first 5 missing
                print(f"  Warning: Missing file for ecg_id={idx}: {header_path}")
    
    if missing_count > 0:
        print(f"Dropped {missing_count} records due to missing files.")
    
    return df.loc[valid_indices].copy()


def build_datasets(config: PTBXLConfig, min_likelihood: float = 0.0):
    """Build train/val/test datasets with multi-label targets (derived NORM)."""
    # Load data
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    
    # Filter missing files (CRITICAL FIX)
    df = filter_missing_files(df, config.data_root, config.filename_column)
    
    # Add derived superclass labels (NORM is derived, not from data)
    df = add_superclass_labels_derived(df, scp_df, min_likelihood)
    
    # Get splits
    train_idx, val_idx, test_idx = get_standard_split(df)
    verify_no_patient_leakage(df, train_idx, val_idx, test_idx)
    
    train_df = df.loc[train_idx]
    val_df = df.loc[val_idx]
    test_df = df.loc[test_idx]
    
    # Extract y_multi4 for pos_weight calculation
    y_train = extract_y_multi4(train_df, scp_df, min_likelihood)
    
    # Compute normalization stats from train only
    print("Computing normalization stats from training data...")
    mean, std = compute_channel_stats_streaming(
        train_df,
        base_path=config.data_root,
        filename_column=config.filename_column,
        batch_size=128,
        progress=True,
        expected_channels=12,
    )
    
    def normalize(signal: np.ndarray) -> np.ndarray:
        """Normalize and transpose to (channels, timesteps)."""
        normalized = normalize_with_stats(signal, mean.flatten(), std.flatten())
        return np.transpose(normalized, (1, 0))
    
    # Build datasets
    datasets = {
        "train": MultiLabelECGDataset(
            train_df, config.data_root, config.filename_column, transform=normalize
        ),
        "val": MultiLabelECGDataset(
            val_df, config.data_root, config.filename_column, transform=normalize
        ),
        "test": MultiLabelECGDataset(
            test_df, config.data_root, config.filename_column, transform=normalize
        ),
    }
    
    # Compute pos_weight from train only (using y_train from earlier)
    pos_weight = compute_pos_weight(train_df, y_train)
    
    return datasets, pos_weight, (mean, std), {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Multi-label Superclass CNN")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-likelihood", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="logs/superclass_cnn")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--debug", action="store_true", help="Quick test with 2 epochs")
    
    args = parser.parse_args()
    
    if args.debug:
        args.epochs = 2
    
    set_seed(args.seed)
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    config = get_default_config()
    print("Building datasets...")
    datasets, pos_weight, norm_stats, dfs = build_datasets(config, args.min_likelihood)
    
    print(f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    print(f"pos_weight: {pos_weight.tolist()}")
    
    # Save normalization stats
    np.savez(
        output_dir / "normalization_stats.npz",
        mean=norm_stats[0],
        std=norm_stats[1],
    )
    
    # DataLoaders
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    
    # Model
    cnn_config = ECGCNNConfig()
    model = MultiLabelECGCNN(cnn_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss with pos_weight
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    
    # Training log
    history = []
    best_val_auroc = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics = validate(model, loaders["val"], criterion, device)
        
        # LR scheduler
        scheduler.step(val_metrics["macro_auroc"])
        
        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auroc": val_metrics["macro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        # Add per-class metrics
        for cls, cls_metrics in val_metrics["per_class"].items():
            log_entry[f"val_{cls}_auroc"] = cls_metrics["auroc"]
            log_entry[f"val_{cls}_auprc"] = cls_metrics["auprc"]
        
        history.append(log_entry)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUROC: {val_metrics['macro_auroc']:.4f} | "
              f"Val AUPRC: {val_metrics['macro_auprc']:.4f}")
        
        # Save best model
        if val_metrics["macro_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["macro_auroc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(cnn_config),
                "val_metrics": val_metrics,
                "pos_weight": pos_weight.cpu().tolist(),
                "class_order": SUPERCLASS_LABELS,
            }, checkpoint_dir / "ecgcnn_superclass.pt")
            print(f"  ✓ New best model saved (AUROC: {best_val_auroc:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete. Best epoch: {best_epoch} (Val AUROC: {best_val_auroc:.4f})")
    print("=" * 60)
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    # Load best model
    checkpoint = torch.load(checkpoint_dir / "ecgcnn_superclass.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_metrics = validate(model, loaders["test"], criterion, device)
    
    print("\nTest Results:")
    print(f"  Loss:       {test_loss:.4f}")
    print(f"  Macro AUROC: {test_metrics['macro_auroc']:.4f}")
    print(f"  Macro AUPRC: {test_metrics['macro_auprc']:.4f}")
    print(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
    print("\nPer-class metrics:")
    for cls, cls_metrics in test_metrics["per_class"].items():
        print(f"  {cls:6}: AUROC={cls_metrics['auroc']:.4f}, "
              f"AUPRC={cls_metrics['auprc']:.4f}, "
              f"F1={cls_metrics['f1']:.4f}, "
              f"support={cls_metrics['support']}")
    
    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "test_metrics": test_metrics,
        "args": vars(args),
        "pos_weight": pos_weight.cpu().tolist(),
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save training history
    with open(output_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Checkpoint saved to {checkpoint_dir / 'ecgcnn_superclass.pt'}")


if __name__ == "__main__":
    main()
