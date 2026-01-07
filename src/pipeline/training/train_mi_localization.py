"""
Train MI Localization Model (5 Anatomical Regions).

Trains a Multi-label CNN to localize MI into 5 regions:
[AMI, ASMI, ALMI, IMI, LMI]

Training Strategy:
- Filter dataset to include ONLY MI-positive samples.
- Use Binary Cross Entropy Loss (BCEWithLogitsLoss).
- Validate on Fold-9 (MI-positive subset).

Usage:
    python -m src.pipeline.train_mi_localization --epochs 30 --batch-size 64
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import logging

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import add_superclass_labels_derived
from src.data.mi_localization import (
    add_mi_localization_labels, 
    MI_LOCALIZATION_REGIONS, 
    NUM_MI_REGIONS
)
from src.data.splits import get_standard_split
from src.models.cnn import ECGCNNConfig, ECGCNN
from src.pipeline.training.train_superclass_cnn import MultiLabelECGDataset

class MILocalizationDataset(torch.utils.data.Dataset):
    def __init__(self, df, base_path, filename_col):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.filename_col = filename_col
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        import wfdb
        row = self.df.iloc[idx]
        path = self.base_path / row[self.filename_col]
        
        # Load signal
        try:
            record = wfdb.rdrecord(str(path))
            signal = record.p_signal
        except Exception as e:
            # Fallback for corrupted/missing files (though we filtered them)
            # Return zeros
            print(f"Error loading {path}: {e}")
            signal = np.zeros((1000, 12), dtype=np.float32)
            
        # Ensure channel first (C, T)
        signal = ensure_channel_first(signal)
        
        # Label: y_loc column (list or array)
        label = np.array(row["y_loc"], dtype=np.float32)
        
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    """Ensure signal is (channels, timesteps) format."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    # Heuristic: 12-lead ECG, if first dim is 12, it's already channel-first
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    
    # Default: assume (timesteps, channels) and transpose
    if signal.shape[0] > signal.shape[1]:
        return signal.T
    
    return signal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for signals, labels in tqdm(loader, desc="Training"):
        signals = signals.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * signals.size(0)
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Validation"):
            signals = signals.to(device)
            labels = labels.to(device)
            
            logits = model(signals)
            loss = criterion(logits, labels)
            total_loss += loss.item() * signals.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    metrics = {}
    try:
        metrics["macro_auroc"] = roc_auc_score(all_targets, all_probs, average="macro")
    except:
        metrics["macro_auroc"] = 0.0
        
    try:
        metrics["macro_auprc"] = average_precision_score(all_targets, all_probs, average="macro")
    except:
        metrics["macro_auprc"] = 0.0
        
    # Per class metrics
    for i, cls in enumerate(MI_LOCALIZATION_REGIONS):
        try:
            metrics[f"auprc_{cls}"] = average_precision_score(all_targets[:, i], all_probs[:, i])
        except:
             metrics[f"auprc_{cls}"] = 0.0
             
    return total_loss / len(loader.dataset), metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Data
    config = get_default_config()
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    
    # 1. Add Superclass Labels (to find MI)
    df = add_superclass_labels_derived(df, scp_df)
    
    # 2. Add Localization Labels
    df = add_mi_localization_labels(df)
    
    # 3. Filter for MI-positive samples WITH localization
    # Plan v1.0 Item 1.4: Option A (Drop unknown/excluded localizations)
    # We keep only rows where label_MI=1 AND has_mi_localization=1
    initial_len = len(df)
    df_mi = df[
        (df["label_MI"] == 1) & 
        (df["has_mi_localization"] == 1)
    ].copy().reset_index(drop=True)
    
    logger.info(f"Filtered dataset matches (MI=1 & Loc!=0): {len(df_mi)} / {initial_len}")
    
    if len(df_mi) == 0:
        raise ValueError("No samples found with MI localization!")

    # Split (strat_fold 1-8 train, 9 val)
    train_idx = df_mi[df_mi["strat_fold"].isin(range(1, 9))].index
    val_idx = df_mi[df_mi["strat_fold"] == 9].index
    
    logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    
    # We need y_loc for pos_weight calculation
    y_loc = np.array(df_mi["y_loc"].tolist())
    
    # Datasets
    # Use config.filename_column (usually filename_lr or filename_hr)
    train_ds = MILocalizationDataset(df_mi.iloc[train_idx], config.data_root, config.filename_column)
    val_ds = MILocalizationDataset(df_mi.iloc[val_idx], config.data_root, config.filename_column)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for safe Windows exe
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model_config = ECGCNNConfig(num_filters=64, dropout=0.5) # Slightly higher dropout for smaller dataset
    model = ECGCNN(model_config, num_classes=NUM_MI_REGIONS)
    model.to(device)
    
    # Compute pos_weight based on Training set (Plan Item 5.1)
    # pos_weight = negative / positive
    y_train = y_loc[train_idx]
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weights_tensor = torch.tensor(
        np.where(pos_counts > 0, neg_counts / pos_counts, 1.0),
        dtype=torch.float32
    ).to(device)
    
    logger.info(f"Using pos_weights: {pos_weights_tensor.cpu().numpy()}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Train Loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro AUPRC: {metrics['macro_auprc']:.4f}"
        )
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'metrics': metrics
            }, args.output_dir / "ecgcnn_localization.pt")
            logger.info("Saved best model.")

if __name__ == "__main__":
    main()
