"""
Generate Validation Predictions for Ensemble Optimization.

This script runs inference on the validation set using both CNN and XGBoost models
and saves the per-sample probabilities to .npz files. These files are required
for threshold optimization.

Usage:
    python -m src.pipeline.generate_validation_predictions
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

from src.config import get_default_config, PTBXLConfig
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import add_superclass_labels_derived
from src.data.splits import get_standard_split, verify_no_patient_leakage
from src.data.signals import SignalDataset, normalize_with_stats
from src.pipeline.train_superclass_cnn import (
    MultiLabelECGCNN, 
    MultiLabelECGDataset, 
    SUPERCLASS_LABELS,
    filter_missing_files
)
from src.pipeline.train_superclass_xgb_ovr import load_xgb_ovr_models
from src.models.cnn import ECGCNNConfig


def main():
    parser = argparse.ArgumentParser(description="Generate Predictions for Validation Set")
    parser.add_argument("--cnn-checkpoint", type=Path, default=Path("checkpoints/ecgcnn_superclass.pt"))
    parser.add_argument("--xgb-dir", type=Path, default=Path("logs/xgb_superclass"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading validation data...")
    config = get_default_config()
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    df = add_superclass_labels_derived(df, scp_df)
    df = filter_missing_files(df, config.data_root, config.filename_column)
    
    train_idx, val_idx, test_idx = get_standard_split(df)
    val_df = df.loc[val_idx]
    
    # Normalization (load from feature config if possible, else recompute/load)
    # We will assume feature config exists from extraction step
    feature_config_path = Path("features_out/superclass_feature_config.json")
    if feature_config_path.exists():
        with open(feature_config_path) as f:
            feat_conf = json.load(f)
            mean = np.array(feat_conf["normalization_mean"])
            std = np.array(feat_conf["normalization_std"])
    else:
        print("Warning: Normalization stats not found. Using dummy stats (Not Recommended).")
        mean = np.zeros(12)
        std = np.ones(12)

    def normalize(signal: np.ndarray) -> np.ndarray:
        normalized = normalize_with_stats(signal, mean.flatten(), std.flatten())
        return np.transpose(normalized, (1, 0))
        
    val_dataset = MultiLabelECGDataset(val_df, config.data_root, config.filename_column, normalize)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Load Models
    print("Loading models...")
    # CNN
    cnn_config = ECGCNNConfig()
    cnn_model = MultiLabelECGCNN(cnn_config).to(device)
    checkpoint = torch.load(args.cnn_checkpoint, map_location=device)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.eval()
    
    # XGB
    xgb_ensemble = load_xgb_ovr_models(args.xgb_dir)
    
    # 3. Inference
    print("Running inference...")
    cnn_probs_list = []
    xgb_probs_list = []
    labels_list = []
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            labels_list.append(labels.numpy())
            
            # CNN Probs (Head)
            logits = cnn_model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            cnn_probs_list.append(probs)
            
            # CNN Embeddings (Backbone)
            embeddings = cnn_model.backbone(signals).cpu().numpy()
            
            # XGB Probs (Calibrated OVR)
            # Use the predict_proba_array method we verified/added in train_superclass_xgb_ovr
            # If it's not exposed, we use the class method
            x_probs = xgb_ensemble.predict_proba_array(embeddings)
            xgb_probs_list.append(x_probs)
            
    # Concatenate
    cnn_all = np.concatenate(cnn_probs_list, axis=0)
    xgb_all = np.concatenate(xgb_probs_list, axis=0)
    y_all = np.concatenate(labels_list, axis=0)
    
    # 4. Save
    print("Saving predictions...")
    
    # Save CNN probs
    cnn_out = {cls: cnn_all[:, i] for i, cls in enumerate(SUPERCLASS_LABELS)}
    np.savez(args.output_dir / "val_cnn_probs.npz", **cnn_out)
    
    # Save XGB probs
    xgb_out = {cls: xgb_all[:, i] for i, cls in enumerate(SUPERCLASS_LABELS)}
    np.savez(args.output_dir / "val_xgb_probs.npz", **xgb_out)
    
    # Save labels
    np.savez(args.output_dir / "val_labels.npz", y_multi=y_all)
    
    print(f"Done. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
