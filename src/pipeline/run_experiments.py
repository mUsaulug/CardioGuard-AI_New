"""Dummy demo entry point for the 1D CNN + XGBoost pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.features.extract_cnn_features import extract_cnn_features
from src.models.cnn import CNNEncoder, ECGCNN, ECGCNNConfig
from src.models.trainer import train_one_epoch, validate
from src.models.xgb import XGBConfig, train_xgb
from src.xai.shap_xgb import explain_xgb


def build_dummy_loader(
    num_samples: int = 64,
    channels: int = 12,
    timesteps: int = 500,
) -> DataLoader:
    """Create dummy ECG batches shaped as (batch, channels, timesteps)."""

    inputs = torch.randn(num_samples, channels, timesteps)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1D CNN + XGBoost dummy demo")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cnn_config = ECGCNNConfig()
    train_loader = build_dummy_loader(channels=cnn_config.in_channels)
    val_loader = build_dummy_loader(channels=cnn_config.in_channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGCNN(cnn_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_metrics = validate(model, val_loader, device)
    train_metrics = {"train_loss": train_loss, "val_loss": val_loss, **val_metrics}

    encoder = CNNEncoder(cnn_config)
    encoder.features.load_state_dict(model.features.state_dict())
    feature_path = output_dir / "cnn_features.npz"
    features, _, _ = extract_cnn_features(encoder, train_loader, "cpu", feature_path)

    labels = np.concatenate([batch[1].numpy() for batch in train_loader], axis=0)
    split = int(0.8 * len(labels))
    X_train, X_val = features[:split], features[split:]
    y_train, y_val = labels[:split], labels[split:]

    xgb_model, xgb_metrics = train_xgb(X_train, y_train, X_val, y_val, XGBConfig())

    shap_results = explain_xgb(xgb_model, X_val)
    np.save(output_dir / "shap_values.npy", shap_results["shap_values"])

    np.save(output_dir / "train_metrics.npy", train_metrics)
    np.save(output_dir / "xgb_metrics.npy", xgb_metrics)


if __name__ == "__main__":
    main()
