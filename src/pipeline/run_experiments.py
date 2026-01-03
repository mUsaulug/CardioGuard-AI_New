"""Single entry point for CNN + XGBoost experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.features.extract_cnn_features import extract_cnn_features
from src.models.cnn import CNNClassifier, CNNConfig
from src.models.trainer import TrainConfig, train_cnn
from src.models.xgb import XGBConfig, train_xgb
from src.xai.gradcam import GradCAM
from src.xai.shap_xgb import explain_xgb


def build_dummy_loader(num_samples: int = 64) -> DataLoader:
    inputs = torch.randn(num_samples, 1, 64, 64)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CNN + XGBoost pipeline")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dummy_loader()
    val_loader = build_dummy_loader()

    cnn_config = CNNConfig()
    model = CNNClassifier(cnn_config)
    train_config = TrainConfig(epochs=1)
    train_metrics = train_cnn(model, train_loader, val_loader, train_config)

    feature_path = output_dir / "cnn_features.npy"
    features = extract_cnn_features(model.encoder, train_loader, "cpu", feature_path)

    labels = np.concatenate([batch[1].numpy() for batch in train_loader], axis=0)
    split = int(0.8 * len(labels))
    X_train, X_val = features[:split], features[split:]
    y_train, y_val = labels[:split], labels[split:]

    xgb_model, xgb_metrics = train_xgb(X_train, y_train, X_val, y_val, XGBConfig())

    gradcam = GradCAM(model, model.encoder.features[4])
    sample_inputs, _ = next(iter(val_loader))
    cam_output = gradcam.generate(sample_inputs[:1])
    np.save(output_dir / "gradcam.npy", cam_output)

    shap_results = explain_xgb(xgb_model, X_val)
    np.save(output_dir / "shap_values.npy", shap_results["shap_values"])

    np.save(output_dir / "train_metrics.npy", train_metrics)
    np.save(output_dir / "xgb_metrics.npy", xgb_metrics)


if __name__ == "__main__":
    main()
