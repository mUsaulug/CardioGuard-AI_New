"""Extract CNN encoder features and save to disk."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.cnn import CNNEncoder


def extract_cnn_features(
    model: CNNEncoder,
    dataloader: DataLoader,
    device: str,
    output_path: str | Path,
) -> np.ndarray:
    """Run the encoder on the dataloader and save features to disk."""

    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    features: List[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, _ = batch
            inputs = inputs.to(device_obj)
            embeddings = model(inputs).cpu().numpy()
            features.append(embeddings)

    feature_array = np.concatenate(list(features), axis=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature_array)
    return feature_array
