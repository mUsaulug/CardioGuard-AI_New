"""Extract CNN encoder features and save to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.cnn import CNNEncoder


def extract_cnn_features(
    model: CNNEncoder,
    dataloader: DataLoader,
    device: str,
    output_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run the encoder on the dataloader and save features to disk."""

    model.eval()
    device_obj = torch.device(device)
    model.to(device_obj)

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    ids: List[str] = []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, batch_labels, batch_ids = batch
            elif len(batch) == 2:
                inputs, batch_labels = batch
                batch_ids = None
            else:
                raise ValueError("Expected batch with 2 or 3 elements (inputs, labels, [ids]).")
            inputs = inputs.to(device_obj)
            embeddings = model(inputs).cpu().numpy()
            features.append(embeddings)
            if batch_labels is not None:
                labels.append(batch_labels.cpu().numpy())
            if batch_ids is not None:
                ids.extend([str(item) for item in batch_ids])

    feature_array = np.concatenate(list(features), axis=0)
    label_array = np.concatenate(labels, axis=0) if labels else None
    if label_array is not None:
        label_array = np.asarray(label_array).reshape(-1)
    ids_array = np.array(ids) if ids else None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savez_payload = {"X": feature_array}
    # Backward-compatible keys
    savez_payload["features"] = feature_array
    if label_array is not None:
        savez_payload["y"] = label_array
        savez_payload["labels"] = label_array
    if ids_array is not None:
        savez_payload["ids"] = ids_array
    np.savez_compressed(output_path, **savez_payload)
    return feature_array, label_array, ids_array


def extract_cnn_feature_splits(
    model: CNNEncoder,
    dataloaders: Dict[str, DataLoader],
    device: str,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Extract features for train/val/test splits and save .npz files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}

    for split, loader in dataloaders.items():
        output_path = output_dir / f"{split}.npz"
        extract_cnn_features(model, loader, device, output_path)
        output_paths[split] = output_path

    return output_paths
