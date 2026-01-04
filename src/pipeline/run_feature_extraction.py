"""Extract CNN embeddings from a trained checkpoint for XGBoost training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import get_default_config
from src.features.extract_cnn_features import extract_cnn_feature_splits
from src.models.cnn import ECGCNN, ECGCNNConfig
from src.pipeline.data_pipeline import ECGDatasetTorch, build_datasets
from src.utils.checkpoints import load_checkpoint_state_dict


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    config,
    batch_size: int,
    num_workers: int,
    cache_dir: Path | None,
) -> Dict[str, DataLoader]:
    datasets, _, _ = build_datasets(config, cache_dir=cache_dir)
    torch_datasets = {split: ECGDatasetTorch(dataset) for split, dataset in datasets.items()}
    return {
        split: DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, torch_dataset in torch_datasets.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNN embeddings from a checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained CNN checkpoint (e.g., checkpoints/ecgcnn.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("features_out"),
        help="Directory to store extracted features (train/val/test npz)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for preprocessed signals",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override PTB-XL root directory (defaults to ./physionet.org/files/ptb-xl/1.0.3).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        choices=[100, 500],
        default=None,
        help="Override sampling rate for records100/records500.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    args = parser.parse_args()

    config = get_default_config()
    config.task = "binary"
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.sampling_rate is not None:
        config.sampling_rate = args.sampling_rate
    set_random_seed(config.random_seed)

    dataloaders = build_dataloaders(config, args.batch_size, args.num_workers, args.cache_dir)

    device = torch.device(args.device)
    model = ECGCNN(ECGCNNConfig())
    state_dict = load_checkpoint_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Extracting embeddings on device: {device}")
    output_paths = extract_cnn_feature_splits(
        model.backbone,
        dataloaders,
        device=str(device),
        output_dir=args.output_dir,
    )

    config_payload = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "random_seed": config.random_seed,
        "data_root": str(config.data_root),
        "sampling_rate": config.sampling_rate,
        "label_task": config.task,
        "splits": {split: str(path) for split, path in output_paths.items()},
    }
    config_path = args.output_dir / "feature_extraction_config.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    print("Saved feature files:")
    for split, path in output_paths.items():
        print(f" - {split}: {path}")
    print(f"Saved config snapshot to {config_path}")


if __name__ == "__main__":
    main()
