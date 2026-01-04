"""Train and evaluate a single ECG experiment run."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import PTBXLConfig, get_default_config
from src.data.signals import SignalDataset, compute_channel_stats_streaming, normalize_with_stats
from src.models.cnn import ECGCNNConfig, build_multitask_cnn, build_sequential_cnn
from src.models.trainer import train_one_epoch, train_one_epoch_multitask, validate, validate_multitask
from src.pipeline.data_pipeline import prepare_splits
from src.pipeline.data_pipeline import ECGDatasetTorch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(
    config: PTBXLConfig,
    df,
    splits: Dict[str, np.ndarray],
    label_column: str,
    localization_column: str | None = None,
    stats_batch_size: int = 128,
) -> Dict[str, ECGDatasetTorch]:
    train_df = df.loc[splits["train"]]
    val_df = df.loc[splits["val"]]
    test_df = df.loc[splits["test"]]

    mean, std = compute_channel_stats_streaming(
        train_df,
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

    datasets = {
        "train": SignalDataset(
            train_df,
            config.data_root,
            filename_column=config.filename_column,
            label_column=label_column,
            localization_column=localization_column,
            transform=normalize,
            expected_channels=12,
        ),
        "val": SignalDataset(
            val_df,
            config.data_root,
            filename_column=config.filename_column,
            label_column=label_column,
            localization_column=localization_column,
            transform=normalize,
            expected_channels=12,
        ),
        "test": SignalDataset(
            test_df,
            config.data_root,
            filename_column=config.filename_column,
            label_column=label_column,
            localization_column=localization_column,
            transform=normalize,
            expected_channels=12,
        ),
    }

    return {split: ECGDatasetTorch(dataset) for split, dataset in datasets.items()}


def train_and_evaluate(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    epochs: int,
    device: torch.device,
    task: str,
    localization_weight: float = 1.0,
    multitask: bool = False,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} - training...")
        if multitask:
            train_loss = train_one_epoch_multitask(
                model,
                loaders["train"],
                optimizer,
                device,
                task=task,
                localization_weight=localization_weight,
            )
        else:
            train_loss = train_one_epoch(model, loaders["train"], optimizer, device, task=task)
        print(f"Epoch {epoch}/{epochs} - validating...")
        if multitask:
            val_loss, val_metrics = validate_multitask(
                model,
                loaders["val"],
                device,
                task=task,
                localization_weight=localization_weight,
            )
        else:
            val_loss, val_metrics = validate(model, loaders["val"], device, task=task)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(record)
        print(
            "Epoch {epoch}/{epochs} summary: "
            "train_loss={train_loss:.4f} "
            "val_loss={val_loss:.4f} "
            "roc_auc={roc_auc:.4f} "
            "pr_auc={pr_auc:.4f} "
            "f1={f1:.4f} "
            "accuracy={accuracy:.4f}".format(
                epoch=epoch,
                epochs=epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                **val_metrics,
            )
        )

    print("Evaluating on test set...")
    if multitask:
        test_loss, test_metrics = validate_multitask(
            model,
            loaders["test"],
            device,
            task=task,
            localization_weight=localization_weight,
        )
    else:
        test_loss, test_metrics = validate(model, loaders["test"], device, task=task)
    print(
        "Test summary: "
        "loss={loss:.4f} "
        "roc_auc={roc_auc:.4f} "
        "pr_auc={pr_auc:.4f} "
        "f1={f1:.4f} "
        "accuracy={accuracy:.4f}".format(
            loss=test_loss,
            **test_metrics,
        )
    )

    return {
        "history": history,
        "test": {"loss": test_loss, **test_metrics},
    }


def write_metrics(metrics: Dict[str, object], logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / "metrics.json"
    csv_path = logs_dir / "metrics.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fieldnames = ["phase", "epoch", "train_loss", "val_loss", "roc_auc", "pr_auc", "f1", "accuracy", "loss"]
    rows = []
    for record in metrics["history"]:
        rows.append(
            {
                "phase": "val",
                "epoch": record["epoch"],
                "train_loss": record["train_loss"],
                "val_loss": record["val_loss"],
                "roc_auc": record["roc_auc"],
                "pr_auc": record["pr_auc"],
                "f1": record["f1"],
                "accuracy": record["accuracy"],
                "loss": "",
            }
        )
    test_metrics = metrics["test"]
    rows.append(
        {
            "phase": "test",
            "epoch": "",
            "train_loss": "",
            "val_loss": "",
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "f1": test_metrics["f1"],
            "accuracy": test_metrics["accuracy"],
            "loss": test_metrics["loss"],
        }
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single ECG experiment")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--strategy", default="cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stats-batch-size", type=int, default=128)
    parser.add_argument(
        "--localization-column",
        type=str,
        default=None,
        help="Optional dataframe column with localization targets (e.g., start/end indices).",
    )
    parser.add_argument(
        "--localization-weight",
        type=float,
        default=1.0,
        help="Loss weight for localization head in multi-task training.",
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
    args = parser.parse_args()

    config = get_default_config()
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.sampling_rate is not None:
        config.sampling_rate = args.sampling_rate
    config.task = args.task

    if args.strategy != "cnn":
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    set_random_seed(config.random_seed)

    df, splits, label_column = prepare_splits(config)
    datasets = build_datasets(
        config,
        df,
        splits,
        label_column,
        localization_column=args.localization_column,
        stats_batch_size=args.stats_batch_size,
    )

    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"))
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        "Dataset sizes - "
        f"train: {len(datasets['train'])}, "
        f"val: {len(datasets['val'])}, "
        f"test: {len(datasets['test'])}"
    )
    model_config = ECGCNNConfig()
    num_classes = 1 if config.task == "binary" else 5
    if args.localization_column:
        model = build_multitask_cnn(model_config, num_classes=num_classes).to(device)
    else:
        model = build_sequential_cnn(model_config, num_classes=num_classes).to(device)

    metrics = train_and_evaluate(
        model,
        loaders,
        args.epochs,
        device,
        task=config.task,
        localization_weight=args.localization_weight,
        multitask=bool(args.localization_column),
    )

    logs_dir = Path("logs")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    write_metrics(metrics, logs_dir)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model_config),
        "data_config": {
            "task": config.task,
            "sampling_rate": config.sampling_rate,
            "random_seed": config.random_seed,
        },
        "metrics": metrics,
    }
    torch.save(checkpoint, checkpoints_dir / "ecgcnn.pt")


if __name__ == "__main__":
    main()
