"""
CardioGuard-AI data pipeline helpers.

Builds label-aware splits and datasets with consistent normalization stats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from src.config import DIAGNOSTIC_SUPERCLASSES, PTBXLConfig
from src.data.labels import add_5class_labels, add_binary_mi_labels, filter_valid_samples
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.signals import (
    SignalDataset,
    compute_channel_stats_streaming,
    normalize_with_stats,
)
from src.data.splits import get_split_from_config, verify_no_patient_leakage


def prepare_splits(config: PTBXLConfig) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """
    Load metadata, assign labels, build splits, and check for leakage.

    Returns:
        Tuple of (df, splits, label_column)
        splits is a dict with "train", "val", and "test" index arrays.
    """
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)

    if config.task == "binary":
        df = add_binary_mi_labels(df, scp_df, min_likelihood=config.min_likelihood)
        label_column = "label_mi_norm"
    elif config.task == "multiclass":
        df = add_5class_labels(
            df,
            scp_df,
            min_likelihood=config.min_likelihood,
            multi_label=False,
        )
        label_column = "label_5class"
    else:
        raise ValueError(f"Unsupported task type: {config.task}")

    df = filter_valid_samples(df, label_column=label_column)
    df = _filter_existing_records(
        df,
        config.data_root,
        config.filename_column,
        expected_channels=12,
    )

    train_indices, val_indices, test_indices = get_split_from_config(df, config)
    verify_no_patient_leakage(df, train_indices, val_indices, test_indices)

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    return df, splits, label_column


def _filter_existing_records(
    df: pd.DataFrame,
    base_path: Path,
    filename_column: str,
    expected_channels: int,
) -> pd.DataFrame:
    base_path = Path(base_path)

    def record_exists(filename: str) -> bool:
        record_path = base_path / filename
        return record_path.with_suffix(".hea").exists() and record_path.with_suffix(".dat").exists()

    mask = df[filename_column].apply(record_exists)
    missing_count = int((~mask).sum())
    if missing_count:
        print(f"Skipping {missing_count} records with missing .hea/.dat files.")
    filtered = df[mask].copy()

    channel_mismatch = 0
    header_errors = 0
    keep_mask = []
    for filename in filtered[filename_column]:
        record_path = base_path / filename
        try:
            header = wfdb.rdheader(str(record_path))
        except Exception as exc:
            header_errors += 1
            print(f"Skipping {filename}: header read failed ({exc})")
            keep_mask.append(False)
            continue

        n_sig = getattr(header, "n_sig", None)
        if n_sig != expected_channels:
            channel_mismatch += 1
            keep_mask.append(False)
            continue
        keep_mask.append(True)

    if header_errors:
        print(f"Skipping {header_errors} records with unreadable headers.")
    if channel_mismatch:
        print(f"Skipping {channel_mismatch} records with unexpected lead counts.")

    return filtered[keep_mask].copy()


def _label_mapping_for_task(task: str) -> Dict[int, str]:
    if task == "binary":
        return {0: "NORM", 1: "MI"}
    if task == "multiclass":
        return {i: name for i, name in enumerate(DIAGNOSTIC_SUPERCLASSES)}
    raise ValueError(f"Unsupported task type: {task}")


def build_datasets(
    config: PTBXLConfig,
    cache_dir: Optional[Path] = None,
    stats_batch_size: int = 128,
) -> Tuple[Dict[str, SignalDataset], Dict[int, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Build SignalDataset objects with train-derived normalization stats.

    Returns:
        Tuple of (datasets, label_mapping, (mean, std))
    """
    df, splits, label_column = prepare_splits(config)

    train_df = df.loc[splits["train"]]
    val_df = df.loc[splits["val"]]
    test_df = df.loc[splits["test"]]

    mean, std = compute_channel_stats_streaming(
        train_df,
        base_path=config.data_root,
        filename_column=config.filename_column,
        batch_size=stats_batch_size,
        progress=False,
        cache_dir=cache_dir,
    )

    def normalize(signal: np.ndarray) -> np.ndarray:
        return normalize_with_stats(signal, mean, std)

    datasets = {
        "train": SignalDataset(
            train_df,
            config.data_root,
            filename_column=config.filename_column,
            label_column=label_column,
            transform=normalize,
            cache_dir=cache_dir,
        ),
        "val": SignalDataset(
            val_df,
            config.data_root,
            filename_column=config.filename_column,
            label_column=label_column,
            transform=normalize,
            cache_dir=cache_dir,
        ),
        "test": SignalDataset(
            test_df,
            config.data_root,
            filename_column=config.filename_column,
            label_column=label_column,
            transform=normalize,
            cache_dir=cache_dir,
        ),
    }

    label_mapping = _label_mapping_for_task(config.task)

    return datasets, label_mapping, (mean, std)


class ECGDatasetTorch(Dataset):
    """
    Torch Dataset wrapper to convert numpy signals/labels to torch tensors.
    """

    def __init__(
        self,
        dataset: SignalDataset,
        signal_dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.long,
    ) -> None:
        self.dataset = dataset
        self.signal_dtype = signal_dtype
        self.label_dtype = label_dtype

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        signal, label, ecg_id = self.dataset[idx]
        signal_tensor = torch.as_tensor(signal, dtype=self.signal_dtype)
        if label is None:
            label_tensor = None
        else:
            label_tensor = torch.tensor(label, dtype=self.label_dtype)
        return signal_tensor, label_tensor, ecg_id
