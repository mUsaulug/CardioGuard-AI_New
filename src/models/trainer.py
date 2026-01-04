"""Training utilities for ECG CNN models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.metrics import compute_classification_metrics


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 4:
            inputs, labels, localization, *_ = batch
            return inputs, labels, localization
        if len(batch) == 3:
            inputs, labels, *_ = batch
            return inputs, labels, None
        if len(batch) == 2:
            inputs, labels = batch
            return inputs, labels, None
    raise ValueError("Unsupported batch structure for training.")


def _extract_logits(output: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "logits" not in output:
            raise KeyError("Multi-task output missing 'logits' key.")
        return output["logits"]
    return output


def _extract_localization(output: torch.Tensor | Dict[str, torch.Tensor]) -> torch.Tensor | None:
    if isinstance(output, dict):
        return output.get("localization")
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str = "binary",
) -> float:
    """Train for one epoch and return average loss."""

    model.train()
    criterion: nn.Module
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task type: {task}")
    running_loss = 0.0

    for batch in loader:
        inputs, labels, _ = _unpack_batch(batch)
        inputs = inputs.to(device)
        if task == "binary":
            labels = labels.to(device).float()
        else:
            labels = labels.to(device).long()

        optimizer.zero_grad()
        output = model(inputs)
        logits = _extract_logits(output)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def train_one_epoch_multitask(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str = "binary",
    localization_weight: float = 1.0,
) -> float:
    """Train for one epoch with classification + localization losses."""

    model.train()
    criterion: nn.Module
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task type: {task}")
    localization_criterion = nn.SmoothL1Loss()
    running_loss = 0.0

    for batch in loader:
        inputs, labels, localization = _unpack_batch(batch)
        inputs = inputs.to(device)
        if task == "binary":
            labels = labels.to(device).float()
        else:
            labels = labels.to(device).long()

        optimizer.zero_grad()
        output = model(inputs)
        logits = _extract_logits(output)
        loss = criterion(logits, labels)

        localization_pred = _extract_localization(output)
        if localization_pred is not None and localization is not None:
            localization = localization.to(device).float()
            loss = loss + localization_weight * localization_criterion(localization_pred, localization)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str = "binary",
) -> Tuple[float, Dict[str, float]]:
    """Validate model and return average loss and metrics."""

    model.eval()
    criterion: nn.Module
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task type: {task}")
    running_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels, _ = _unpack_batch(batch)
            inputs = inputs.to(device)
            if task == "binary":
                labels = labels.to(device).float()
            else:
                labels = labels.to(device).long()

            output = model(inputs)
            logits = _extract_logits(output)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )
    return avg_loss, metrics


def validate_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str = "binary",
    localization_weight: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Validate multi-task model and return average loss and metrics."""

    model.eval()
    criterion: nn.Module
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported task type: {task}")
    localization_criterion = nn.SmoothL1Loss()
    running_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs, labels, localization = _unpack_batch(batch)
            inputs = inputs.to(device)
            if task == "binary":
                labels = labels.to(device).float()
            else:
                labels = labels.to(device).long()

            output = model(inputs)
            logits = _extract_logits(output)
            loss = criterion(logits, labels)

            localization_pred = _extract_localization(output)
            if localization_pred is not None and localization is not None:
                localization = localization.to(device).float()
                loss = loss + localization_weight * localization_criterion(localization_pred, localization)

            running_loss += loss.item() * inputs.size(0)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )
    return avg_loss, metrics
