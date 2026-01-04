"""Training utilities for ECG CNN models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.metrics import compute_classification_metrics


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

    for inputs, labels, *_ in loader:
        inputs = inputs.to(device)
        if task == "binary":
            labels = labels.to(device).float()
        else:
            labels = labels.to(device).long()

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
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
        for inputs, labels, *_ in loader:
            inputs = inputs.to(device)
            if task == "binary":
                labels = labels.to(device).float()
            else:
                labels = labels.to(device).long()

            logits = model(inputs)
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
