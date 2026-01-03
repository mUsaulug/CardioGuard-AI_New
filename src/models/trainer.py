"""Training utilities for CNN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.cnn import CNNClassifier


@dataclass
class TrainConfig:
    """Configuration for CNN training."""

    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"


def train_cnn(
    model: CNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: TrainConfig,
) -> Dict[str, float]:
    """Train a CNN model and return summary metrics."""

    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": 0.0, "val_loss": 0.0}
    for _ in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        history["train_loss"] = running_loss / len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits, _ = model(inputs)
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * inputs.size(0)
            history["val_loss"] = val_loss / len(val_loader.dataset)

    return history
