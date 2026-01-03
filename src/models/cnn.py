"""Simple CNN encoder model for feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class CNNConfig:
    """Configuration for the CNN encoder."""

    in_channels: int = 1
    num_classes: int = 2
    embedding_dim: int = 128
    dropout: float = 0.2


class CNNEncoder(nn.Module):
    """Convolutional encoder that returns embedding vectors."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(config.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.embedding(features)


class CNNClassifier(nn.Module):
    """CNN model with classification head."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits, embedding
