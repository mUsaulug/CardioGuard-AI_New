"""1D CNN model for ECG classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ECGCNNConfig:
    """Configuration for ECGCNN."""

    in_channels: int = 12
    num_filters: int = 64
    kernel_size: int = 7
    dropout: float = 0.3


class ECGCNN(nn.Module):
    """Conv1D-based ECG classifier returning a single logit."""

    def __init__(self, config: ECGCNNConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(config.in_channels, config.num_filters, config.kernel_size, padding=padding),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.num_filters, config.num_filters, config.kernel_size, padding=padding),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(config.num_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting shape (batch, channels, timesteps)."""

        features = self.features(x)
        pooled = features.squeeze(-1)
        return self.classifier(pooled).squeeze(-1)


class CNNEncoder(nn.Module):
    """Conv1D CNN encoder returning pooled embeddings."""

    def __init__(self, config: ECGCNNConfig) -> None:
        super().__init__()
        padding = config.kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(config.in_channels, config.num_filters, config.kernel_size, padding=padding),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.num_filters, config.num_filters, config.kernel_size, padding=padding),
            nn.BatchNorm1d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting shape (batch, channels, timesteps)."""

        features = self.features(x)
        return features.squeeze(-1)
