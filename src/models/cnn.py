"""1D CNN model components for ECG classification."""

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
    localization_output_dim: int = 2


class ECGBackbone(nn.Module):
    """Conv1D CNN backbone returning pooled embeddings."""

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


class BinaryHead(nn.Module):
    """Binary classification head returning a single logit per sample."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class MultiClassHead(nn.Module):
    """Multi-class classification head returning logits for each class."""

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("MultiClassHead requires num_classes >= 2.")
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FiveClassHead(MultiClassHead):
    """Five-class classification head returning logits for 5 classes."""

    def __init__(self, in_features: int) -> None:
        super().__init__(in_features, num_classes=5)


class LocalizationHead(nn.Module):
    """Localization head returning regression logits."""

    def __init__(self, in_features: int, output_dim: int = 2) -> None:
        super().__init__()
        self.regressor = nn.Linear(in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class ECGCNN(nn.Module):
    """Conv1D-based ECG classifier returning logits."""

    def __init__(self, config: ECGCNNConfig, num_classes: int = 1) -> None:
        super().__init__()
        self.backbone = ECGBackbone(config)
        self.head = build_classification_head(config.num_filters, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting shape (batch, channels, timesteps)."""

        embeddings = self.backbone(x)
        return self.head(embeddings)


class MultiTaskECGCNN(nn.Module):
    """Conv1D-based ECG model with shared backbone and localization head."""

    def __init__(self, config: ECGCNNConfig, num_classes: int = 1) -> None:
        super().__init__()
        self.backbone = ECGBackbone(config)
        self.head = build_classification_head(config.num_filters, num_classes=num_classes)
        self.localization_head = LocalizationHead(
            config.num_filters,
            output_dim=config.localization_output_dim,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning classification logits and localization output."""

        embeddings = self.backbone(x)
        logits = self.head(embeddings)
        localization = self.localization_head(embeddings)
        return {"logits": logits, "localization": localization}


class CNNEncoder(ECGBackbone):
    """Backward-compatible alias for ECGBackbone."""


def build_classification_head(in_features: int, num_classes: int) -> nn.Module:
    """Create a classification head for binary or multi-class logits."""
    if num_classes == 1:
        return BinaryHead(in_features)
    return MultiClassHead(in_features, num_classes=num_classes)


def build_localization_head(in_features: int, output_dim: int = 2) -> nn.Module:
    """Create a localization head for regression outputs."""
    return LocalizationHead(in_features, output_dim=output_dim)


def build_sequential_cnn(config: ECGCNNConfig, num_classes: int) -> nn.Sequential:
    """Build Sequential(backbone, head) for stable checkpoint schemas."""
    backbone = ECGBackbone(config)
    head = build_classification_head(config.num_filters, num_classes=num_classes)
    return nn.Sequential(backbone, head)


def build_multitask_cnn(config: ECGCNNConfig, num_classes: int) -> MultiTaskECGCNN:
    """Build a multi-task CNN with shared backbone."""
    return MultiTaskECGCNN(config, num_classes=num_classes)
