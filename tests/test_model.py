"""Model shape tests for ECGCNN."""

import torch

from src.models.cnn import ECGCNN, ECGCNNConfig


def test_ecgcnn_forward_shape() -> None:
    config = ECGCNNConfig(in_channels=12, num_filters=32)
    model = ECGCNN(config)
    dummy = torch.randn(4, 12, 1000)
    output = model(dummy)

    assert output.shape == (4,)
