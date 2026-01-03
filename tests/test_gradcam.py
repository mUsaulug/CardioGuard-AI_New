"""Grad-CAM tests for 1D ECGCNN."""

import torch

from src.models.cnn import ECGCNN, ECGCNNConfig
from src.xai.gradcam import GradCAM


def test_gradcam_ecgcnn_shape() -> None:
    config = ECGCNNConfig(in_channels=12, num_filters=16)
    model = ECGCNN(config)
    target_layer = model.features[4]
    gradcam = GradCAM(model, target_layer)
    dummy = torch.randn(2, 12, 128)

    cam = gradcam.generate(dummy)

    assert cam.shape == (2, 128)
