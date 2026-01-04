"""Tests for checkpoint utilities."""

import torch

from src.utils.checkpoints import remap_sequential_state_dict


def test_remap_sequential_state_dict() -> None:
    state_dict = {
        "0.features.0.weight": torch.randn(1),
        "1.classifier.weight": torch.randn(1),
        "module.0.features.1.bias": torch.randn(1),
    }

    remapped = remap_sequential_state_dict(state_dict)

    assert "backbone.features.0.weight" in remapped
    assert "head.classifier.weight" in remapped
    assert "backbone.features.1.bias" in remapped
