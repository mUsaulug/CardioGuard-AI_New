"""Grad-CAM implementation for CNN models."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


class GradCAM:
    """Compute Grad-CAM heatmaps for a target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, inputs: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for the given inputs."""

        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if isinstance(logits, (tuple, list)):
            raise TypeError("GradCAM expects model output to be logits only.")
        if logits.dim() == 1:
            score = logits.sum()
        elif logits.dim() == 2:
            if class_index is None:
                class_index = int(torch.argmax(logits, dim=1)[0])
            score = logits[:, class_index].sum()
        else:
            raise ValueError("Logits tensor must be 1D or 2D.")
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations are not captured.")

        if self.gradients.dim() == 3:
            weights = torch.mean(self.gradients, dim=2, keepdim=True)
        elif self.gradients.dim() == 4:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        else:
            raise ValueError("Gradients tensor must be 3D or 4D.")
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()
