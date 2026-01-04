"""Checkpoint utilities for model loading."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def remap_sequential_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap state dict keys from Sequential(0,1) to backbone/head modules."""

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)
        if new_key.startswith("0."):
            new_key = new_key.replace("0.", "backbone.", 1)
        elif new_key.startswith("1."):
            new_key = new_key.replace("1.", "head.", 1)
        remapped[new_key] = value
    return remapped


def load_checkpoint_state_dict(checkpoint_path: str | Path, device: str | torch.device) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return a normalized state dict."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    return remap_sequential_state_dict(state_dict)
