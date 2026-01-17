"""
Safe Model Loader with Key Normalization.

Handles different checkpoint schemas:
- Sequential: keys like 0.*, 1.classifier.* (binary)
- ECGCNN: keys like backbone.*, head.classifier.* (localization)
- MultiLabelECGCNN: keys like backbone.*, head.* (superclass)

Fail-fast with strict=True to prevent silent mismatches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Literal, Tuple, Optional

import torch
import torch.nn as nn


# =============================================================================
# Key Normalization
# =============================================================================

def normalize_state_dict_keys(msd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize state dict keys to consistent schema.
    
    Handles:
    - head.weight -> head.classifier.weight
    - head.bias -> head.classifier.bias
    """
    normalized = {}
    
    for key, value in msd.items():
        new_key = key
        
        # Normalize head.weight -> head.classifier.weight
        if key == "head.weight":
            new_key = "head.classifier.weight"
        elif key == "head.bias":
            new_key = "head.classifier.bias"
        
        normalized[new_key] = value
    
    return normalized


def detect_checkpoint_schema(msd: Dict[str, torch.Tensor]) -> Literal["sequential", "ecgcnn", "multilabel"]:
    """
    Detect checkpoint schema from state dict keys.
    
    Returns:
        "sequential": nn.Sequential(backbone, head) schema (0.*, 1.*)
        "ecgcnn": ECGCNN schema (backbone.*, head.classifier.*)
        "multilabel": MultiLabelECGCNN schema (backbone.*, head.*)
    """
    keys = set(msd.keys())
    
    # Sequential schema: 0.features.*, 1.classifier.*
    if any(k.startswith("0.") or k.startswith("1.") for k in keys):
        return "sequential"
    
    # ECGCNN/MultiLabel: backbone.*, head.*
    if any(k.startswith("backbone.") for k in keys):
        # Check if head uses classifier wrapper
        if any(k.startswith("head.classifier.") for k in keys):
            return "ecgcnn"
        elif any(k.startswith("head.") for k in keys):
            return "multilabel"
    
    # Fallback
    return "ecgcnn"


def get_output_dim_from_msd(msd: Dict[str, torch.Tensor]) -> int:
    """Extract output dimension from state dict."""
    patterns = [
        "head.classifier.weight",
        "head.weight",
        "1.classifier.weight",
    ]
    
    for pattern in patterns:
        if pattern in msd:
            return msd[pattern].shape[0]
    
    raise ValueError(f"Could not find output dimension. Keys: {list(msd.keys())[:10]}")


# =============================================================================
# Model Loading
# =============================================================================

def load_model_safe(
    checkpoint_path: Path,
    expected_task: Literal["binary", "superclass", "mi_localization"],
    device: str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint with schema detection and key normalization.
    
    Fail-fast with strict=True on load_state_dict.
    
    Args:
        checkpoint_path: Path to .pt file
        expected_task: Expected task type
        device: Target device
    
    Returns:
        (model, metadata) tuple
    
    Raises:
        ValueError: If checkpoint doesn't match expected task
        RuntimeError: If state dict loading fails
    """
    from src.models.cnn import ECGCNN, ECGCNNConfig, build_sequential_cnn
    
    TASK_DIMS = {"binary": 1, "superclass": 4, "mi_localization": 5}
    expected_dim = TASK_DIMS[expected_task]
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msd = checkpoint.get("model_state_dict", checkpoint)
    
    # Detect schema
    schema = detect_checkpoint_schema(msd)
    
    # Get and validate output dimension
    out_dim = get_output_dim_from_msd(msd)
    if out_dim != expected_dim:
        raise ValueError(
            f"Checkpoint {checkpoint_path.name} has out_dim={out_dim}, "
            f"expected {expected_dim} for task '{expected_task}'"
        )
    
    # Build model based on schema
    config = ECGCNNConfig()
    
    if schema == "sequential":
        # Binary checkpoint uses nn.Sequential(backbone, head)
        model = build_sequential_cnn(config, num_classes=out_dim)
    else:
        # ECGCNN or MultiLabel - normalize keys and use ECGCNN
        msd = normalize_state_dict_keys(msd)
        model = ECGCNN(config, num_classes=out_dim)
    
    # Load state dict with strict=True (FAIL-FAST)
    model.load_state_dict(msd, strict=True)
    model.to(device)
    model.eval()
    
    # Compute checkpoint hash for versioning
    with open(checkpoint_path, "rb") as f:
        checkpoint_hash = hashlib.md5(f.read()).hexdigest()[:8]
    
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "schema": schema,
        "out_dim": out_dim,
        "task": expected_task,
    }
    
    return model, metadata


# =============================================================================
# XGBoost Feature Schema
# =============================================================================

def create_feature_schema(
    feature_count: int,
    cnn_checkpoint_hash: str,
    scaler_path: Optional[Path] = None,
    xgb_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Create feature schema for XGBoost safety.
    
    This schema must be created at training time and validated at inference.
    """
    schema = {
        "version": "1.0.0",
        "feature_count": feature_count,
        "feature_names": [f"cnn_feat_{i}" for i in range(feature_count)],
        "embedder": {
            "type": "ECGCNN_backbone",
            "checkpoint_hash": cnn_checkpoint_hash,
        },
        "scaler_hash": None,
        "xgb_hash": None,
    }
    
    if scaler_path and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            schema["scaler_hash"] = hashlib.sha256(f.read()).hexdigest()[:16]
    
    if xgb_model_path and xgb_model_path.exists():
        with open(xgb_model_path, "rb") as f:
            schema["xgb_hash"] = hashlib.sha256(f.read()).hexdigest()[:16]
    
    return schema


def validate_feature_schema(
    embeddings_shape: Tuple[int, ...],
    schema: Dict[str, Any],
    strict: bool = True,
) -> bool:
    """
    Validate embeddings match feature schema.
    
    Raises:
        ValueError: If strict=True and validation fails
    """
    expected_count = schema.get("feature_count")
    actual_count = embeddings_shape[-1] if len(embeddings_shape) >= 1 else 0
    
    if actual_count != expected_count:
        msg = f"Feature count mismatch: got {actual_count}, expected {expected_count}"
        if strict:
            raise ValueError(msg)
        return False
    
    return True
