"""
Checkpoint Validation Module.

Provides fail-fast validation for model checkpoints to prevent
loading wrong checkpoints for wrong tasks.

Tasks:
- binary: 1 output (MI vs NORM)
- superclass: 4 outputs [MI, STTC, CD, HYP]
- mi_localization: 5 outputs [AMI, ASMI, ALMI, IMI, LMI]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch


# =============================================================================
# Custom Exceptions
# =============================================================================

class CheckpointMismatchError(Exception):
    """Raised when checkpoint doesn't match expected task."""
    pass


class MappingDriftError(Exception):
    """Raised when MI localization mapping has changed since training."""
    pass


# =============================================================================
# Constants
# =============================================================================

TASK_OUTPUT_DIMS: Dict[str, int] = {
    "binary": 1,
    "superclass": 4,
    "mi_localization": 5,
}

TASK_LABELS: Dict[str, list] = {
    "binary": ["MI"],  # vs NORM
    "superclass": ["MI", "STTC", "CD", "HYP"],
    "mi_localization": ["AMI", "ASMI", "ALMI", "IMI", "LMI"],
}

# Locked at training time - update if MI_CODE_TO_REGIONS changes
MI_LOCALIZATION_FINGERPRINT = "8ab274e06afa1be8"


# =============================================================================
# Core Functions
# =============================================================================

def get_output_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Extract output dimension from model state dict.
    
    Handles different head patterns:
    - head.classifier.weight -> MultiClassHead
    - head.weight -> Direct Linear
    - 1.classifier.weight -> Sequential binary
    
    Raises:
        ValueError: If no classifier weights found
    """
    patterns = [
        "head.classifier.weight",
        "head.weight",
        "1.classifier.weight",
    ]
    
    for pattern in patterns:
        if pattern in state_dict:
            return state_dict[pattern].shape[0]
    
    raise ValueError(
        f"Could not find classifier weights. "
        f"Available keys: {[k for k in state_dict.keys() if 'weight' in k]}"
    )


def has_regression_head(state_dict: Dict[str, torch.Tensor]) -> bool:
    """
    Check if checkpoint contains MultiTaskECGCNN 2-dim regression head.
    
    This head is for temporal localization regression,
    NOT the MI localization 5-label classification task.
    """
    regression_keys = [
        "localization_head.regressor.weight",
        "localization_head.regressor.bias",
    ]
    
    for key in regression_keys:
        if key in state_dict:
            out_dim = state_dict[key].shape[0]
            if out_dim == 2:
                return True
    
    return False


def validate_checkpoint_task(
    checkpoint_path: Path,
    expected_task: Literal["binary", "superclass", "mi_localization"],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint and validate output dimension matches expected task.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        expected_task: One of "binary", "superclass", "mi_localization"
        strict: If True, raise exception on mismatch
    
    Returns:
        Validation result dict with keys: task, out_dim, valid, error
    
    Raises:
        CheckpointMismatchError: If strict=True and validation fails
    """
    result = {
        "checkpoint": str(checkpoint_path),
        "expected_task": expected_task,
        "out_dim": None,
        "valid": False,
        "error": None,
    }
    
    try:
        sd = torch.load(checkpoint_path, map_location="cpu")
        msd = sd.get("model_state_dict", sd)
        
        # Check for regression head misuse
        if expected_task == "mi_localization" and has_regression_head(msd):
            error = (
                "Checkpoint contains 2-dim regression head (MultiTaskECGCNN). "
                "This is NOT the MI localization 5-label classifier."
            )
            result["error"] = error
            if strict:
                raise CheckpointMismatchError(error)
            return result
        
        # Get and validate output dimension
        out_dim = get_output_dim(msd)
        result["out_dim"] = out_dim
        
        expected_dim = TASK_OUTPUT_DIMS[expected_task]
        if out_dim != expected_dim:
            error = (
                f"Checkpoint has out_dim={out_dim}, "
                f"expected {expected_dim} for task '{expected_task}'"
            )
            result["error"] = error
            if strict:
                raise CheckpointMismatchError(error)
            return result
        
        result["valid"] = True
        return result
        
    except FileNotFoundError:
        result["error"] = f"Checkpoint not found: {checkpoint_path}"
        if strict:
            raise
        return result


def compute_mapping_fingerprint() -> str:
    """
    Compute fingerprint of current MI localization mapping.
    
    Returns:
        16-character hex fingerprint
    """
    from src.data.mi_localization import MI_CODE_TO_REGIONS, MI_LOCALIZATION_REGIONS
    
    data = {
        "regions": MI_LOCALIZATION_REGIONS,
        "mapping": dict(sorted(MI_CODE_TO_REGIONS.items()))
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()[:16]


def validate_localization_fingerprint(
    expected: str = MI_LOCALIZATION_FINGERPRINT,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate that MI localization mapping hasn't changed since training.
    
    Args:
        expected: Expected fingerprint from training time
        strict: If True, raise exception on mismatch
    
    Returns:
        Validation result dict
    
    Raises:
        MappingDriftError: If strict=True and fingerprints don't match
    """
    computed = compute_mapping_fingerprint()
    
    result = {
        "computed": computed,
        "expected": expected,
        "valid": computed == expected,
        "error": None,
    }
    
    if not result["valid"]:
        error = f"MI localization mapping changed! Expected {expected}, got {computed}"
        result["error"] = error
        if strict:
            raise MappingDriftError(error)
    
    return result


def validate_all_checkpoints(
    binary_path: Path = Path("checkpoints/ecgcnn.pt"),
    superclass_path: Path = Path("checkpoints/ecgcnn_superclass.pt"),
    localization_path: Path = Path("checkpoints/ecgcnn_localization.pt"),
    strict: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate all three model checkpoints in one call.
    
    Returns:
        Dict with validation results for each task
    """
    results = {}
    
    results["binary"] = validate_checkpoint_task(
        binary_path, "binary", strict=strict
    )
    results["superclass"] = validate_checkpoint_task(
        superclass_path, "superclass", strict=strict
    )
    results["mi_localization"] = validate_checkpoint_task(
        localization_path, "mi_localization", strict=strict
    )
    
    # Also validate mapping fingerprint
    results["fingerprint"] = validate_localization_fingerprint(strict=strict)
    
    return results
