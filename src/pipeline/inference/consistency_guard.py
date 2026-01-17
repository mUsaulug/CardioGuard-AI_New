"""
Consistency Guard Module.

Compares Binary MI model output with Superclass MI output
to detect model disagreement and flag for review.

Agreement Types:
- AGREE_MI: Both models detect MI
- AGREE_NO_MI: Neither model detects MI
- DISAGREE_TYPE_1: Superclass MI=True, Binary MI=False (low confidence MI)
- DISAGREE_TYPE_2: Superclass MI=False, Binary MI=True (missed by superclass)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AgreementType(Enum):
    """Classification of model agreement."""
    AGREE_MI = "AGREE_MI"
    AGREE_NO_MI = "AGREE_NO_MI"
    DISAGREE_TYPE_1 = "DISAGREE_TYPE_1"  # Superclass MI, Binary No
    DISAGREE_TYPE_2 = "DISAGREE_TYPE_2"  # Superclass No, Binary MI


@dataclass
class ConsistencyResult:
    """Result of consistency check between models."""
    superclass_mi_prob: float
    binary_mi_prob: float
    superclass_mi_decision: bool
    binary_mi_decision: bool
    agreement: AgreementType
    triage_level: str  # "HIGH", "MEDIUM", "LOW", "REVIEW"
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "superclass_mi_prob": self.superclass_mi_prob,
            "binary_mi_prob": self.binary_mi_prob,
            "superclass_mi_decision": self.superclass_mi_decision,
            "binary_mi_decision": self.binary_mi_decision,
            "agreement": self.agreement.value,
            "triage_level": self.triage_level,
            "warnings": self.warnings,
        }


def check_consistency(
    superclass_mi_prob: float,
    binary_mi_prob: float,
    superclass_threshold: float = 0.01,  # Low threshold for high recall
    binary_threshold: float = 0.5,
) -> ConsistencyResult:
    """
    Check agreement between superclass and binary MI models.
    
    Args:
        superclass_mi_prob: MI probability from superclass model
        binary_mi_prob: MI probability from binary model
        superclass_threshold: Threshold for superclass MI decision
        binary_threshold: Threshold for binary MI decision
    
    Returns:
        ConsistencyResult with agreement type and warnings
    """
    superclass_mi = superclass_mi_prob >= superclass_threshold
    binary_mi = binary_mi_prob >= binary_threshold
    
    warnings = []
    
    if superclass_mi and binary_mi:
        agreement = AgreementType.AGREE_MI
        triage = "HIGH"
    elif not superclass_mi and not binary_mi:
        agreement = AgreementType.AGREE_NO_MI
        triage = "LOW"
    elif superclass_mi and not binary_mi:
        agreement = AgreementType.DISAGREE_TYPE_1
        triage = "REVIEW"
        warnings.append(
            f"Binary model disagrees—may be low confidence MI. "
            f"Superclass MI={superclass_mi_prob:.3f}, Binary MI={binary_mi_prob:.3f}"
        )
    else:  # not superclass_mi and binary_mi
        agreement = AgreementType.DISAGREE_TYPE_2
        triage = "REVIEW"
        warnings.append(
            f"Binary model detects MI not found by superclass. "
            f"Superclass MI={superclass_mi_prob:.3f}, Binary MI={binary_mi_prob:.3f}"
        )
    
    return ConsistencyResult(
        superclass_mi_prob=superclass_mi_prob,
        binary_mi_prob=binary_mi_prob,
        superclass_mi_decision=superclass_mi,
        binary_mi_decision=binary_mi,
        agreement=agreement,
        triage_level=triage,
        warnings=warnings,
    )


def should_run_localization(
    consistency: ConsistencyResult,
    gate_mode: str = "strict",
) -> bool:
    """
    Determine if MI localization should run based on consistency result.
    
    Args:
        consistency: Result from check_consistency()
        gate_mode: 
            "strict" - require both models to agree on MI
            "recall_first" - only require superclass MI
    
    Returns:
        True if localization should run
    """
    if gate_mode == "strict":
        return consistency.agreement == AgreementType.AGREE_MI
    elif gate_mode == "recall_first":
        return consistency.superclass_mi_decision
    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")


def derive_norm_from_superclass(
    superclass_probs: Dict[str, float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Derive NORM score and decision from superclass probabilities.
    
    NORM is NOT a classifier output. It is derived as:
    norm_score = 1 - max(pathology_probs)
    norm_decision = all pathologies below threshold
    
    Args:
        superclass_probs: Dict with MI, STTC, CD, HYP probabilities
        threshold: Threshold for pathology detection
    
    Returns:
        Dict with norm_score, norm_decision, derived=True
    """
    pathology_probs = [
        superclass_probs.get("MI", 0),
        superclass_probs.get("STTC", 0),
        superclass_probs.get("CD", 0),
        superclass_probs.get("HYP", 0),
    ]
    
    max_prob = max(pathology_probs)
    any_pathology = any(p >= threshold for p in pathology_probs)
    
    return {
        "derived": True,
        "norm_score": 1.0 - max_prob,
        "norm_decision": not any_pathology,
        "derived_rule": "1 - max(pathology_probabilities)"
    }
