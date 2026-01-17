"""
AIResult v1.0 Mapper Module.

Maps predict() output from run_inference_superclass.py to the standardized
AIResult contract for Spring↔Python↔RAG integration.

Version: 1.0.0
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import uuid


# =============================================================================
# Constants
# =============================================================================

AIRESULT_VERSION = "1.0.0"

PATHOLOGY_CLASSES = ["MI", "STTC", "CD", "HYP"]

# MI Localization: DERIVED anatomical regions from PTB-XL SCP codes
# NOT raw SCP codes - 13+ codes mapped to 5 targets via MI_CODE_TO_REGIONS
MI_LOCALIZATION_LABELS = ["AMI", "ASMI", "ALMI", "IMI", "LMI"]

# Mapping fingerprint for version tracking (sha256[:16] of sorted mapping dict)
# Recompute if MI_CODE_TO_REGIONS changes in mi_localization.py
MI_LOCALIZATION_MAPPING_FINGERPRINT = "8ab274e06afa1be8"

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# =============================================================================
# Helper Functions
# =============================================================================

def clamp(value: Any, min_val: float = 0.0, max_val: float = 1.0) -> Optional[float]:
    """
    Clamp value to [min_val, max_val]. Handle string parsing and invalid values.
    
    Returns None for NaN/Inf (triggers error).
    """
    if value is None:
        return None
    
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    
    if math.isnan(value) or math.isinf(value):
        return None
    
    return max(min_val, min(max_val, value))


def _validate_probability(prob: Any, class_name: str, errors: List[str]) -> float:
    """Validate and clamp a probability value, recording errors."""
    result = clamp(prob)
    if result is None:
        errors.append(f"Invalid probability for {class_name}: {prob}")
        return 0.0
    return result


# =============================================================================
# Triage Logic
# =============================================================================

def compute_triage(
    predictions: Dict[str, Any],
    input_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute triage level for UI display.
    
    Levels:
    - HIGH: MI detected
    - MEDIUM: Other pathology detected (STTC/CD/HYP)
    - LOW: NORM (no pathology)
    - REVIEW: Input validation failed
    """
    # Validation check first
    shape = input_meta.get("shape", [])
    if shape != [12, 1000]:
        return {
            "level": "REVIEW",
            "rule": "Input validation failed",
            "notes": f"Expected shape [12, 1000], got {shape}"
        }
    
    pathologies = predictions.get("multilabel", {}).get("pathologies", [])
    primary = predictions.get("primary", {})
    
    # Check MI first
    mi_decision = any(
        p.get("decision", False) 
        for p in pathologies 
        if p.get("class") == "MI"
    )
    if mi_decision or primary.get("label") == "MI":
        return {
            "level": "HIGH",
            "rule": "MI detected",
            "notes": f"Primary: {primary.get('label')} ({primary.get('confidence', 0):.2f})"
        }
    
    # Check other pathologies
    other_pathology = any(
        p.get("decision", False) 
        for p in pathologies 
        if p.get("class") in ["STTC", "CD", "HYP"]
    )
    if other_pathology:
        detected = [p["class"] for p in pathologies if p.get("decision")]
        return {
            "level": "MEDIUM",
            "rule": "Non-MI pathology detected",
            "notes": f"Detected: {', '.join(detected)}"
        }
    
    # NORM
    norm_score = predictions.get("multilabel", {}).get("norm", {}).get("norm_score", 0)
    return {
        "level": "LOW",
        "rule": "Normal ECG",
        "notes": f"NORM score: {norm_score:.2f}"
    }


# =============================================================================
# Input Metadata Derivation
# =============================================================================

def derive_input_meta(
    signal_path: Optional[Path] = None,
    request_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Derive input metadata without reading raw data.
    
    Uses defaults from PTB-XL dataset configuration.
    """
    meta = {
        "format": "npz",
        "sample_rate_hz": 100,
        "duration_sec": 10.0,
        "shape": [12, 1000],
        "leads": LEADS,
        "quality_flags": None
    }
    
    if signal_path is not None:
        meta["format"] = signal_path.suffix.lstrip(".")
    
    if request_payload:
        # Override with explicit values
        for key in ["format", "sample_rate_hz", "duration_sec", "shape"]:
            if key in request_payload and request_payload[key] is not None:
                meta[key] = request_payload[key]
    
    return meta


# =============================================================================
# Main Mapping Function
# =============================================================================

def map_predict_output_to_airesult(
    predict_out: Dict[str, Any],
    case_id: str,
    sample_id: str,
    request_id: Optional[str] = None,
    input_meta: Optional[Dict[str, Any]] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Deterministic mapping from predict() output to AIResult v1.0.
    
    Args:
        predict_out: Output from predict() in run_inference_superclass.py
        case_id: Unique case identifier (external)
        sample_id: Sample identifier (from filename or external)
        request_id: Pipeline execution ID (auto-generated if None)
        input_meta: Metadata about input signal
        run_dir: Optional XAI run directory for artifact discovery
    
    Returns:
        AIResult v1.0 compliant dict with both raw_predict and airesult
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    if input_meta is None:
        input_meta = derive_input_meta()
    
    # Extract probabilities from sources.ensemble (canonical for decisions)
    sources = predict_out.get("sources", {})
    ensemble_probs = sources.get("ensemble", {})
    
    if not ensemble_probs:
        # Fallback to multi.probabilities
        multi = predict_out.get("multi", {})
        ensemble_probs = multi.get("probabilities", {})
        if ensemble_probs:
            warnings.append("Using multi.probabilities instead of sources.ensemble")
    
    # Get thresholds
    multi = predict_out.get("multi", {})
    thresholds = multi.get("thresholds", {cls: 0.5 for cls in PATHOLOGY_CLASSES})
    
    # --- Build predictions.multilabel.pathologies ---
    pathologies = []
    for cls in PATHOLOGY_CLASSES:
        prob = _validate_probability(ensemble_probs.get(cls, 0.0), cls, errors)
        thresh = float(thresholds.get(cls, 0.5))
        pathologies.append({
            "class": cls,
            "probability": prob,
            "threshold": thresh,
            "decision": prob >= thresh
        })
    
    # --- Build predictions.multilabel.norm (DERIVED) ---
    max_pathology_prob = max(p["probability"] for p in pathologies) if pathologies else 0.0
    any_pathology_detected = any(p["decision"] for p in pathologies)
    
    norm = {
        "derived": True,
        "decision": not any_pathology_detected,
        "norm_score": clamp(1.0 - max_pathology_prob) or 0.0,
        "derived_rule": "1 - max(pathology_probabilities)"
    }
    
    # --- Build predictions.primary ---
    primary_out = predict_out.get("primary", {})
    if primary_out:
        primary = {
            "label": primary_out.get("label", "NORM"),
            "confidence": clamp(primary_out.get("confidence", 0.0)) or 0.0,
            "rule": primary_out.get("rule", "MI-first-then-priority")
        }
    else:
        # Compute primary using MI-first-then-priority
        primary_label, primary_conf = _compute_primary_label(
            {p["class"]: p["probability"] for p in pathologies},
            thresholds
        )
        primary = {
            "label": primary_label,
            "confidence": primary_conf,
            "rule": "MI-first-then-priority"
        }
    
    # --- Build localization (if present and MI detected) ---
    localization = None
    loc_out = predict_out.get("mi_localization")
    mi_detected = any(p["decision"] for p in pathologies if p["class"] == "MI")
    
    if loc_out and mi_detected:
        subtypes = []
        for code in MI_LOCALIZATION_LABELS:
            prob = loc_out.get(code)
            if prob is not None:
                prob_val = clamp(prob) or 0.0
                subtypes.append({
                    "code": code,
                    "probability": prob_val,
                    "detected": prob_val >= 0.5
                })
        
        localization = {
            "mi_detected": True,
            "label_space": "ptbxl_derived_anatomical_v1",
            "labels": MI_LOCALIZATION_LABELS,
            "mapping_source": "src/data/mi_localization.py",
            "mapping_fingerprint": MI_LOCALIZATION_MAPPING_FINGERPRINT,
            "subtypes": subtypes,
            "notes": [
                "Labels derived from PTB-XL SCP codes via MI_CODE_TO_REGIONS",
                "ILMI/IPLMI mapped to IMI+LMI (multi-hot)",
                "PMI (posterior) excluded"
            ]
        }
    
    # --- Build sources ---
    def parse_source_dict(d: Optional[Dict]) -> Optional[Dict[str, float]]:
        if d is None:
            return None
        return {k: clamp(v) or 0.0 for k, v in d.items()}
    
    sources_mapped = {
        "cnn": parse_source_dict(sources.get("cnn")),
        "xgb": parse_source_dict(sources.get("xgb")),
        "ensemble": parse_source_dict(sources.get("ensemble"))
    }
    
    # --- Build explanations (if present) ---
    explanations = None
    explanation_out = predict_out.get("explanation")
    
    if explanation_out or run_dir:
        from src.contracts.artifacts import discover_xai_artifacts, extract_highlights, extract_sanity
        
        artifacts = discover_xai_artifacts(run_dir) if run_dir else []
        highlights = extract_highlights(run_dir, sample_id) if run_dir else None
        sanity = extract_sanity(explanation_out) if explanation_out else None
        
        explanations = {
            "enabled": True,
            "run_dir": str(run_dir) if run_dir else None,
            "artifacts": artifacts,
            "highlights": highlights,
            "sanity": sanity
        }
    
    # --- Build versions ---
    versions_out = predict_out.get("versions", {})
    versions = {
        "cnn_checkpoint": versions_out.get("cnn_checkpoint", "checkpoints/ecgcnn_superclass.pt"),
        "xgb_dir": versions_out.get("xgb_dir"),
        "thresholds_file": versions_out.get("thresholds_file", "artifacts/thresholds_superclass.json"),
        "ensemble_best_alpha": 0.15,  # From ensemble_config.json
        "run_manifest": str(run_dir / "manifest.json") if run_dir else None,
        "airesult_version": AIRESULT_VERSION
    }
    
    # --- Build warnings ---
    mi_threshold = thresholds.get("MI", 0.5)
    if mi_threshold < 0.05:
        warnings.append(f"MI threshold ({mi_threshold}) indicates high-recall mode")
    
    # Check for NORM misuse
    if predict_out.get("multi", {}).get("probabilities", {}).get("NORM") is not None:
        warnings.append("NORM value in probabilities is derived, not a classifier output")
    
    # --- Build predictions ---
    predictions = {
        "multilabel": {
            "pathologies": pathologies,
            "norm": norm
        },
        "primary": primary
    }
    
    # --- Build triage ---
    triage = compute_triage(predictions, input_meta)
    
    # --- Assemble AIResult ---
    airesult = {
        "identity": {
            "case_id": case_id,
            "request_id": request_id,
            "sample_id": sample_id,
            "created_at": datetime.utcnow().isoformat() + "Z"
        },
        "mode": predict_out.get("mode", "multilabel-superclass"),
        "input": input_meta,
        "predictions": predictions,
        "localization": localization,
        "confidence": {
            "uncertainty": None,
            "calibration_method": "isotonic",
            "notes": "Uncertainty quantification not implemented"
        },
        "sources": sources_mapped,
        "explanations": explanations,
        "triage": triage,
        "reports": {
            "clinician_report_md": None,
            "patient_summary_md": None,
            "citations": []
        },
        "versions": versions,
        "warnings": warnings,
        "errors": errors
    }
    
    return airesult


def _compute_primary_label(
    probs: Dict[str, float],
    thresholds: Dict[str, float],
) -> Tuple[str, float]:
    """
    Compute primary label using MI-first-then-priority rule.
    
    1. MI first (highest clinical priority)
    2. Other pathologies in priority order
    3. NORM if no pathology detected
    """
    # 1. MI first
    if probs.get("MI", 0) >= thresholds.get("MI", 0.5):
        return "MI", probs["MI"]
    
    # 2. Other pathologies
    for cls in ["STTC", "CD", "HYP"]:
        if probs.get(cls, 0) >= thresholds.get(cls, 0.5):
            return cls, probs[cls]
    
    # 3. NORM
    max_pathology = max(probs.get(cls, 0) for cls in PATHOLOGY_CLASSES)
    norm_prob = 1.0 - max_pathology
    return "NORM", norm_prob
