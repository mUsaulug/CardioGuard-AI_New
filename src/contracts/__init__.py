"""
AIResult Contract Module.

Provides standardized mapping between raw predict() output
and the AIResult v1.0 contract for Spring↔Python↔RAG integration.

Usage:
    from src.contracts import map_predict_output_to_airesult, discover_xai_artifacts
    
    # Map predict output
    airesult = map_predict_output_to_airesult(
        predict_out=result,
        case_id="CASE-001",
        sample_id="ecg_sample_001",
    )
    
    # Discover XAI artifacts
    artifacts = discover_xai_artifacts(run_dir)
"""

from src.contracts.airesult_mapper import (
    AIRESULT_VERSION,
    PATHOLOGY_CLASSES,
    MI_LOCALIZATION_LABELS,
    MI_LOCALIZATION_MAPPING_FINGERPRINT,
    LEADS,
    map_predict_output_to_airesult,
    compute_triage,
    derive_input_meta,
    clamp,
)

from src.contracts.artifacts import (
    discover_xai_artifacts,
    extract_highlights,
    extract_sanity,
)

__all__ = [
    # Version
    "AIRESULT_VERSION",
    
    # Constants
    "PATHOLOGY_CLASSES",
    "MI_LOCALIZATION_LABELS",
    "MI_LOCALIZATION_MAPPING_FINGERPRINT",
    "LEADS",
    
    # Mapper
    "map_predict_output_to_airesult",
    "compute_triage",
    "derive_input_meta",
    "clamp",
    
    # Artifacts
    "discover_xai_artifacts",
    "extract_highlights",
    "extract_sanity",
]
