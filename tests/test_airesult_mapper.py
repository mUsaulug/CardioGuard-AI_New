"""
Unit tests for AIResult Mapper.

Tests the core mapping functionality without requiring model inference.
Uses mocked predict() outputs based on actual repo structure.
"""

import pytest
from datetime import datetime
from src.contracts.airesult_mapper import (
    map_predict_output_to_airesult,
    compute_triage,
    derive_input_meta,
    clamp,
    PATHOLOGY_CLASSES,
    MI_LOCALIZATION_LABELS,
    LEADS,
)


# =============================================================================
# Sample predict() outputs for testing
# =============================================================================

SAMPLE_PREDICT_OUTPUT = {
    "mode": "multilabel-superclass",
    "multi": {
        "probabilities": {
            "MI": 0.6523,
            "STTC": 0.4297,
            "CD": 0.2341,
            "HYP": 0.1892,
            "NORM": 0.3477,  # This is derived, should trigger warning
        },
        "thresholds": {
            "MI": 0.01,
            "STTC": 0.4184,
            "CD": 0.4200,
            "HYP": 0.2585,
        },
    },
    "primary": {
        "label": "MI",
        "confidence": 0.6523,
        "rule": "MI-first-then-priority",
    },
    "sources": {
        "cnn": {"MI": 0.7102, "STTC": 0.4853, "CD": 0.2012, "HYP": 0.1543},
        "xgb": {"MI": 0.5944, "STTC": 0.3741, "CD": 0.2670, "HYP": 0.2241},
        "ensemble": {"MI": 0.6523, "STTC": 0.4297, "CD": 0.2341, "HYP": 0.1892},
    },
    "mi_localization": {
        "AMI": 0.78,
        "ASMI": 0.32,
        "ALMI": 0.15,
        "IMI": 0.65,
        "LMI": 0.45,
    },
    "versions": {
        "cnn_checkpoint": "checkpoints/ecgcnn_superclass.pt",
        "xgb_dir": "logs/xgb_superclass",
        "thresholds_file": "artifacts/thresholds_superclass.json",
    },
}


SAMPLE_NORM_OUTPUT = {
    "mode": "multilabel-superclass",
    "multi": {
        "probabilities": {
            "MI": 0.005,
            "STTC": 0.12,
            "CD": 0.08,
            "HYP": 0.05,
        },
        "thresholds": {
            "MI": 0.01,
            "STTC": 0.4184,
            "CD": 0.4200,
            "HYP": 0.2585,
        },
    },
    "primary": {
        "label": "NORM",
        "confidence": 0.88,
    },
    "sources": {
        "cnn": {"MI": 0.005, "STTC": 0.12, "CD": 0.08, "HYP": 0.05},
        "ensemble": {"MI": 0.005, "STTC": 0.12, "CD": 0.08, "HYP": 0.05},
    },
    "versions": {},
}


# =============================================================================
# Test: clamp()
# =============================================================================

class TestClamp:
    def test_clamp_normal_value(self):
        assert clamp(0.5) == 0.5
    
    def test_clamp_below_min(self):
        assert clamp(-0.1) == 0.0
    
    def test_clamp_above_max(self):
        assert clamp(1.5) == 1.0
    
    def test_clamp_string_value(self):
        assert clamp("0.65229607") == pytest.approx(0.65229607)
    
    def test_clamp_none(self):
        assert clamp(None) is None
    
    def test_clamp_nan(self):
        import math
        assert clamp(float("nan")) is None
    
    def test_clamp_inf(self):
        assert clamp(float("inf")) is None


# =============================================================================
# Test: derive_input_meta()
# =============================================================================

class TestDeriveInputMeta:
    def test_default_values(self):
        meta = derive_input_meta()
        assert meta["format"] == "npz"
        assert meta["sample_rate_hz"] == 100
        assert meta["duration_sec"] == 10.0
        assert meta["shape"] == [12, 1000]
        assert meta["leads"] == LEADS
        assert len(meta["leads"]) == 12
    
    def test_correct_leads_order(self):
        meta = derive_input_meta()
        assert meta["leads"][0] == "I"
        assert meta["leads"][5] == "aVF"
        assert meta["leads"][6] == "V1"
        assert meta["leads"][11] == "V6"


# =============================================================================
# Test: compute_triage()
# =============================================================================

class TestComputeTriage:
    def test_triage_high_mi(self):
        predictions = {
            "multilabel": {
                "pathologies": [
                    {"class": "MI", "probability": 0.65, "threshold": 0.01, "decision": True},
                    {"class": "STTC", "probability": 0.43, "threshold": 0.42, "decision": True},
                    {"class": "CD", "probability": 0.23, "threshold": 0.42, "decision": False},
                    {"class": "HYP", "probability": 0.19, "threshold": 0.26, "decision": False},
                ],
                "norm": {"norm_score": 0.35, "decision": False}
            },
            "primary": {"label": "MI", "confidence": 0.65}
        }
        input_meta = {"shape": [12, 1000]}
        
        triage = compute_triage(predictions, input_meta)
        assert triage["level"] == "HIGH"
        assert triage["rule"] == "MI detected"
    
    def test_triage_medium_sttc(self):
        predictions = {
            "multilabel": {
                "pathologies": [
                    {"class": "MI", "probability": 0.008, "threshold": 0.01, "decision": False},
                    {"class": "STTC", "probability": 0.55, "threshold": 0.42, "decision": True},
                    {"class": "CD", "probability": 0.23, "threshold": 0.42, "decision": False},
                    {"class": "HYP", "probability": 0.19, "threshold": 0.26, "decision": False},
                ],
                "norm": {"norm_score": 0.45, "decision": False}
            },
            "primary": {"label": "STTC", "confidence": 0.55}
        }
        input_meta = {"shape": [12, 1000]}
        
        triage = compute_triage(predictions, input_meta)
        assert triage["level"] == "MEDIUM"
        assert triage["rule"] == "Non-MI pathology detected"
    
    def test_triage_low_norm(self):
        predictions = {
            "multilabel": {
                "pathologies": [
                    {"class": "MI", "probability": 0.005, "threshold": 0.01, "decision": False},
                    {"class": "STTC", "probability": 0.12, "threshold": 0.42, "decision": False},
                    {"class": "CD", "probability": 0.08, "threshold": 0.42, "decision": False},
                    {"class": "HYP", "probability": 0.05, "threshold": 0.26, "decision": False},
                ],
                "norm": {"norm_score": 0.88, "decision": True}
            },
            "primary": {"label": "NORM", "confidence": 0.88}
        }
        input_meta = {"shape": [12, 1000]}
        
        triage = compute_triage(predictions, input_meta)
        assert triage["level"] == "LOW"
        assert triage["rule"] == "Normal ECG"
    
    def test_triage_review_bad_shape(self):
        predictions = {"multilabel": {"pathologies": [], "norm": {}}, "primary": {}}
        input_meta = {"shape": [12, 500]}  # Wrong shape
        
        triage = compute_triage(predictions, input_meta)
        assert triage["level"] == "REVIEW"
        assert "validation failed" in triage["rule"].lower()


# =============================================================================
# Test: map_predict_output_to_airesult()
# =============================================================================

class TestMapPredictOutputToAIResult:
    def test_basic_mapping(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        # Check identity
        assert result["identity"]["case_id"] == "CASE-001"
        assert result["identity"]["sample_id"] == "sample_001"
        assert "created_at" in result["identity"]
        
        # Check mode
        assert result["mode"] == "multilabel-superclass"
    
    def test_pathologies_mapping(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        pathologies = result["predictions"]["multilabel"]["pathologies"]
        assert len(pathologies) == 4
        
        # Check classes
        classes = [p["class"] for p in pathologies]
        assert classes == PATHOLOGY_CLASSES
        
        # Check MI decision (threshold 0.01, probability 0.6523)
        mi = next(p for p in pathologies if p["class"] == "MI")
        assert mi["decision"] is True
        assert mi["probability"] == pytest.approx(0.6523)
        assert mi["threshold"] == 0.01
    
    def test_norm_derivation(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        norm = result["predictions"]["multilabel"]["norm"]
        
        # NORM must be marked as derived
        assert norm["derived"] is True
        assert norm["derived_rule"] == "1 - max(pathology_probabilities)"
        
        # With MI at 0.6523 as max, NORM score should be ~0.3477
        assert norm["norm_score"] == pytest.approx(1.0 - 0.6523)
        
        # With pathologies detected, NORM decision should be False
        assert norm["decision"] is False
    
    def test_norm_decision_true_when_no_pathology(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_NORM_OUTPUT,
            case_id="CASE-002",
            sample_id="sample_002",
        )
        
        norm = result["predictions"]["multilabel"]["norm"]
        
        # No pathology detected -> NORM decision True
        assert norm["decision"] is True
    
    def test_localization_mapping(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        loc = result["localization"]
        assert loc is not None
        assert loc["mi_detected"] is True
        assert loc["label_space"] == "ptbxl_derived_anatomical_v1"  # Derived, not raw SCP
        assert loc["labels"] == MI_LOCALIZATION_LABELS
        assert "mapping_fingerprint" in loc
        
        # Check subtypes
        subtypes = loc["subtypes"]
        assert len(subtypes) == 5
        
        ami = next(s for s in subtypes if s["code"] == "AMI")
        assert ami["probability"] == pytest.approx(0.78)
        assert ami["detected"] is True  # >= 0.5
    
    def test_localization_none_when_no_mi(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_NORM_OUTPUT,
            case_id="CASE-002",
            sample_id="sample_002",
        )
        
        # MI not detected -> no localization
        assert result["localization"] is None
    
    def test_warning_mi_low_threshold(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        warnings = result["warnings"]
        # MI threshold is 0.01 < 0.05 -> should warn
        assert any("MI threshold" in w and "high-recall" in w for w in warnings)
    
    def test_warning_norm_in_probabilities(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        warnings = result["warnings"]
        # NORM is in multi.probabilities -> should warn
        assert any("NORM" in w and "derived" in w for w in warnings)
    
    def test_triage_included(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001", 
            sample_id="sample_001",
        )
        
        triage = result["triage"]
        assert triage["level"] == "HIGH"
        assert "MI" in triage["rule"]
    
    def test_sources_preserved(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        sources = result["sources"]
        assert sources["cnn"]["MI"] == pytest.approx(0.7102)
        assert sources["xgb"]["MI"] == pytest.approx(0.5944)
        assert sources["ensemble"]["MI"] == pytest.approx(0.6523)
    
    def test_versions_preserved(self):
        result = map_predict_output_to_airesult(
            predict_out=SAMPLE_PREDICT_OUTPUT,
            case_id="CASE-001",
            sample_id="sample_001",
        )
        
        versions = result["versions"]
        assert versions["cnn_checkpoint"] == "checkpoints/ecgcnn_superclass.pt"
        assert versions["thresholds_file"] == "artifacts/thresholds_superclass.json"
        assert versions["ensemble_best_alpha"] == 0.15


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
