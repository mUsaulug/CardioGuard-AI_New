"""
Unit tests for Consistency Guard.

Tests model agreement classification and triage assignment.
"""

import pytest

from src.pipeline.inference.consistency_guard import (
    AgreementType,
    ConsistencyResult,
    check_consistency,
    should_run_localization,
    derive_norm_from_superclass,
)


# =============================================================================
# Test: Agreement Classification
# =============================================================================

class TestCheckConsistency:
    def test_agree_mi(self):
        """Both models detect MI -> AGREE_MI."""
        result = check_consistency(
            superclass_mi_prob=0.85,
            binary_mi_prob=0.72,
            superclass_threshold=0.01,
            binary_threshold=0.5,
        )
        
        assert result.agreement == AgreementType.AGREE_MI
        assert result.triage_level == "HIGH"
        assert len(result.warnings) == 0
    
    def test_agree_no_mi(self):
        """Neither model detects MI -> AGREE_NO_MI."""
        result = check_consistency(
            superclass_mi_prob=0.005,
            binary_mi_prob=0.12,
            superclass_threshold=0.01,
            binary_threshold=0.5,
        )
        
        assert result.agreement == AgreementType.AGREE_NO_MI
        assert result.triage_level == "LOW"
        assert len(result.warnings) == 0
    
    def test_disagree_type_1(self):
        """Superclass MI=True, Binary MI=False -> DISAGREE_TYPE_1."""
        result = check_consistency(
            superclass_mi_prob=0.9,
            binary_mi_prob=0.3,
            superclass_threshold=0.01,
            binary_threshold=0.5,
        )
        
        assert result.agreement == AgreementType.DISAGREE_TYPE_1
        assert result.triage_level == "REVIEW"
        assert len(result.warnings) == 1
        assert "Binary model disagrees" in result.warnings[0]
    
    def test_disagree_type_2(self):
        """Superclass MI=False, Binary MI=True -> DISAGREE_TYPE_2."""
        result = check_consistency(
            superclass_mi_prob=0.005,
            binary_mi_prob=0.8,
            superclass_threshold=0.01,
            binary_threshold=0.5,
        )
        
        assert result.agreement == AgreementType.DISAGREE_TYPE_2
        assert result.triage_level == "REVIEW"
        assert len(result.warnings) == 1
        assert "detects MI not found by superclass" in result.warnings[0]
    
    def test_to_dict(self):
        """ConsistencyResult can be serialized to dict."""
        result = check_consistency(
            superclass_mi_prob=0.85,
            binary_mi_prob=0.72,
        )
        
        d = result.to_dict()
        assert d["agreement"] == "AGREE_MI"
        assert "superclass_mi_prob" in d
        assert "binary_mi_prob" in d


# =============================================================================
# Test: Localization Gate
# =============================================================================

class TestShouldRunLocalization:
    def test_strict_mode_agree_mi(self):
        """Strict mode: AGREE_MI -> run localization."""
        result = check_consistency(
            superclass_mi_prob=0.85,
            binary_mi_prob=0.72,
        )
        
        should_run = should_run_localization(result, gate_mode="strict")
        assert should_run is True
    
    def test_strict_mode_disagree(self):
        """Strict mode: DISAGREE -> don't run localization."""
        result = check_consistency(
            superclass_mi_prob=0.9,
            binary_mi_prob=0.3,
        )
        
        should_run = should_run_localization(result, gate_mode="strict")
        assert should_run is False
    
    def test_recall_first_mode_superclass_mi(self):
        """Recall-first mode: superclass MI=True -> run localization."""
        result = check_consistency(
            superclass_mi_prob=0.9,
            binary_mi_prob=0.3,  # Binary disagrees
        )
        
        should_run = should_run_localization(result, gate_mode="recall_first")
        assert should_run is True
    
    def test_recall_first_mode_no_superclass_mi(self):
        """Recall-first mode: superclass MI=False -> don't run."""
        result = check_consistency(
            superclass_mi_prob=0.005,
            binary_mi_prob=0.8,  # Binary detects, but superclass doesn't
        )
        
        should_run = should_run_localization(result, gate_mode="recall_first")
        assert should_run is False


# =============================================================================
# Test: NORM Derivation
# =============================================================================

class TestDeriveNorm:
    def test_norm_derived_no_pathology(self):
        """NORM decision should be True when no pathology detected."""
        probs = {"MI": 0.1, "STTC": 0.2, "CD": 0.15, "HYP": 0.1}
        
        norm = derive_norm_from_superclass(probs, threshold=0.5)
        
        assert norm["derived"] is True
        assert norm["norm_decision"] is True
        assert norm["norm_score"] == pytest.approx(0.8)  # 1 - max(0.1, 0.2, 0.15, 0.1)
    
    def test_norm_derived_with_pathology(self):
        """NORM decision should be False when pathology detected."""
        probs = {"MI": 0.7, "STTC": 0.2, "CD": 0.15, "HYP": 0.1}
        
        norm = derive_norm_from_superclass(probs, threshold=0.5)
        
        assert norm["derived"] is True
        assert norm["norm_decision"] is False
        assert norm["norm_score"] == pytest.approx(0.3)  # 1 - 0.7
    
    def test_norm_has_rule(self):
        """NORM result should include derivation rule."""
        probs = {"MI": 0.1, "STTC": 0.2, "CD": 0.15, "HYP": 0.1}
        
        norm = derive_norm_from_superclass(probs)
        
        assert "derived_rule" in norm
        assert "1 - max" in norm["derived_rule"]


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
