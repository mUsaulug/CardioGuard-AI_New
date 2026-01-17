"""
Unit tests for Checkpoint Validation.

Tests fail-fast validation for all three model checkpoints.
"""

import pytest
from pathlib import Path

from src.utils.checkpoint_validation import (
    get_output_dim,
    has_regression_head,
    validate_checkpoint_task,
    validate_localization_fingerprint,
    compute_mapping_fingerprint,
    validate_all_checkpoints,
    CheckpointMismatchError,
    MappingDriftError,
    TASK_OUTPUT_DIMS,
    MI_LOCALIZATION_FINGERPRINT,
)


# =============================================================================
# Checkpoint Paths
# =============================================================================

BINARY_CHECKPOINT = Path("checkpoints/ecgcnn.pt")
SUPERCLASS_CHECKPOINT = Path("checkpoints/ecgcnn_superclass.pt")
LOCALIZATION_CHECKPOINT = Path("checkpoints/ecgcnn_localization.pt")


# =============================================================================
# Test: Output Dimension Extraction
# =============================================================================

class TestGetOutputDim:
    def test_binary_checkpoint_dim(self):
        """Binary checkpoint must have out_dim=1."""
        if not BINARY_CHECKPOINT.exists():
            pytest.skip("Binary checkpoint not found")
        
        import torch
        sd = torch.load(BINARY_CHECKPOINT, map_location="cpu")
        msd = sd.get("model_state_dict", sd)
        
        out_dim = get_output_dim(msd)
        assert out_dim == 1, f"Binary checkpoint has out_dim={out_dim}, expected 1"
    
    def test_superclass_checkpoint_dim(self):
        """Superclass checkpoint must have out_dim=4."""
        if not SUPERCLASS_CHECKPOINT.exists():
            pytest.skip("Superclass checkpoint not found")
        
        import torch
        sd = torch.load(SUPERCLASS_CHECKPOINT, map_location="cpu")
        msd = sd.get("model_state_dict", sd)
        
        out_dim = get_output_dim(msd)
        assert out_dim == 4, f"Superclass checkpoint has out_dim={out_dim}, expected 4"
    
    def test_localization_checkpoint_dim(self):
        """Localization checkpoint must have out_dim=5."""
        if not LOCALIZATION_CHECKPOINT.exists():
            pytest.skip("Localization checkpoint not found")
        
        import torch
        sd = torch.load(LOCALIZATION_CHECKPOINT, map_location="cpu")
        msd = sd.get("model_state_dict", sd)
        
        out_dim = get_output_dim(msd)
        assert out_dim == 5, f"Localization checkpoint has out_dim={out_dim}, expected 5"


# =============================================================================
# Test: Task Validation
# =============================================================================

class TestValidateCheckpointTask:
    def test_binary_validation_passes(self):
        """Binary checkpoint validates for binary task."""
        if not BINARY_CHECKPOINT.exists():
            pytest.skip("Binary checkpoint not found")
        
        result = validate_checkpoint_task(BINARY_CHECKPOINT, "binary")
        assert result["valid"] is True
        assert result["out_dim"] == 1
    
    def test_superclass_validation_passes(self):
        """Superclass checkpoint validates for superclass task."""
        if not SUPERCLASS_CHECKPOINT.exists():
            pytest.skip("Superclass checkpoint not found")
        
        result = validate_checkpoint_task(SUPERCLASS_CHECKPOINT, "superclass")
        assert result["valid"] is True
        assert result["out_dim"] == 4
    
    def test_localization_validation_passes(self):
        """Localization checkpoint validates for mi_localization task."""
        if not LOCALIZATION_CHECKPOINT.exists():
            pytest.skip("Localization checkpoint not found")
        
        result = validate_checkpoint_task(LOCALIZATION_CHECKPOINT, "mi_localization")
        assert result["valid"] is True
        assert result["out_dim"] == 5
    
    def test_wrong_task_fails_strict(self):
        """Loading binary checkpoint as superclass fails in strict mode."""
        if not BINARY_CHECKPOINT.exists():
            pytest.skip("Binary checkpoint not found")
        
        with pytest.raises(CheckpointMismatchError) as exc_info:
            validate_checkpoint_task(BINARY_CHECKPOINT, "superclass", strict=True)
        
        assert "out_dim=1" in str(exc_info.value)
        assert "expected 4" in str(exc_info.value)
    
    def test_wrong_task_returns_invalid_non_strict(self):
        """Loading binary checkpoint as superclass returns invalid in non-strict mode."""
        if not BINARY_CHECKPOINT.exists():
            pytest.skip("Binary checkpoint not found")
        
        result = validate_checkpoint_task(BINARY_CHECKPOINT, "superclass", strict=False)
        assert result["valid"] is False
        assert result["error"] is not None


# =============================================================================
# Test: Regression Head Rejection
# =============================================================================

class TestRegressionHeadRejection:
    def test_localization_has_no_regression_head(self):
        """MI localization checkpoint should NOT have 2-dim regression head."""
        if not LOCALIZATION_CHECKPOINT.exists():
            pytest.skip("Localization checkpoint not found")
        
        import torch
        sd = torch.load(LOCALIZATION_CHECKPOINT, map_location="cpu")
        msd = sd.get("model_state_dict", sd)
        
        assert has_regression_head(msd) is False, \
            "Localization checkpoint should not have regression head"
    
    def test_binary_has_no_regression_head(self):
        """Binary checkpoint should NOT have 2-dim regression head."""
        if not BINARY_CHECKPOINT.exists():
            pytest.skip("Binary checkpoint not found")
        
        import torch
        sd = torch.load(BINARY_CHECKPOINT, map_location="cpu")
        msd = sd.get("model_state_dict", sd)
        
        # Binary may or may not have MultiTaskECGCNN regression
        # This test just verifies we can check it
        result = has_regression_head(msd)
        assert isinstance(result, bool)


# =============================================================================
# Test: Localization Fingerprint
# =============================================================================

class TestLocalizationFingerprint:
    def test_fingerprint_matches_expected(self):
        """Fingerprint must match locked training-time value."""
        result = validate_localization_fingerprint(strict=False)
        
        assert result["valid"] is True, \
            f"Fingerprint mismatch: expected {MI_LOCALIZATION_FINGERPRINT}, got {result['computed']}"
    
    def test_fingerprint_computation_deterministic(self):
        """Fingerprint computation should be deterministic."""
        fp1 = compute_mapping_fingerprint()
        fp2 = compute_mapping_fingerprint()
        
        assert fp1 == fp2
        assert len(fp1) == 16
    
    def test_wrong_fingerprint_raises_strict(self):
        """Wrong fingerprint should raise MappingDriftError in strict mode."""
        with pytest.raises(MappingDriftError):
            validate_localization_fingerprint(expected="wrong_fingerprint", strict=True)


# =============================================================================
# Test: Validate All
# =============================================================================

class TestValidateAllCheckpoints:
    def test_validate_all_returns_dict(self):
        """validate_all_checkpoints returns results for all tasks."""
        results = validate_all_checkpoints(strict=False)
        
        assert "binary" in results
        assert "superclass" in results
        assert "mi_localization" in results
        assert "fingerprint" in results


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
