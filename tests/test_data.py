"""
CardioGuard-AI Data Layer Tests

Unit tests for metadata loading, label extraction, and splitting.
Run with: python -m pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PTBXLConfig, MI_CODES, DIAGNOSTIC_SUPERCLASSES, get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements, get_diagnostic_codes
from src.data.labels import (
    add_binary_mi_labels,
    add_superclass_labels,
    add_5class_labels,
    has_mi_code,
    has_norm_code,
    get_mi_codes,
    filter_valid_samples
)
from src.data.splits import (
    get_standard_split,
    verify_no_patient_leakage,
    get_split_statistics,
    filter_split_by_label
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config(project_root):
    """Get default configuration."""
    return get_default_config(project_root)


@pytest.fixture
def metadata(config):
    """Load PTB-XL metadata."""
    return load_ptbxl_metadata(config.metadata_path)


@pytest.fixture
def scp_statements(config):
    """Load SCP statements."""
    return load_scp_statements(config.scp_statements_path)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfig:
    """Tests for configuration module."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = PTBXLConfig()
        
        assert config.sampling_rate == 100
        assert config.train_folds == [1, 2, 3, 4, 5, 6, 7, 8]
        assert config.val_folds == [9]
        assert config.test_folds == [10]
        assert config.random_seed == 42
        assert config.min_likelihood == 50.0
    
    def test_config_paths(self, config):
        """Test that config paths exist."""
        assert config.metadata_path.exists(), f"Metadata not found: {config.metadata_path}"
        assert config.scp_statements_path.exists(), f"SCP statements not found: {config.scp_statements_path}"
        assert config.records_path.exists(), f"Records not found: {config.records_path}"
    
    def test_mi_codes_defined(self):
        """Test that MI codes are properly defined."""
        assert len(MI_CODES) > 0
        assert "IMI" in MI_CODES
        assert "ASMI" in MI_CODES
        assert "AMI" in MI_CODES


# ============================================================================
# Loader Tests
# ============================================================================

class TestLoader:
    """Tests for metadata loading."""
    
    def test_load_metadata_shape(self, metadata):
        """Test metadata has expected shape."""
        # PTB-XL has ~21800 records
        assert len(metadata) > 21000
        assert len(metadata) < 22000
    
    def test_metadata_required_columns(self, metadata):
        """Test metadata has required columns."""
        required_cols = [
            "patient_id", "age", "sex", "scp_codes",
            "strat_fold", "filename_lr", "filename_hr"
        ]
        for col in required_cols:
            assert col in metadata.columns, f"Missing column: {col}"
    
    def test_scp_codes_parsed(self, metadata):
        """Test that scp_codes are parsed as dicts."""
        first_scp = metadata.iloc[0]["scp_codes"]
        assert isinstance(first_scp, dict), "scp_codes should be dict"
    
    def test_load_scp_statements(self, scp_statements):
        """Test SCP statements loading."""
        # Should have ~70 statement types
        assert len(scp_statements) > 60
        
        # Check required columns
        assert "diagnostic_class" in scp_statements.columns
        assert "description" in scp_statements.columns
    
    def test_diagnostic_codes_exist(self, scp_statements):
        """Test that diagnostic codes can be filtered."""
        diagnostic = get_diagnostic_codes(scp_statements)
        assert len(diagnostic) > 0
        assert len(diagnostic) < len(scp_statements)


# ============================================================================
# Label Tests
# ============================================================================

class TestLabels:
    """Tests for label extraction."""
    
    def test_mi_code_detection(self):
        """Test MI code detection."""
        mi_record = {"IMI": 100.0, "SR": 0.0}
        norm_record = {"NORM": 100.0, "SR": 0.0}
        
        assert has_mi_code(mi_record) == True
        assert has_mi_code(norm_record) == False
    
    def test_norm_code_detection(self):
        """Test NORM code detection."""
        norm_record = {"NORM": 100.0, "SR": 0.0}
        mi_record = {"IMI": 100.0, "SR": 0.0}
        
        assert has_norm_code(norm_record) == True
        assert has_norm_code(mi_record) == False
    
    def test_binary_labels_distribution(self, metadata, scp_statements):
        """Test binary label extraction produces expected distribution."""
        df = add_binary_mi_labels(metadata, scp_statements)
        
        counts = df["label_mi_norm"].value_counts()
        
        # Should have samples in all categories
        assert 0 in counts.index, "No NORM samples found"
        assert 1 in counts.index, "No MI samples found"
        assert -1 in counts.index, "No excluded samples found"
        
        # NORM should be most common
        assert counts[0] > 8000, f"Too few NORM samples: {counts[0]}"
        
        # MI should have substantial samples
        assert counts[1] > 4000, f"Too few MI samples: {counts[1]}"
    
    def test_superclass_labels(self, metadata, scp_statements):
        """Test superclass label extraction."""
        df = add_superclass_labels(metadata.head(100), scp_statements)
        
        assert "diagnostic_superclass" in df.columns
        
        # Check that superclass is a list
        first_superclass = df.iloc[0]["diagnostic_superclass"]
        assert isinstance(first_superclass, list)
    
    def test_5class_labels_multi_hot(self, metadata, scp_statements):
        """Test 5-class multi-hot encoding."""
        df = add_5class_labels(metadata.head(100), scp_statements, multi_label=True)
        
        for superclass in DIAGNOSTIC_SUPERCLASSES:
            col = f"label_{superclass}"
            assert col in df.columns, f"Missing column: {col}"
            assert df[col].isin([0, 1]).all(), f"Invalid values in {col}"
    
    def test_filter_valid_samples(self, metadata, scp_statements):
        """Test filtering out excluded samples."""
        df = add_binary_mi_labels(metadata, scp_statements)
        
        original_len = len(df)
        filtered = filter_valid_samples(df, "label_mi_norm")
        
        assert len(filtered) < original_len
        assert -1 not in filtered["label_mi_norm"].values


# ============================================================================
# Split Tests
# ============================================================================

class TestSplits:
    """Tests for data splitting."""
    
    def test_standard_split_sizes(self, metadata):
        """Test that standard split produces expected sizes."""
        train_idx, val_idx, test_idx = get_standard_split(metadata)
        
        total = len(train_idx) + len(val_idx) + len(test_idx)
        assert total == len(metadata), "Split sizes don't sum to total"
        
        # Train should be ~80%, val ~10%, test ~10%
        train_ratio = len(train_idx) / total
        assert 0.75 < train_ratio < 0.85, f"Unexpected train ratio: {train_ratio}"
    
    def test_no_patient_leakage(self, metadata):
        """Test that no patient appears in multiple splits."""
        train_idx, val_idx, test_idx = get_standard_split(metadata)
        
        # This should not raise
        result = verify_no_patient_leakage(
            metadata, train_idx, val_idx, test_idx
        )
        assert result == True
    
    def test_unique_patients_per_split(self, metadata):
        """Test that patients are disjoint across splits."""
        train_idx, val_idx, test_idx = get_standard_split(metadata)
        
        train_patients = set(metadata.loc[train_idx, "patient_id"].dropna())
        val_patients = set(metadata.loc[val_idx, "patient_id"].dropna())
        test_patients = set(metadata.loc[test_idx, "patient_id"].dropna())
        
        assert len(train_patients & val_patients) == 0
        assert len(train_patients & test_patients) == 0
        assert len(val_patients & test_patients) == 0
    
    def test_split_statistics(self, metadata):
        """Test split statistics calculation."""
        train_idx, val_idx, test_idx = get_standard_split(metadata)
        
        stats = get_split_statistics(
            metadata, train_idx, val_idx, test_idx
        )
        
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats
        assert stats["train"]["samples"] > 0
        assert stats["val"]["samples"] > 0
        assert stats["test"]["samples"] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_full_binary_pipeline(self, metadata, scp_statements):
        """Test complete binary classification data preparation."""
        # Add labels
        df = add_binary_mi_labels(metadata, scp_statements)
        
        # Filter valid samples
        df_valid = filter_valid_samples(df, "label_mi_norm")
        
        # Get splits
        train_idx, val_idx, test_idx = get_standard_split(df_valid)
        
        # Verify no leakage
        verify_no_patient_leakage(df_valid, train_idx, val_idx, test_idx)
        
        # Check we have samples
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        
        # Check label distribution in each split
        for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_df = df_valid.loc[indices]
            assert 0 in split_df["label_mi_norm"].values, f"No NORM in {name}"
            assert 1 in split_df["label_mi_norm"].values, f"No MI in {name}"
    
    def test_full_5class_pipeline(self, metadata, scp_statements):
        """Test complete 5-class classification data preparation."""
        # Add labels
        df = add_5class_labels(metadata, scp_statements, multi_label=True)
        
        # Get splits
        train_idx, val_idx, test_idx = get_standard_split(df)
        
        # Verify we have samples in splits
        assert len(train_idx) > 15000
        assert len(val_idx) > 1500
        assert len(test_idx) > 1500


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
