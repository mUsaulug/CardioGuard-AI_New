"""
Unit tests for XAI Artifact Discovery.

Uses existing run directory fixtures from reports/xai/runs/.
"""

import pytest
from pathlib import Path
from src.contracts.artifacts import (
    discover_xai_artifacts,
    extract_highlights,
    extract_sanity,
)


# =============================================================================
# Fixtures
# =============================================================================

# Use existing XAI run directory from the repo
EXISTING_RUN_DIR = Path("reports/xai/runs/20260106_085756__6eb3716__ecgcnn__multiclass")


# =============================================================================
# Test: discover_xai_artifacts()
# =============================================================================

class TestDiscoverXAIArtifacts:
    def test_discover_existing_run(self):
        """Test discovery on actual existing run directory."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        artifacts = discover_xai_artifacts(EXISTING_RUN_DIR)
        
        # Should find manifest and cards
        types = [a["type"] for a in artifacts]
        assert "manifest" in types
        assert "cards_jsonl" in types
    
    def test_discover_finds_visuals(self):
        """Test that visual artifacts are discovered."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        artifacts = discover_xai_artifacts(EXISTING_RUN_DIR)
        
        # Should find report PNGs
        visual_artifacts = [a for a in artifacts if a["mime"] == "image/png"]
        assert len(visual_artifacts) > 0
        
        # Check for combined_report type based on *_report.png pattern
        report_types = [a["type"] for a in visual_artifacts]
        assert "combined_report_png" in report_types
    
    def test_discover_finds_tensors(self):
        """Test that tensor artifacts are discovered."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        artifacts = discover_xai_artifacts(EXISTING_RUN_DIR)
        
        tensor_artifacts = [a for a in artifacts if a["type"] == "tensors_npz"]
        assert len(tensor_artifacts) > 0
    
    def test_discover_finds_narratives(self):
        """Test that narrative markdown files are discovered."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        artifacts = discover_xai_artifacts(EXISTING_RUN_DIR)
        
        narrative_artifacts = [a for a in artifacts if a["type"] == "narrative_md"]
        assert len(narrative_artifacts) > 0
    
    def test_discover_none_path(self):
        """Test with None path returns empty list."""
        artifacts = discover_xai_artifacts(None)
        assert artifacts == []
    
    def test_discover_missing_dir(self):
        """Test with non-existent directory returns empty list."""
        artifacts = discover_xai_artifacts(Path("nonexistent/path"))
        assert artifacts == []
    
    def test_artifact_structure(self):
        """Test that each artifact has required fields."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        artifacts = discover_xai_artifacts(EXISTING_RUN_DIR)
        
        for artifact in artifacts:
            assert "type" in artifact
            assert "path" in artifact
            assert "mime" in artifact
            assert "role" in artifact


# =============================================================================
# Test: extract_highlights()
# =============================================================================

class TestExtractHighlights:
    def test_extract_existing_sample(self):
        """Test highlight extraction for existing sample."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        # Use a sample ID that exists in cards.jsonl
        highlights = extract_highlights(EXISTING_RUN_DIR, "sample_000_normal")
        
        if highlights is not None:
            assert isinstance(highlights, list)
            assert len(highlights) <= 8  # Top 8
            
            # Check structure
            for h in highlights:
                assert "window_idx" in h
                assert "start_ms" in h
                assert "end_ms" in h
                assert "activation" in h
    
    def test_extract_nonexistent_sample(self):
        """Test with non-existent sample returns None or empty."""
        if not EXISTING_RUN_DIR.exists():
            pytest.skip("Test run directory does not exist")
        
        highlights = extract_highlights(EXISTING_RUN_DIR, "nonexistent_sample_xyz")
        assert highlights is None
    
    def test_extract_none_dir(self):
        """Test with None directory returns None."""
        highlights = extract_highlights(None, "sample_001")
        assert highlights is None


# =============================================================================
# Test: extract_sanity()
# =============================================================================

class TestExtractSanity:
    def test_extract_skipped_sanity(self):
        """Test extraction of SKIPPED sanity status."""
        explanation = {
            "sanity_check": {
                "overall": {
                    "status": "SKIPPED",
                    "passed_checks": 0,
                    "total_checks": 4
                }
            }
        }
        
        result = extract_sanity(explanation)
        
        assert result is not None
        assert result["status"] == "SKIPPED"
        assert result["passed_checks"] == 0
        assert result["total_checks"] == 4
    
    def test_extract_pass_sanity(self):
        """Test extraction of PASS sanity status."""
        explanation = {
            "sanity_check": {
                "overall": {
                    "status": "PASS",
                    "passed_checks": 4,
                    "total_checks": 4
                }
            }
        }
        
        result = extract_sanity(explanation)
        
        assert result["status"] == "PASS"
        assert result["passed_checks"] == 4
    
    def test_extract_fail_sanity(self):
        """Test extraction of FAIL sanity status."""
        explanation = {
            "sanity_check": {
                "overall": {
                    "status": "FAIL",
                    "passed_checks": 2,
                    "total_checks": 4
                }
            }
        }
        
        result = extract_sanity(explanation)
        
        assert result["status"] == "FAIL"
    
    def test_extract_none_explanation(self):
        """Test with None explanation returns None."""
        result = extract_sanity(None)
        assert result is None
    
    def test_extract_empty_explanation(self):
        """Test with empty explanation returns None."""
        result = extract_sanity({})
        assert result is None
    
    def test_derive_status_from_counts(self):
        """Test status derivation when status is invalid."""
        explanation = {
            "sanity_check": {
                "overall": {
                    "status": "UNKNOWN",  # Invalid
                    "passed_checks": 4,
                    "total_checks": 4
                }
            }
        }
        
        result = extract_sanity(explanation)
        
        # Should derive PASS from counts
        assert result["status"] == "PASS"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
