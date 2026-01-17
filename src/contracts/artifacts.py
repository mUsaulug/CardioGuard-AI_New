"""
XAI Artifact Discovery Module.

Discovers and normalizes XAI artifacts from run directories
under reports/xai/runs/<run_id>/.

Version: 1.0.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Artifact Patterns
# =============================================================================

# Fixed-path artifacts that always have the same location
FIXED_ARTIFACTS = {
    "manifest.json": {
        "type": "manifest",
        "mime": "application/json",
        "role": "metadata"
    },
    "cards.jsonl": {
        "type": "cards_jsonl",
        "mime": "application/jsonl",
        "role": "explanations"
    },
    "tables/sample_summary.csv": {
        "type": "summary_csv",
        "mime": "text/csv",
        "role": "summary"
    },
}

# Visual artifact patterns (matched in filename)
VISUAL_PATTERNS = {
    "gradcam": {"type": "gradcam_png", "role": "visual_explanation"},
    "shap_waterfall": {"type": "shap_waterfall_png", "role": "feature_attribution"},
    "shap_summary": {"type": "shap_summary_png", "role": "global_attribution"},
    "lead_attention": {"type": "lead_attention_png", "role": "lead_importance"},
    "ecg_sample": {"type": "ecg_png", "role": "signal_visualization"},
    "report": {"type": "combined_report_png", "role": "combined_explanation"},
}


# =============================================================================
# Discovery Functions
# =============================================================================

def discover_xai_artifacts(run_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """
    Discover XAI artifacts in a run directory.
    
    Args:
        run_dir: Path to reports/xai/runs/<run_id>/
    
    Returns:
        List of artifact descriptors with type, path, mime, role
    """
    if run_dir is None:
        return []
    
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    
    artifacts = []
    
    # --- Fixed-path artifacts ---
    for rel_path, meta in FIXED_ARTIFACTS.items():
        full_path = run_dir / rel_path
        if full_path.exists():
            artifacts.append({
                "type": meta["type"],
                "path": rel_path,
                "mime": meta["mime"],
                "role": meta["role"],
                "description": None
            })
    
    # --- Visual artifacts (pattern matching in visuals/) ---
    visuals_dir = run_dir / "visuals"
    if visuals_dir.exists():
        for png_file in visuals_dir.glob("*.png"):
            artifact_type = "unknown_png"
            role = "visual"
            
            stem_lower = png_file.stem.lower()
            for pattern, meta in VISUAL_PATTERNS.items():
                if pattern in stem_lower:
                    artifact_type = meta["type"]
                    role = meta["role"]
                    break
            
            artifacts.append({
                "type": artifact_type,
                "path": f"visuals/{png_file.name}",
                "mime": "image/png",
                "role": role,
                "description": None
            })
    
    # --- Tensor artifacts (tensors/) ---
    tensors_dir = run_dir / "tensors"
    if tensors_dir.exists():
        for npz_file in tensors_dir.glob("*.npz"):
            artifacts.append({
                "type": "tensors_npz",
                "path": f"tensors/{npz_file.name}",
                "mime": "application/octet-stream",
                "role": "raw_data",
                "description": None
            })
    
    # --- Narrative markdown (text/) ---
    text_dir = run_dir / "text"
    if text_dir.exists():
        for md_file in text_dir.glob("*.md"):
            artifacts.append({
                "type": "narrative_md",
                "path": f"text/{md_file.name}",
                "mime": "text/markdown",
                "role": "narrative",
                "description": None
            })
    
    return artifacts


def extract_highlights(
    run_dir: Optional[Path],
    sample_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract top_windows from cards.jsonl for a specific sample.
    
    The cards.jsonl structure has:
    - meta.sample_id: Sample identifier
    - xai_combined.top_windows: List of window objects with:
      - window_idx, start_ms, end_ms, mean_activation, max_activation
    
    Args:
        run_dir: Path to run directory
        sample_id: Sample identifier to find
    
    Returns:
        List of highlight windows or None if not found
    """
    if run_dir is None:
        return None
    
    run_dir = Path(run_dir)
    cards_path = run_dir / "cards.jsonl"
    
    if not cards_path.exists():
        return None
    
    try:
        with open(cards_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    card = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Match sample_id from meta
                if card.get("meta", {}).get("sample_id") == sample_id:
                    combined = card.get("xai_combined", {})
                    top_windows = combined.get("top_windows", [])
                    
                    return [
                        {
                            "window_idx": w.get("window_idx"),
                            "start_ms": w.get("start_ms"),
                            "end_ms": w.get("end_ms"),
                            "activation": w.get("max_activation", w.get("mean_activation"))
                        }
                        for w in top_windows[:8]  # Top 8
                    ]
    except Exception:
        pass
    
    return None


def extract_sanity(explanation_out: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract sanity check results from explanation output.
    
    Maps sanity.overall.status to enum: PASS/FAIL/SKIPPED
    
    Args:
        explanation_out: The "explanation" dict from predict() output
    
    Returns:
        Sanity summary dict or None
    """
    if explanation_out is None:
        return None
    
    # Check for sanity_check or sanity in the output
    sanity_check = explanation_out.get("sanity_check") or explanation_out.get("sanity")
    if not sanity_check:
        return None
    
    overall = sanity_check.get("overall", {})
    
    # Normalize status
    status = overall.get("status", "SKIPPED")
    if status not in ["PASS", "FAIL", "SKIPPED"]:
        # Try to derive from passed/total
        passed = overall.get("passed_checks", 0)
        total = overall.get("total_checks", 0)
        if total > 0:
            status = "PASS" if passed == total else "FAIL"
        else:
            status = "SKIPPED"
    
    return {
        "status": status,
        "passed_checks": overall.get("passed_checks", 0),
        "total_checks": overall.get("total_checks", 4)
    }
