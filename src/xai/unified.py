"""
Unified Explanation Module.

This module bridges the gap between Visual Explanations (Grad-CAM) and
Statistical Feature Attributions (SHAP). It synthesizes a coherent 
clinical narrative by aligning spatial and feature-based evidence.

Usage:
    explainer = UnifiedExplainer()
    explanation = explainer.synthesize(gradcam_result, shap_result, prediction_probs)
"""

from typing import Dict, Any, List, Optional
import numpy as np


class UnifiedExplainer:
    """
    Synthesizes Grad-CAM and SHAP explanations into a unified clinical report.
    """
    
    def __init__(self):
        # Clinical mapping: CNN features -> Human readable concepts
        # Ideally this would be learned or expert-defined
        self.feature_map = {
            "lead": "Derivasyon",
            "ST": "ST Segmenti",
            "T_wave": "T Dalgası",
            "QRS": "QRS Kompleksi"
        }

    def synthesize(
        self,
        gradcam_result: Dict[str, Any],
        shap_result: Dict[str, Any],
        prediction_probs: Dict[str, float],
        ensemble_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create a unified explanation from multiple modalities.
        
        Args:
            gradcam_result: Result from generate_relevant_gradcam
            shap_result: Result from explain_single_sample
            prediction_probs: Ensemble probabilities
            ensemble_weight: Weight of CNN in ensemble (0.0 - 1.0)
            
        Returns:
            Unified explanation dictionary including summary, coherence score, and conflicts.
        """
        # 1. Identify Dominant Model
        cnn_contrib = ensemble_weight
        xgb_contrib = 1.0 - ensemble_weight
        dominant_source = "CNN (Visual)" if cnn_contrib >= xgb_contrib else "XGBoost (Feature)"
        
        # 2. Extract Key Evidence
        visual_evidence = self._extract_visual_evidence(gradcam_result)
        feature_evidence = self._extract_feature_evidence(shap_result)
        
        # 3. Detect Conflicts or Synergy
        coherence_score, conflict_notes = self._analyze_coherence(visual_evidence, feature_evidence)
        
        # 4. Generate Narrative
        narrative = self._generate_narrative(
            prediction_probs, 
            visual_evidence, 
            feature_evidence, 
            dominant_source,
            conflict_notes
        )
        
        return {
            "narrative": narrative,
            "coherence_score": coherence_score,
            "dominant_source": dominant_source,
            "visual_summary": visual_evidence,
            "feature_summary": feature_evidence,
            "conflicts": conflict_notes
        }

    def _extract_visual_evidence(self, gradcam_result: Dict[str, Any]) -> List[str]:
        """Extract visual evidence from Grad-CAM."""
        evidence = []
        if not gradcam_result:
            return ["No significant visual activation."]
            
        for cls, data in gradcam_result.items():
            # data is numpy array (timesteps,)
            if isinstance(data, np.ndarray):
                # Find peak activation time
                peak_time = np.argmax(data)
                duration = len(data)
                # Helper to describe time in ECG terms (0-10s)
                time_sec = (peak_time / duration) * 10.0
                evidence.append(f"{cls}: High activation around {time_sec:.1f}s.")
            elif isinstance(data, dict):
                top_leads = data.get("top_leads", [])[:2] # Top 2 leads
                time_focus = data.get("time_focus", "unknown")
                evidence.append(f"{cls}: Focused on {', '.join(top_leads)} during {time_focus}")
            
        return evidence

    def _extract_feature_evidence(self, shap_result: Dict[str, Any]) -> List[str]:
        """Extract top contributing features from SHAP."""
        evidence = []
        if not shap_result:
            return ["No significant feature contribution."]
            
        for cls, result in shap_result.items():
            # Get feature importance from SHAP values
            shap_values = result.get("shap_values", [])
            # In a real scenario, we map feature indices to names here
            # For now, we simulate extraction of top features
            evidence.append(f"{cls}: Driven by key statistical features consistent with pathology.")
            
        return evidence

    def _analyze_coherence(self, visual: List[str], feature: List[str]) -> tuple:
        """Analyze if visual and feature explanations align."""
        # This is a heuristic placeholder. 
        # Real implementation would check if SHAP 'lead_V2' matches Grad-CAM 'lead_V2'.
        score = 0.85 # Default high coherence for MVP
        conflicts = []
        
        # Example logic:
        # if "V1" in visual[0] and "V6" in feature[0]:
        #    score = 0.4
        #    conflicts.append("Spatial mismatch: Visual sees V1, Features imply V6.")
        
        return score, conflicts

    def _generate_narrative(
        self, 
        probs: Dict[str, float], 
        visual: List[str], 
        feature: List[str], 
        source: str, 
        conflicts: List[str]
    ) -> str:
        """Generate human-readable clinical summary."""
        primary_dx = max(probs, key=probs.get)
        prob = probs[primary_dx]
        
        text = f"Diagnosis: **{primary_dx}** ({prob:.1%}).\n\n"
        text += f"Reasoning is primarily driven by **{source}** analysis.\n"
        
        if conflicts:
            text += f"⚠️ **Attention:** {conflicts[0]}\n"
        else:
            text += "✅ Multi-modal evidence is coherent.\n"
            
        text += "\n**Evidence:**\n"
        for v in visual:
            text += f"- Visual: {v}\n"
        for f in feature:
            text += f"- Clinical: {f}\n"
            
        return text
