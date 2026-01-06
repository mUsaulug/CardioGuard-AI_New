"""
Multi-label Superclass Inference Entrypoint.

Runs inference with CNN + XGBoost OVR ensemble.
Outputs multi-label predictions + primary label based on priority rule.

Usage:
    python -m src.pipeline.run_inference_superclass --input sample.npz --output result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import joblib
from sklearn.isotonic import IsotonicRegression  # Import needed for instance check

from src.models.cnn import ECGCNNConfig, ECGBackbone, ECGCNN
from src.pipeline.train_superclass_cnn import MultiLabelECGCNN, SUPERCLASS_LABELS
from src.pipeline.train_mi_localization import MI_LOCALIZATION_REGIONS
from src.xai.gradcam import generate_relevant_gradcam
from src.xai.shap_ovr import explain_single_sample
from src.xai.unified import UnifiedExplainer
from src.xai.unified import UnifiedExplainer
from src.xai.sanity import XAISanityChecker
from src.xai.visualize import plot_12lead_gradcam


# Default paths
DEFAULT_CNN_CHECKPOINT = Path("checkpoints/ecgcnn_superclass.pt")
DEFAULT_XGB_DIR = Path("logs/xgb_superclass")
DEFAULT_XGB_DIR = Path("logs/xgb_superclass")
DEFAULT_THRESHOLDS = Path("artifacts/thresholds_superclass.json")
DEFAULT_LOCALIZATION_CHECKPOINT = Path("checkpoints/ecgcnn_localization.pt")
DEFAULT_LOCALIZATION_THRESHOLDS = None # Use default 0.5 for now, or implement optimization later


def get_primary_label(probs: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[str, float]:
    """
    Determine primary label using MI-first-then-priority rule.
    
    Args:
        probs: Dict of class -> probability
        thresholds: Dict of class -> threshold
        
    Returns:
        (primary_label, confidence)
    """
    # 1. MI first (highest priority for clinical importance)
    if probs.get("MI", 0) >= thresholds.get("MI", 0.5):
        return "MI", probs["MI"]
    
    # 2. Other pathologies in priority order
    for cls in ["STTC", "CD", "HYP"]:
        if probs.get(cls, 0) >= thresholds.get(cls, 0.5):
            return cls, probs[cls]
    
    # 3. If no pathology detected, return NORM
    # NORM probability = 1 - max(pathology probs)
    max_pathology = max(probs.get(cls, 0) for cls in SUPERCLASS_LABELS)
    norm_prob = 1.0 - max_pathology
    return "NORM", norm_prob


def load_cnn_model(checkpoint_path: Path, device: torch.device) -> MultiLabelECGCNN:
    """Load trained multi-label CNN."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = ECGCNNConfig()
    model = MultiLabelECGCNN(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    return model


def load_localization_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load trained MI localization model."""
    if not checkpoint_path.exists():
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Config must match training (64 filters, 0.5 dropout)
    config = ECGCNNConfig(num_filters=64, dropout=0.5)
    model = ECGCNN(config, num_classes=len(MI_LOCALIZATION_REGIONS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    return model


def load_xgb_models(xgb_dir: Path) -> Dict[str, Any]:
    """Load XGBoost OVR models and calibrators."""
    from xgboost import XGBClassifier
    
    models = {}
    calibrators = {}
    scaler = None
    
    # Load scaler
    scaler_path = xgb_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    
    # Load per-class models
    for cls in SUPERCLASS_LABELS:
        cls_dir = xgb_dir / cls
        
        # Load model
        model_path = cls_dir / "xgb_model.json"
        if model_path.exists():
            model = XGBClassifier()
            model.load_model(model_path)
            models[cls] = model
        
        # Load calibrator
        calibrator_path = cls_dir / "calibrator.joblib"
        if calibrator_path.exists():
            calibrators[cls] = joblib.load(calibrator_path)
    
    return {"models": models, "calibrators": calibrators, "scaler": scaler}


def load_thresholds(thresholds_path: Path) -> Dict[str, float]:
    """Load optimized thresholds."""
    if not thresholds_path.exists():
        # Default thresholds
        return {cls: 0.5 for cls in SUPERCLASS_LABELS}
    
    with open(thresholds_path) as f:
        data = json.load(f)
    
    return data.get("thresholds", {cls: 0.5 for cls in SUPERCLASS_LABELS})


def load_ecg_signal(input_path: Path) -> np.ndarray:
    """Load ECG signal from various formats."""
    if input_path.suffix == ".npz":
        data = np.load(input_path)
        if "signal" in data:
            signal = data["signal"]
        elif "X" in data:
            signal = data["X"]
        else:
            # Assume first array
            signal = data[list(data.keys())[0]]
    elif input_path.suffix == ".npy":
        signal = np.load(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")
    
    return signal


def ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    """Ensure signal is (channels, timesteps) format."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    
    # Heuristic: 12-lead ECG, if first dim is 12, it's already channel-first
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    
    # Default: assume (timesteps, channels) and transpose
    if signal.shape[0] > signal.shape[1]:
        return signal.T
    
    return signal


def predict(
    signal: np.ndarray,
    cnn_model: MultiLabelECGCNN,
    xgb_data: Dict[str, Any],
    thresholds: Dict[str, float],
    localization_model: Optional[nn.Module],
    device: torch.device,
    ensemble_weight: float = 0.5,
    explain: bool = False,
    sanity_check: bool = False,
    save_plot: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run multi-label prediction.
    
    Args:
        signal: ECG signal (channels, timesteps)
        cnn_model: Trained CNN model
        xgb_data: XGBoost models, calibrators, scaler
        thresholds: Per-class thresholds
        device: Torch device
        ensemble_weight: Weight for CNN (1-weight for XGB)
        
    Returns:
        Prediction result dict
    """
    # Ensure correct format
    signal = ensure_channel_first(signal)
    
    # CNN prediction
    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        cnn_logits = cnn_model(signal_tensor)
        cnn_probs = torch.sigmoid(cnn_logits).cpu().numpy()[0]
    
    cnn_probs_dict = {cls: float(cnn_probs[i]) for i, cls in enumerate(SUPERCLASS_LABELS)}
    
    # XGBoost prediction (if available)
    xgb_probs_dict = {}
    if xgb_data["models"]:
        # Extract embeddings
        with torch.no_grad():
            embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()
        
        # Scale if scaler available
        if xgb_data["scaler"] is not None:
            embeddings = xgb_data["scaler"].transform(embeddings)
        
        # Predict with each OVR model
        for cls in SUPERCLASS_LABELS:
            if cls in xgb_data["models"]:
                model = xgb_data["models"][cls]
                raw_prob = model.predict_proba(embeddings)[0, 1]
                
                # Calibrate if calibrator available
                if cls in xgb_data["calibrators"]:
                    calibrator = xgb_data["calibrators"][cls]
                    
                    if isinstance(calibrator, IsotonicRegression):
                        # IsotonicRegressor only has predict(), no predict_proba
                        # And it expects 1D array
                        prob = calibrator.predict([raw_prob])[0]
                    else:
                        # LogisticRegression (Platt)
                        prob = calibrator.predict_proba([[raw_prob]])[0, 1]
                else:
                    prob = raw_prob
                
                xgb_probs_dict[cls] = float(prob)
    
    # Ensemble
    if xgb_probs_dict:
        w = ensemble_weight
        ensemble_probs = {
            cls: w * cnn_probs_dict[cls] + (1 - w) * xgb_probs_dict.get(cls, cnn_probs_dict[cls])
            for cls in SUPERCLASS_LABELS
        }
    else:
        ensemble_probs = cnn_probs_dict
    
    # Determine predicted labels (multi-label)
    predicted_labels = [
        cls for cls in SUPERCLASS_LABELS
        if ensemble_probs[cls] >= thresholds.get(cls, 0.5)
    ]
    
    # Determine primary label
    primary_label, primary_confidence = get_primary_label(ensemble_probs, thresholds)
    
    # NORM probability (derived)
    norm_prob = 1.0 - max(ensemble_probs.values())
    
    # ---------------------------------------------------------
    # MI Localization (Conditional)
    # ---------------------------------------------------------
    localization_result = None
    if localization_model and "MI" in predicted_labels:
        with torch.no_grad():
            signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
            loc_logits = localization_model(signal_tensor)
            loc_probs = torch.sigmoid(loc_logits).cpu().numpy()[0]
            
        localization_result = {
            region: float(prob)
            for region, prob in zip(MI_LOCALIZATION_REGIONS, loc_probs)
        }
        # Filter by threshold (default 0.5)
        detected_regions = [
            region for region, prob in localization_result.items()
            if prob >= 0.5
        ]
        localization_result["predicted_regions"] = detected_regions
    
    # ---------------------------------------------------------
    # XAI: Unified Explanation & Sanity Checks
    # ---------------------------------------------------------
    explanation_result = None
    sanity_result = None
    
    if explain:
        # 1. Grad-CAM (Visual)
        # Target layer: usually the last conv block of the backbone
        # We need to access it dynamically. For ECGBackbone, it's typically 'blocks'[-1]
        target_layer = cnn_model.backbone.features[-3] 
        gradcam_res = generate_relevant_gradcam(
            cnn_model, target_layer, signal_tensor, cnn_probs_dict, thresholds, top_k=2
        )
        
        # 2. SHAP (Feature)
        # Explain relevance for classes that have high probability or primary label
        relevant_for_shap = list(gradcam_res.keys())
        if primary_label != "NORM" and primary_label not in relevant_for_shap:
            relevant_for_shap.append(primary_label)
            
        shap_res = {}
        if xgb_data["models"] and relevant_for_shap:
            # We need embeddings for SHAP (1, 64)
            # embeddings was computed above
            shap_res = explain_single_sample(
                xgb_data["models"], 
                embeddings, # (1, 64)
                relevant_classes=relevant_for_shap
            )

        # 3. Unified Synthesis
        unifier = UnifiedExplainer()
        explanation_result = unifier.synthesize(
            gradcam_res, 
            shap_res, 
            ensemble_probs, 
            ensemble_weight
        )
        # Add raw results for external tools (like comprehensive test)
        explanation_result["raw_gradcam"] = gradcam_res
        explanation_result["raw_shap"] = shap_res
        
        # 4. Sanity Checks (Optional)
        if sanity_check:
            # We need a wrapper function for XAISanityChecker that matches signature (model, input) -> explanation
            # For simplicity, we test Grad-CAM stability on the primary label
            class_idx = SUPERCLASS_LABELS.index(primary_label) if primary_label != "NORM" else 0
            
            def explanation_func(m, inp):
                # Simple Grad-CAM generation for sanity check
                from src.xai.gradcam import GradCAM
                gcam = GradCAM(m, m.backbone.features[-3])
                return gcam.generate(inp, class_index=class_idx)
                
            checker = XAISanityChecker(cnn_model)
            sanity_result = checker.run_checks(
                signal_tensor, 
                gradcam_res.get(primary_label) if gradcam_res else None, 
                explanation_func
            )
            explanation_result["sanity_check"] = sanity_result
            
        # 5. Visualization (Optional)
        # 5. Visualization (Optional)
        if save_plot:
            from src.xai.visualize import plot_12lead_gradcam, plot_ecg_with_localization
            
            if gradcam_res:
                plot_12lead_gradcam(
                    signal, 
                    gradcam_res, 
                    save_plot, 
                    title=f"Prediction: {primary_label} ({primary_confidence:.2f})"
                )
                
            if localization_result:
                # Save as separate file: original_name_loc.png
                loc_plot_path = save_plot.parent / f"{save_plot.stem}_loc{save_plot.suffix}"
                plot_ecg_with_localization(
                    signal,
                    localization_result,
                    loc_plot_path,
                    title=f"MI Localization: {primary_label}"
                )

    return {
        "mode": "multilabel-superclass",
        "multi": {
            "probabilities": {**ensemble_probs, "NORM": norm_prob},
            "predicted_labels": predicted_labels if predicted_labels else ["NORM"],
            "thresholds": thresholds,
        },
        "primary": {
            "label": primary_label,
            "confidence": primary_confidence,
            "rule": "MI-first-then-priority",
        },
        "mi_localization": localization_result,
        "explanation": explanation_result,
        "sources": {
            "cnn": cnn_probs_dict,
            "xgb": xgb_probs_dict if xgb_probs_dict else None,
            "ensemble": ensemble_probs,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-label Superclass Inference")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to ECG signal (.npz or .npy)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for results (default: stdout)")
    parser.add_argument("--cnn-checkpoint", type=Path, default=DEFAULT_CNN_CHECKPOINT)
    parser.add_argument("--xgb-dir", type=Path, default=DEFAULT_XGB_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--localization-checkpoint", type=Path, default=DEFAULT_LOCALIZATION_CHECKPOINT)
    parser.add_argument("--ensemble-weight", type=float, default=0.5,
                        help="Weight for CNN (1-weight for XGB)")
    parser.add_argument("--explain", action="store_true", help="Generate Unified XAI explanation")
    parser.add_argument("--sanity-check", action="store_true", help="Run XAI sanity checks")
    parser.add_argument("--save-plot", type=Path, default=None, help="Path to save explanation plot")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    cnn_model = load_cnn_model(args.cnn_checkpoint, device)
    xgb_data = load_xgb_models(args.xgb_dir)
    thresholds = load_thresholds(args.thresholds)
    
    localization_model = load_localization_model(args.localization_checkpoint, device)
    if localization_model:
        print("Loaded MI Localization model.")
    else:
        print("Warning: MI Localization model not found, skipping localization.")
    
    # Load signal
    print(f"Loading signal from {args.input}...")
    signal = load_ecg_signal(args.input)
    
    # Predict
    print("Running inference...")
    result = predict(
        signal, cnn_model, xgb_data, thresholds, localization_model, device,
        ensemble_weight=args.ensemble_weight,
        explain=args.explain,
        sanity_check=args.sanity_check,
        save_plot=args.save_plot,
    )
    
    # Add metadata
    result["versions"] = {
        "cnn_checkpoint": str(args.cnn_checkpoint),
        "xgb_dir": str(args.xgb_dir),
        "thresholds_file": str(args.thresholds),
    }
    
    # Output
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(output_json)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Primary: {result['primary']['label']} ({result['primary']['confidence']:.3f})")
    print(f"Multi-label: {result['multi']['predicted_labels']}")
    if result.get("mi_localization"):
        print(f"Loc: {result['mi_localization']['predicted_regions']}")


if __name__ == "__main__":
    main()
