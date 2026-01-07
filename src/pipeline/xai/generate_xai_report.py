"""
XAI Report Generation CLI.

Generates comprehensive XAI reports for ECG predictions including:
- SHAP feature importance (XGBoost)
- Grad-CAM saliency (CNN)
- Combined SHAP-weighted explanation
- Sanity checks (randomization, faithfulness, stability)
- Structured outputs (JSONL, Parquet, NPZ, PNG, MD)

Usage:
    # Single sample
    python -m src.pipeline.generate_xai_report --input sample.npz --task multiclass
    
    # Batch
    python -m src.pipeline.generate_xai_report --input-dir samples/ --task multiclass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
from xgboost import XGBClassifier

import numpy as np
import torch

from src.models.cnn import ECGCNNConfig, ECGCNN, MultiTaskECGCNN
from src.pipeline.training.train_superclass_cnn import MultiLabelECGCNN, SUPERCLASS_LABELS
from src.pipeline.run_inference_superclass import (
    load_cnn_model as load_superclass_cnn_model, load_xgb_models, load_thresholds, load_ecg_signal,
    DEFAULT_CNN_CHECKPOINT, DEFAULT_XGB_DIR, DEFAULT_THRESHOLDS
)
from src.utils.signal import ensure_channel_first
from src.utils.checkpoints import load_checkpoint_state_dict
from src.data.mi_localization import MI_LOCALIZATION_REGIONS
from src.xai.gradcam import GradCAM, generate_relevant_gradcam, smooth_gradcam
from src.xai.sanity import XAISanityChecker
from src.xai.combined import CombinedExplainer, create_explanation_card
from src.xai.reporting import XAIReporter, generate_run_id
from src.xai.visualize import generate_xai_report_png

# Task-specific configurations
BINARY_LABELS = ["NORM", "MI"]
LEAD_LABELS = MI_LOCALIZATION_REGIONS  # ["AMI", "ASMI", "ALMI", "IMI", "LMI"]

# Default model paths for each task
DEFAULT_BINARY_CNN = Path("checkpoints/ecgcnn.pt")
DEFAULT_BINARY_XGB = Path("logs/xgb")
DEFAULT_LOCALIZATION_CNN = Path("checkpoints/ecgcnn_localization.pt")


def load_cnn_model(checkpoint_path: Path, device: torch.device, task: str = "multiclass"):
    """
    Load CNN model with task-appropriate architecture.
    
    - binary: ECGCNN (1 output)
    - multiclass: MultiLabelECGCNN (4 outputs)
    - localization: MultiTaskECGCNN or MultiLabelECGCNN (12 outputs)
    """
    if task == "multiclass":
        return load_superclass_cnn_model(checkpoint_path, device)
    
    # Binary or Localization - use ECGCNN base
    config = ECGCNNConfig()
    
    # Check for localization head or task
    if task == "localization":
        num_classes = len(MI_LOCALIZATION_REGIONS)
        model = ECGCNN(config, num_classes=num_classes)
    elif any(key.startswith("localization_head") for key in load_checkpoint_state_dict(checkpoint_path, str(device))):
        model = MultiTaskECGCNN(config)
    else:
        model = ECGCNN(config)
    
    state_dict = load_checkpoint_state_dict(checkpoint_path, str(device))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_binary_xgb_models(xgb_dir: Path) -> Dict[str, Any]:
    """Load binary XGBoost model (MI detection) from flat directory."""
    models = {}
    calibrators = {}
    scaler = None
    
    if not xgb_dir.exists():
        return {"models": {}, "calibrators": {}, "scaler": None}
    
    # Load scaler (binary uses xgb_scaler.joblib)
    scaler_path = xgb_dir / "xgb_scaler.joblib"
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Warning: Could not load scaler: {e}")
    
    # Load model (prefer calibrated)
    calibrated_path = xgb_dir / "xgb_calibrated.joblib"
    model_path = xgb_dir / "xgb_model.json"
    
    if calibrated_path.exists():
        try:
            models["MI"] = joblib.load(calibrated_path)
            print(f"Loaded calibrated XGBoost from {calibrated_path}")
        except Exception as e:
            print(f"Error loading calibrated XGB: {e}")
    elif model_path.exists():
        try:
            model = XGBClassifier()
            model.load_model(model_path)
            models["MI"] = model
            print(f"Loaded XGBoost model from {model_path}")
        except Exception as e:
            print(f"Error loading XGB model: {e}")
            
    return {"models": models, "calibrators": calibrators, "scaler": scaler}


def load_baseline(baseline_path: Optional[Path], train_mean_path: Optional[Path] = None) -> Optional[np.ndarray]:
    """Load baseline for faithfulness tests."""
    if baseline_path and baseline_path.exists():
        data = np.load(baseline_path)
        if "mean" in data:
            return data["mean"]
        return data[list(data.keys())[0]]
    
    if train_mean_path and train_mean_path.exists():
        data = np.load(train_mean_path)
        if "mean" in data:
            return data["mean"]
    
    return None


def get_gradcam_func(cnn_model, target_layer_name: str = "backbone.features"):
    """Create Grad-CAM function for sanity checker."""
    # Find target layer
    target_layer = None
    for name, module in cnn_model.named_modules():
        if "features" in name and isinstance(module, torch.nn.Sequential):
            # Get second-to-last layer (before pooling)
            target_layer = module[-3] if len(module) > 3 else module[0]
            break
    
    if target_layer is None:
        # Fallback: try backbone.features directly
        if hasattr(cnn_model, 'backbone') and hasattr(cnn_model.backbone, 'features'):
            features = cnn_model.backbone.features
            target_layer = features[-3] if len(features) > 3 else features[0]
    
    if target_layer is None:
        raise ValueError("Could not find target layer for Grad-CAM")
    
    def gradcam_func(model, input_tensor, target_class=None):
        """Generate Grad-CAM for given input."""
        gradcam = GradCAM(model, target_layer)
        class_idx = SUPERCLASS_LABELS.index(target_class) if target_class in SUPERCLASS_LABELS else 0
        cam = gradcam.generate(input_tensor, class_index=class_idx)
        return cam
    
    return gradcam_func, target_layer


def process_localization_sample(
    signal: np.ndarray,
    cnn_model: torch.nn.Module,
    device: torch.device,
    sample_id: str,
    reporter,  # XAIReporter
    baseline: Optional[np.ndarray] = None,
    run_sanity: bool = True
) -> Dict[str, Any]:
    """
    Process a sample for MI lead localization XAI.
    
    Unlike multiclass, localization outputs per-lead MI probabilities
    and generates Grad-CAM for each affected lead (no SHAP).
    """
    signal = ensure_channel_first(signal)
    
    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        logits = cnn_model(signal_tensor)
        lead_probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Map to lead names
    lead_prob_dict = {lead: float(lead_probs[i]) for i, lead in enumerate(LEAD_LABELS)}
    
    # Identify affected leads (threshold 0.5)
    affected_leads = [lead for lead, prob in lead_prob_dict.items() if prob > 0.5]
    
    # Generate Grad-CAM for the most affected lead
    gradcam_func, target_layer = get_gradcam_func(cnn_model)
    
    cnn_model.train()
    signal_tensor_grad = signal_tensor.clone().requires_grad_(True)
    
    # Use highest probability lead for primary CAM
    primary_lead_idx = int(np.argmax(lead_probs))
    
    gradcam_result = smooth_gradcam(
        cnn_model, target_layer, signal_tensor_grad,
        class_index=primary_lead_idx, n_samples=5, noise_std=0.1
    )
    cnn_model.eval()
    
    # Sanity checks (Grad-CAM only, no SHAP)
    sanity = {"overall": {"status": "SKIPPED", "passed_checks": 0, "total_checks": 4}}
    
    if run_sanity:
        checker = XAISanityChecker(cnn_model)
        
        def explanation_func(model, inp):
            gc = GradCAM(model, target_layer)
            return gc.generate(inp, class_index=primary_lead_idx)
        
        sanity = checker.run_checks(
            signal_tensor_grad.clone().requires_grad_(True),
            gradcam_result,
            explanation_func,
            baseline=baseline,
            class_index=primary_lead_idx
        )
    
    # Create prediction summary
    prediction = {
        "pred_class": LEAD_LABELS[primary_lead_idx],
        "pred_proba": float(lead_probs[primary_lead_idx]),
        "affected_leads": affected_leads,
        "lead_probs": lead_prob_dict
    }
    
    # Create explanation card (localization-specific)
    card = {
        "meta": {
            "run_id": reporter.run_id,
            "sample_id": sample_id,
            "task": "localization",
        },
        "prediction": prediction,
        "xai_gradcam": {
            "heatmap": gradcam_result.squeeze().tolist() if isinstance(gradcam_result, np.ndarray) else [],
            "primary_lead": LEAD_LABELS[primary_lead_idx]
        },
        "sanity": sanity
    }
    
    # Write card directly to JSONL (add_sample expects different args)
    reporter.cards_file.write(json.dumps(card, default=str) + "\n")
    
    # Generate narrative
    narrative = f"""**MI Lead Localization**

**Affected Leads**: {', '.join(affected_leads) if affected_leads else 'None detected'}

**Lead Probabilities**:
"""
    for lead, prob in sorted(lead_prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        narrative += f"- {lead}: {prob:.1%}\n"
    
    narrative_path = reporter.base_dir / "text" / f"{sample_id}__narrative.md"
    with open(narrative_path, "w", encoding="utf-8") as f:
        f.write(narrative)
    
    # Generate simple PNG (ECG + heatmap only)
    combined_heatmap = gradcam_result.squeeze()
    visual_path = reporter.base_dir / "visuals" / f"{sample_id}__report.png"
    generate_xai_report_png(
        signal=signal,
        combined_heatmap=combined_heatmap,
        shap_features=[],  # No SHAP for localization
        sanity_metrics=sanity,
        prediction=prediction,
        output_path=visual_path
    )
    
    return {
        "sample_id": sample_id,
        "prediction": prediction,
        "sanity_status": sanity.get("overall", {}).get("status")
    }


def process_single_sample(
    signal: np.ndarray,
    cnn_model: torch.nn.Module,
    xgb_data: Dict[str, Any],
    thresholds: Dict[str, float],
    device: torch.device,
    sample_id: str,
    reporter: XAIReporter,
    baseline: Optional[np.ndarray] = None,
    true_label: Optional[str] = None,
    run_sanity: bool = True,
    task: str = "multiclass",
    labels: List[str] = None
) -> Dict[str, Any]:
    """Process a single sample and add to reporter."""
    
    if labels is None:
        labels = SUPERCLASS_LABELS
    
    # Ensure correct format
    signal = ensure_channel_first(signal)
    
    # Handle localization task differently (per-lead output)
    if task == "localization":
        return process_localization_sample(
            signal, cnn_model, device, sample_id, reporter, baseline, run_sanity
        )
    
    # Get predictions (binary/multiclass)
    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
        cnn_logits = cnn_model(signal_tensor)
        cnn_probs = torch.sigmoid(cnn_logits).cpu().numpy()[0]
        embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()
    
    probs_dict = {}
    if cnn_probs.size == 1:
        # Binary case: Single probability (MI)
        mi_prob = float(cnn_probs) if cnn_probs.ndim == 0 else float(cnn_probs[0])
        # Force binary labels if scalar
        probs_dict = {"MI": mi_prob, "NORM": 1.0 - mi_prob}
    else:
        # Multiclass case
        probs_dict = {cls: float(cnn_probs[i]) for i, cls in enumerate(labels)}
    
    # XGBoost predictions (if available)
    if xgb_data and xgb_data.get("models"):
        if xgb_data.get("scaler") is not None:
            embeddings_scaled = xgb_data["scaler"].transform(embeddings)
        else:
            embeddings_scaled = embeddings
        
        for cls in labels:
            if cls in xgb_data["models"]:
                model = xgb_data["models"][cls]
                raw_prob = model.predict_proba(embeddings_scaled)[0, 1]
                # Simple average with CNN
                probs_dict[cls] = 0.5 * probs_dict[cls] + 0.5 * raw_prob
    
    # Determine prediction
    pred_class = max(probs_dict, key=probs_dict.get)
    pred_proba = probs_dict[pred_class]
    
    # Find runnerup
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    runnerup = sorted_probs[1][0] if len(sorted_probs) > 1 else None
    runnerup_proba = sorted_probs[1][1] if len(sorted_probs) > 1 else None
    
    prediction = {
        "pred_class": pred_class,
        "pred_proba": pred_proba,
        "runnerup": runnerup,
        "runnerup_proba": runnerup_proba,
        "all_probs": probs_dict
    }
    
    # Get Grad-CAM function
    gradcam_func, target_layer = get_gradcam_func(cnn_model)
    
    # Generate Grad-CAM (requires gradients)
    # Using SmoothGrad-CAM for more stable explanations
    def gradcam_wrapper(model, inp, target=None):
        gc = GradCAM(model, target_layer)
        class_idx = labels.index(pred_class) if pred_class in labels else 0
        return gc.generate(inp, class_index=class_idx)
    
    # Enable gradients for Grad-CAM
    cnn_model.train()
    signal_tensor_grad = signal_tensor.clone().requires_grad_(True)
    class_idx = labels.index(pred_class) if pred_class in labels else 0
    
    # Use SmoothGrad-CAM for more stable explanations (n_samples=5)
    gradcam_result = smooth_gradcam(
        cnn_model, target_layer, signal_tensor_grad,
        class_index=class_idx, n_samples=5, noise_std=0.1
    )
    cnn_model.eval()
    
    # Combined explanation
    combined_explainer = CombinedExplainer(
        cnn_model, xgb_data, class_order=SUPERCLASS_LABELS
    )
    
    explanation = combined_explainer.explain(
        signal_tensor, embeddings, pred_class, probs_dict,
        contrastive=True,
        gradcam_func=lambda m, t, c: gradcam_result
    )
    explanation["gradcam"] = {"heatmap": gradcam_result.squeeze()}
    
    # Sanity checks
    sanity = {"overall": {"status": "SKIPPED", "passed_checks": 0, "total_checks": 4}}
    
    if run_sanity:
        checker = XAISanityChecker(cnn_model)
        
        def explanation_func(model, inp):
            gc = GradCAM(model, target_layer)
            class_idx = SUPERCLASS_LABELS.index(pred_class) if pred_class in SUPERCLASS_LABELS else 0
            return gc.generate(inp, class_index=class_idx)
        
        class_idx = SUPERCLASS_LABELS.index(pred_class) if pred_class in SUPERCLASS_LABELS else 0
        
        # Ensure input has gradients for sanity checks
        signal_for_sanity = signal_tensor.clone().requires_grad_(True)
        sanity = checker.run_checks(
            signal_for_sanity,
            gradcam_result,
            explanation_func,
            baseline=baseline,
            class_index=class_idx
        )
    
    # Generate narrative
    narrative = combined_explainer.generate_narrative(explanation, pred_class, pred_proba)
    
    # Add to reporter
    reporter.add_sample(
        sample_id=sample_id,
        explanation=explanation,
        sanity=sanity,
        prediction=prediction,
        signal=signal,
        true_label=true_label,
        narrative=narrative
    )
    
    # Generate visual report
    shap_features = explanation.get("shap", {}).get("top_features", [])
    combined_heatmap = explanation.get("combined", {}).get("heatmap")
    if combined_heatmap is None or (isinstance(combined_heatmap, np.ndarray) and combined_heatmap.size == 0):
        combined_heatmap = gradcam_result.squeeze()
    
    visual_path = reporter.base_dir / "visuals" / f"{sample_id}__report.png"
    generate_xai_report_png(
        signal=signal,
        combined_heatmap=combined_heatmap,
        shap_features=shap_features,
        sanity_metrics=sanity,
        prediction=prediction,
        output_path=visual_path
    )
    
    return {
        "sample_id": sample_id,
        "prediction": prediction,
        "sanity_status": sanity.get("overall", {}).get("status")
    }


def main():
    parser = argparse.ArgumentParser(description="Generate XAI Reports")
    parser.add_argument("--input", type=Path, help="Single input file (.npz or .npy)")
    parser.add_argument("--input-dir", type=Path, help="Directory of input files")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/xai/runs"),
                        help="Output directory for reports")
    parser.add_argument("--task", type=str, default="multiclass",
                        choices=["binary", "multiclass", "localization"],
                        help="Task type")
    parser.add_argument("--cnn-checkpoint", type=Path, default=DEFAULT_CNN_CHECKPOINT)
    parser.add_argument("--xgb-dir", type=Path, default=DEFAULT_XGB_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Path to baseline for faithfulness tests")
    parser.add_argument("--model-tag", type=str, default="ecgcnn",
                        help="Model tag for run ID")
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip sanity checks (faster)")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load models based on task
    print("Loading models...")
    
    # Select correct model paths based on task
    if args.task == "binary":
        cnn_path = args.cnn_checkpoint if args.cnn_checkpoint != DEFAULT_CNN_CHECKPOINT else DEFAULT_BINARY_CNN
        xgb_path = args.xgb_dir if args.xgb_dir != DEFAULT_XGB_DIR else DEFAULT_BINARY_XGB
        labels = BINARY_LABELS
    elif args.task == "localization":
        cnn_path = args.cnn_checkpoint if args.cnn_checkpoint != DEFAULT_CNN_CHECKPOINT else DEFAULT_LOCALIZATION_CNN
        xgb_path = None  # No XGBoost for localization
        labels = LEAD_LABELS
    else:  # multiclass
        cnn_path = args.cnn_checkpoint
        xgb_path = args.xgb_dir
        labels = SUPERCLASS_LABELS
    
    cnn_model = load_cnn_model(cnn_path, device, task=args.task)
    
    # Load XGBoost models
    if args.task == "binary" and xgb_path:
        xgb_data = load_binary_xgb_models(xgb_path)
    elif xgb_path:
        xgb_data = load_xgb_models(xgb_path)
    else:
        xgb_data = None
            
    thresholds = load_thresholds(args.thresholds) if args.task != "localization" else {}
    
    # Load baseline
    baseline = load_baseline(args.baseline)
    baseline_source = str(args.baseline) if args.baseline else "zeros"
    
    # Generate run ID
    run_id = generate_run_id(args.model_tag, args.task)
    print(f"Run ID: {run_id}")
    
    # Create reporter
    reporter = XAIReporter(
        run_id=run_id,
        output_dir=args.output_dir,
        task=args.task,
        model_id=str(cnn_path.stem),
        xgb_id=str(xgb_path.name) if xgb_path and xgb_path.exists() else "none",
        baseline_source=baseline_source
    )
    
    # Collect input files
    input_files = []
    if args.input:
        input_files.append(args.input)
    elif args.input_dir:
        input_files = list(args.input_dir.glob("*.npz")) + list(args.input_dir.glob("*.npy"))
    
    print(f"Processing {len(input_files)} samples...")
    
    # Process each sample
    results = []
    for i, input_path in enumerate(input_files):
        sample_id = input_path.stem
        print(f"[{i+1}/{len(input_files)}] Processing {sample_id}...")
        
        try:
            signal = load_ecg_signal(input_path)
            result = process_single_sample(
                signal=signal,
                cnn_model=cnn_model,
                xgb_data=xgb_data,
                thresholds=thresholds,
                device=device,
                sample_id=sample_id,
                reporter=reporter,
                baseline=baseline,
                run_sanity=not args.skip_sanity,
                task=args.task,
                labels=labels
            )
            results.append(result)
            print(f"  -> {result['prediction']['pred_class']} "
                  f"({result['prediction']['pred_proba']:.1%}) "
                  f"[{result['sanity_status']}]")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({"sample_id": sample_id, "error": str(e)})
    
    # Finalize report
    manifest_path = reporter.finalize()
    print(f"\n{'='*60}")
    print(f"Report generated: {reporter.base_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Samples: {len([r for r in results if 'error' not in r])}/{len(results)} successful")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
