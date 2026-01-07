"""
Binary inference entrypoint for CardioGuard-AI.

Loads CNN checkpoint, XGBoost artifacts (scaler + calibrated model),
performs prediction, applies threshold, and computes ensemble probability.
Optionally generates XAI artifacts (Grad-CAM, SHAP, lead attention).

Usage:
    python -m src.pipeline.run_inference_binary --input path/to/ecg.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch

from src.data.signals import normalize_with_stats
from src.models.cnn import ECGCNN, ECGCNNConfig, MultiTaskECGCNN
from src.models.xgb import load_xgb
from src.utils.checkpoints import load_checkpoint_state_dict
from src.xai.gradcam import GradCAM
from src.xai.shap_xgb import explain_xgb, plot_shap_waterfall
from src.xai.summary import summarize_visual_explanations
from src.xai.visualize import plot_ecg_with_localization, plot_gradcam_heatmap, plot_lead_attention
from src.utils.llm_prompt import build_clinical_prompt, format_explanation_text


DEFAULT_ALPHA = 0.15


def load_cnn_model(checkpoint_path: Path, device: str) -> ECGCNN:
    config = ECGCNNConfig()
    state_dict = load_checkpoint_state_dict(checkpoint_path, device)
    if any(key.startswith("localization_head") for key in state_dict):
        model: ECGCNN | MultiTaskECGCNN = MultiTaskECGCNN(config)
    else:
        model = ECGCNN(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _extract_signal_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    for key in ("signal", "signals", "ecg", "data"):
        if key in data:
            signal = data[key]
            break
    else:
        arrays = [data[key] for key in data.files]
        if not arrays:
            raise ValueError(f"No arrays found in {npz_path}")
        signal = arrays[0]
    if signal.ndim == 3:
        signal = signal[0]
    return signal


def load_ecg_signal(input_path: Path) -> np.ndarray:
    if input_path.suffix.lower() == ".npz":
        signal = _extract_signal_from_npz(input_path)
    elif input_path.suffix.lower() == ".npy":
        signal = np.load(input_path)
    elif input_path.suffix.lower() in {".csv", ".txt"}:
        signal = np.loadtxt(input_path, delimiter=",")
    else:
        raise ValueError("Unsupported input format. Use .npz, .npy, or .csv/.txt")
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D ECG signal, got shape {signal.shape}")
    return signal


def ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    if signal.shape[0] == 12:
        return signal
    if signal.shape[1] == 12:
        return signal.T
    return signal


def load_normalization_stats(stats_path: Optional[Path]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if stats_path is None:
        return None
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found at {stats_path}")
    payload = np.load(stats_path)
    mean = payload["mean"].reshape(-1)
    std = payload["std"].reshape(-1)
    return mean, std


def apply_normalization(signal: np.ndarray, stats: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if stats is None:
        return signal
    mean, std = stats
    normalized = normalize_with_stats(signal.T, mean, std)
    return normalized.T


def decode_localization_bounds(
    localization: Optional[torch.Tensor],
    num_samples: int,
) -> Optional[Tuple[int, int]]:
    if localization is None:
        return None
    localization_np = localization.detach().cpu().numpy().reshape(-1)
    if localization_np.size < 2:
        return None
    normalized = 1 / (1 + np.exp(-localization_np[:2]))
    start_norm = float(min(normalized[0], normalized[1]))
    end_norm = float(max(normalized[0], normalized[1]))
    start_idx = int(round(start_norm * (num_samples - 1)))
    end_idx = int(round(end_norm * (num_samples - 1)))
    start_idx = max(0, min(start_idx, num_samples - 1))
    end_idx = max(0, min(end_idx, num_samples - 1))
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, num_samples - 1)
    return start_idx, end_idx


def load_threshold(metrics_path: Optional[Path]) -> float:
    if metrics_path is None or not metrics_path.exists():
        return 0.5
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    val_payload = payload.get("val", {})
    threshold = val_payload.get("best_threshold", 0.5)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return 0.5


def load_ensemble_alpha(config_path: Optional[Path]) -> float:
    if config_path is None or not config_path.exists():
        return DEFAULT_ALPHA
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return float(payload.get("best_alpha", DEFAULT_ALPHA))
    except (TypeError, ValueError):
        return DEFAULT_ALPHA


def maybe_generate_xai(
    signal: np.ndarray,
    cam: np.ndarray,
    shap_values: np.ndarray,
    base_value: float,
    shap_embedding: np.ndarray,
    feature_names: list[str],
    output_dir: Optional[Path],
    localization_bounds: Optional[Tuple[int, int]] = None,
) -> Dict[str, str]:
    if output_dir is None:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "inference"

    gradcam_path = output_dir / f"{stem}_gradcam.png"
    plot_gradcam_heatmap(
        signal,
        cam,
        save_path=gradcam_path,
        title="Grad-CAM Attention",
        localization_bounds=localization_bounds,
    )

    lead_attention_path = output_dir / f"{stem}_lead_attention.png"
    plot_lead_attention(cam, signal=signal, save_path=lead_attention_path, title="Lead Attention")

    if localization_bounds is not None:
        localization_path = output_dir / f"{stem}_localization.png"
        plot_ecg_with_localization(
            signal,
            localization_bounds=localization_bounds,
            save_path=localization_path,
            title="Localization Overlay",
        )
    else:
        localization_path = None

    shap_path = output_dir / f"{stem}_shap_waterfall.png"
    plot_shap_waterfall(
        shap_values,
        base_value,
        sample_idx=0,
        features=shap_embedding,
        feature_names=feature_names,
        save_path=shap_path,
        title="SHAP Explanation",
    )

    return {
        "gradcam": gradcam_path.as_posix(),
        "lead_attention": lead_attention_path.as_posix(),
        "localization": localization_path.as_posix() if localization_path else "",
        "shap": shap_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary inference entrypoint")
    parser.add_argument("--input", type=Path, required=True, help="ECG input file (.npz/.npy/.csv)")
    parser.add_argument("--cnn-path", type=Path, default=Path("checkpoints/ecgcnn.pt"))
    parser.add_argument("--xgb-dir", type=Path, default=Path("logs/xgb"))
    parser.add_argument("--xgb-path", type=Path, default=Path("logs/xgb/xgb_model.json"))
    parser.add_argument("--ensemble-config", type=Path, default=Path("reports/ensemble_config.json"))
    parser.add_argument("--metrics-path", type=Path, default=Path("logs/xgb/metrics.json"))
    parser.add_argument("--stats-npz", type=Path, default=None, help="NPZ with mean/std arrays")
    parser.add_argument("--xai-output-dir", type=Path, default=Path("reports/xai"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.cnn_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_path}")
    if not args.xgb_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {args.xgb_path}")

    signal = load_ecg_signal(args.input)
    signal = ensure_channel_first(signal)
    stats = load_normalization_stats(args.stats_npz)
    signal = apply_normalization(signal, stats)

    cnn_model = load_cnn_model(args.cnn_path, args.device)
    xgb_model = load_xgb(args.xgb_path)

    calibrated_path = args.xgb_dir / "xgb_calibrated.joblib"
    if calibrated_path.exists():
        xgb_calibrated = joblib.load(calibrated_path)
    else:
        xgb_calibrated = xgb_model

    scaler_path = args.xgb_dir / "xgb_scaler.joblib"
    xgb_scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    if signal_tensor.shape[1] != 12:
        signal_tensor = signal_tensor.permute(0, 2, 1)
    signal_tensor = signal_tensor.to(args.device)

    with torch.no_grad():
        output = cnn_model(signal_tensor)
        if isinstance(output, dict):
            cnn_logit = output["logits"]
            localization_pred = output.get("localization")
        else:
            cnn_logit = output
            localization_pred = None
        cnn_prob = torch.sigmoid(cnn_logit).item()
        embedding = cnn_model.backbone(signal_tensor).cpu().numpy()

    xgb_embedding = xgb_scaler.transform(embedding) if xgb_scaler is not None else embedding
    xgb_prob = float(xgb_calibrated.predict_proba(xgb_embedding)[0, 1])

    target_layer = cnn_model.backbone.features[4]
    gradcam = GradCAM(cnn_model, target_layer)
    cam = gradcam.generate(signal_tensor).squeeze()

    shap_model = getattr(xgb_calibrated, "base_model", xgb_calibrated)
    shap_result = explain_xgb(shap_model, xgb_embedding)
    shap_values = shap_result["shap_values"]
    base_value = shap_result["base_value"]
    feature_names = [f"CNN_Feature_{i}" for i in range(xgb_embedding.shape[1])]

    visual_summary = summarize_visual_explanations(
        cam=cam,
        signal=signal,
        shap_values=shap_values,
        feature_names=feature_names,
    )

    alpha = load_ensemble_alpha(args.ensemble_config)
    ensemble_prob = float(alpha * cnn_prob + (1 - alpha) * xgb_prob)

    threshold = load_threshold(args.metrics_path)
    prediction_label = "MI" if ensemble_prob >= threshold else "NORM"

    localization_bounds = decode_localization_bounds(
        localization_pred,
        num_samples=signal_tensor.shape[-1],
    )

    xai_images = maybe_generate_xai(
        signal,
        cam,
        shap_values,
        base_value,
        xgb_embedding,
        feature_names,
        args.xai_output_dir,
        localization_bounds=localization_bounds,
    )

    lead_summary = visual_summary["lead_attention"]
    shap_summary = visual_summary["shap_summary"]

    explanation_text = format_explanation_text(
        model_prediction=prediction_label,
        probability=ensemble_prob,
        lead_attention=lead_summary,
        shap_summary=shap_summary,
    )

    llm_prompt = build_clinical_prompt(
        model_prediction=prediction_label,
        probability=ensemble_prob,
        lead_attention=lead_summary,
        shap_summary=shap_summary,
        gradcam_images=[xai_images.get("gradcam", "")],
    )

    payload = {
        "mode": "binary",
        "prediction": {
            "label": prediction_label,
            "value": int(ensemble_prob >= threshold),
            "threshold": threshold,
        },
        "probabilities": {
            "cnn": cnn_prob,
            "xgb": xgb_prob,
            "ensemble": ensemble_prob,
        },
        "localization": {
            "bounds": localization_bounds,
        },
        "xai_images": xai_images,
        "explanation_text": explanation_text,
        "llm_prompt": llm_prompt,
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
