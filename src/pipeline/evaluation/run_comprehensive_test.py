
import argparse
import json
import random
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata, load_scp_statements
from src.data.labels_superclass import add_superclass_labels_derived
from src.data.mi_localization import add_mi_localization_labels
from src.pipeline.run_inference_superclass import (
    load_cnn_model as load_cnn_superclass,
    load_xgb_models as load_xgb_superclass,
    load_thresholds as load_thresholds_superclass,
    load_localization_model,
    predict as predict_superclass
)
from src.pipeline.run_inference_binary import (
    load_cnn_model as load_cnn_binary,
    load_ensemble_alpha
)
from src.models.xgb import load_xgb
from src.xai.visualize import plot_12lead_gradcam, plot_ecg_with_localization, plot_gradcam_heatmap, plot_lead_attention
from src.xai.gradcam import GradCAM
from src.xai.shap_xgb import explain_xgb, plot_shap_waterfall
from src.xai.unified import UnifiedExplainer

def ensure_channel_first(signal: np.ndarray) -> np.ndarray:
    if signal.shape[0] == 12: return signal
    if signal.shape[1] == 12: return signal.T
    return signal

def run_binary_inference(signal, cnn_model, xgb_model, xgb_scaler, xgb_calibrated, alpha, device, output_dir):
    """Run legacy binary inference and return results + XAI."""
    
    # 1. Preprocess
    signal = ensure_channel_first(signal)
    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 2. CNN Inference
    with torch.no_grad():
        logits = cnn_model(signal_tensor)
        if isinstance(logits, dict): logits = logits["logits"]
        cnn_prob = torch.sigmoid(logits).item()
        embeddings = cnn_model.backbone(signal_tensor).cpu().numpy()
        
    # 3. XGB Inference
    xgb_emb = xgb_scaler.transform(embeddings) if xgb_scaler else embeddings
    xgb_prob = float(xgb_calibrated.predict_proba(xgb_emb)[0, 1])
    
    # 4. Ensemble
    final_prob = alpha * cnn_prob + (1 - alpha) * xgb_prob
    prediction = "MI" if final_prob >= 0.5 else "NORM"
    
    # 5. XAI - GradCAM
    target_layer = cnn_model.backbone.features[-3]
    gcam = GradCAM(cnn_model, target_layer)
    cam = gcam.generate(signal_tensor).squeeze()
    
    plot_gradcam_heatmap(
        cam, 
        output_dir / "binary_gradcam.png",
        title=f"Binary Grad-CAM (Prob: {final_prob:.2f})"
    )
    
    # 6. XAI - SHAP
    shap_res = explain_xgb(getattr(xgb_calibrated, "base_model", xgb_calibrated), xgb_emb)
    # Waterfall plot
    shap_target = output_dir / "binary_shap.png"
    # We need to manually construct a simpler waterfall here or reuse the one in shap_xgb if adaptable
    # For now, let's skip complex waterfall reuse to avoid API mismatch and just save data
    # Or try to plot it if possible
    try:
        plot_shap_waterfall(
             shap_res["shap_values"],
             shap_res["base_value"],
             0,
             xgb_emb,
             [f"Feat {i}" for i in range(64)],
             shap_target,
             title="Binary SHAP"
        )
    except:
        pass

    return {
        "label": prediction,
        "prob": final_prob,
        "cnn_prob": cnn_prob,
        "xgb_prob": xgb_prob
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/comprehensive_test"))
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cpu") # Test on CPU for stability
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading Data...")
    config = get_default_config()
    df = load_ptbxl_metadata(config.metadata_path)
    scp_df = load_scp_statements(config.scp_statements_path)
    df = add_superclass_labels_derived(df, scp_df)
    df = add_mi_localization_labels(df)
    
    # Filter for Validation Fold (9)
    val_df = df[df["strat_fold"] == 9].reset_index(drop=True)
    
    # Select samples: 7 MI, 8 Others (Total 15)
    mi_samples = val_df[val_df["label_MI"] == 1]
    other_samples = val_df[val_df["label_MI"] == 0]
    
    mi_indices = mi_samples.sample(n=min(7, len(mi_samples)), random_state=42).index.tolist()
    other_indices = other_samples.sample(n=min(8, len(other_samples)), random_state=42).index.tolist()
    test_indices = mi_indices + other_indices
    
    random.shuffle(test_indices)
    print(f"Selected {len(test_indices)} samples for testing ({len(mi_indices)} MI, {len(other_indices)} Other).")
    
    print("2. Loading Models...")
    # Binary
    print("  - Loading Binary...")
    bin_cnn = load_cnn_binary(Path("checkpoints/ecgcnn.pt"), "cpu")
    bin_xgb = load_xgb(Path("logs/xgb/xgb_model.json"))
    bin_scaler = joblib.load("logs/xgb/xgb_scaler.joblib")
    bin_calibrated = joblib.load("logs/xgb/xgb_calibrated.joblib")
    bin_alpha = 0.15
    
    # Superclass
    print("  - Loading Superclass...")
    sup_cnn = load_cnn_superclass(Path("checkpoints/ecgcnn_superclass.pt"), device)
    sup_xgb = load_xgb_superclass(Path("logs/xgb_superclass"))
    sup_thresh = load_thresholds_superclass(Path("artifacts/thresholds_superclass.json"))
    sup_loc = load_localization_model(Path("checkpoints/ecgcnn_localization.pt"), device)
    
    results = []
    
    print("3. Running Tests...")
    for idx in tqdm(test_indices):
        valid_sample = False
        try:
            row = val_df.loc[idx]
            file_path = config.data_root / row[config.filename_column]
            
            # Load Signal
            import wfdb
            try:
                record = wfdb.rdrecord(str(file_path))
                signal = record.p_signal
                valid_sample = True
            except:
                print(f"Skipping missing file: {file_path}")
                continue
                
            sample_dir = args.output_dir / f"sample_{idx}"
            sample_dir.mkdir(exist_ok=True)
            
            # --- Binary Inference ---
            bin_res = run_binary_inference(
                signal, bin_cnn, bin_xgb, bin_scaler, bin_calibrated, bin_alpha, device, sample_dir
            )
            
            # --- Superclass Inference ---
            sup_res = predict_superclass(
                signal, sup_cnn, sup_xgb, sup_thresh, sup_loc, device,
                explain=True, save_plot=sample_dir / "superclass_explanation.png"
            )
            
            # Collect Metrics
            record = {
                "sample_id": int(idx),
                "true_labels": [col for col in ["MI", "STTC", "CD", "HYP", "NORM"] if row.get(f"label_{col}", 0) == 1],
                "binary_pred": bin_res["label"],
                "binary_conf": bin_res["prob"],
                "superclass_pred": sup_res["primary"]["label"],
                "superclass_conf": sup_res["primary"]["confidence"],
                "superclass_all": sup_res["multi"]["predicted_labels"],
                "localization": sup_res["mi_localization"]["predicted_regions"] if sup_res["mi_localization"] else None,
                "loc_probs": sup_res["mi_localization"] if sup_res["mi_localization"] else None
            }
            results.append(record)
            
            # Save Narrative
            with open(sample_dir / "explanation.txt", "w", encoding="utf-8") as f:
                f.write(sup_res["explanation"]["narrative"])
            
            # --- Save Individual Superclass XAI Plots ---
            # Save individual Grad-CAM maps
            raw_gradcam = sup_res["explanation"].get("raw_gradcam", {})
            for cls, cam in raw_gradcam.items():
                plot_gradcam_heatmap(cam, sample_dir / f"sup_gradcam_{cls}.png", title=f"Superclass Grad-CAM: {cls}")
                
            # Save individual SHAP plots if available
            raw_shap = sup_res["explanation"].get("raw_shap", {})
            feature_summary = sup_res["explanation"].get("feature_summary", {})
            # Note: sup_res["explanation"]["feature_summary"] is a list of strings, 
            # but we want raw data for plotting.
            # In run_inference_superclass, shap_res is Dict[str, Dict] where each dict
            # has 'shap_values' and 'expected_value'.
            
            for cls, data in raw_shap.items():
                if isinstance(data, dict) and "shap_values" in data:
                    sv = data["shap_values"]
                    # Since we don't have feature names easily here, use indices
                    # Or get them from a standard list
                    indices = np.argsort(np.abs(sv))[::-1][:15]
                    vals = sv[indices]
                    names = [f"Feature {i}" for i in indices]
                    
                    plt.figure(figsize=(10, 6))
                    plt.barh(names[::-1], vals[::-1], color='orange')
                    plt.title(f"SHAP Values: {cls}")
                    plt.xlabel("SHAP Value (Impact)")
                    plt.tight_layout()
                    plt.savefig(sample_dir / f"sup_shap_{cls}.png")
                    plt.close()
                            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    # Save Summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(args.output_dir / "summary_results.csv", index=False)
    summary_df.to_json(args.output_dir / "summary_results.json", orient="records", indent=2)
    
    print("\n" + "="*50)
    print(f"Test Completed. Results saved to {args.output_dir}")
    print("="*50)
    if not summary_df.empty:
        cols_to_print = ["sample_id", "true_labels", "binary_pred", "superclass_pred", "localization"]
        # Only print columns that actually exist in the dataframe
        existing_cols = [c for c in cols_to_print if c in summary_df.columns]
        print(summary_df[existing_cols])
    else:
        print("No samples were successfully processed.")

if __name__ == "__main__":
    main()
