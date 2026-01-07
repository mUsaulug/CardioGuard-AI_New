
"""
Script to generate all academic assets (figures, tables, plots) for the CardioGuard-AI paper.
"""

import sys
import json
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from src.config import get_default_config
from src.data.loader import load_ptbxl_metadata
from src.data.labels import add_binary_mi_labels
from src.data.signals import SignalDataset
from src.models.cnn import ECGCNN, ECGCNNConfig
from src.models.xgb import load_xgb, ManualCalibratedModel
from src.xai.gradcam import GradCAM
from src.xai.shap_xgb import explain_xgb, plot_shap_summary, plot_shap_waterfall

# Configuration
ASSETS_DIR = PROJECT_ROOT / "academic_assets"
ASSETS_DIR.mkdir(exist_ok=True)
LOGS_DIR = PROJECT_ROOT / "logs"

# Set style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["savefig.dpi"] = 300

def save_fig(fig, name):
    """Save figure with consistent naming and feedback."""
    path = ASSETS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"[Generated] {path.name}")
    plt.close(fig)

def generate_training_curves():
    """Generates training loss and accuracy curves for CNN."""
    print("\n--- Generating Training Curves ---")
    
    # Load CNN metrics
    cnn_metrics_path = LOGS_DIR / "cnn" / "metrics.json"
    if not cnn_metrics_path.exists():
        print(f"Skipping CNN curves: {cnn_metrics_path} not found")
        return

    with open(cnn_metrics_path) as f:
        data = json.load(f)
    
    if "history" not in data:
         print("Skipping CNN curves: 'history' key not found")
         return
         
    df = pd.DataFrame(data["history"])
    
    # 1. Loss Curve
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="epoch", y="train_loss", label="Train Loss", marker="o", ax=ax)
    sns.lineplot(data=df, x="epoch", y="val_loss", label="Val Loss", marker="o", ax=ax)
    ax.set_title("CNN Training History: Cross-Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    save_fig(fig, "fig_01_cnn_training_loss")

    # 2. AUC Curve
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="epoch", y="roc_auc", label="Val AUC", marker="o", color="green", ax=ax)
    ax.set_title("CNN Validation Performance: ROC-AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    save_fig(fig, "fig_02_cnn_validation_auc")

def generate_confusion_matrices():
    """Generates confusion matrices for Binary and Multiclass models."""
    print("\n--- Generating Confusion Matrices ---")
    
    # 1. Binary XGBoost
    xgb_metrics_path = LOGS_DIR / "xgb" / "metrics.json"
    if xgb_metrics_path.exists():
        with open(xgb_metrics_path) as f:
            data = json.load(f)
        
        # Prefer test set, fall back to val
        if "test" in data:
            cm = np.array(data["test"]["confusion_matrix"])
            split_name = "Test"
        else:
            cm = np.array(data["val"]["confusion_matrix"])
            split_name = "Validation"
            
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORM", "MI"])
        disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
        ax.grid(False)
        ax.set_title(f"Binary Classification Confusion Matrix ({split_name} Set)")
        save_fig(fig, "fig_03_binary_confusion_matrix")

def generate_xai_visualizations():
    """Generates XAI plots (Grad-CAM, SHAP) for a sample MI case."""
    print("\n--- Generating XAI Visualizations ---")
    
    # Setup
    config = get_default_config(PROJECT_ROOT)
    
    # Load Metadata & Create Dataset manually
    # Load Metadata & Create Dataset manually
    try:
        from src.data.loader import load_scp_statements # Import here to avoid top-level clutter if not needed elsewhere
        
        scp_df = load_scp_statements(config.scp_statements_path)
        df = load_ptbxl_metadata(config.metadata_path)
        df = add_binary_mi_labels(df, scp_df)
        test_df = df[df.strat_fold == 10]
        
        # Use SignalDataset
        dataset = SignalDataset(
            test_df,
            config.data_root,  # Fixed: Use data_root because filenames already contain 'records100/'
            label_column="label_mi_norm",
            cache_dir=None,
            use_cache=False
        )
    except Exception as e:
        print(f"Skipping XAI: Could not load dataset ({e})")
        return

    # Find a strong MI sample
    # SignalDataset doesn't expose samples directly like PyTorch dataset list
    # We iterate until we find one
    target_idx = -1
    for i in range(len(dataset)):
        # SignalDataset returns (signal, label, ecg_id)
        # Note: label might be returned as float or int
        _, label, _ = dataset[i]
        if label == 1:
             target_idx = i
             break
             
    if target_idx == -1:
        print("Skipping XAI: No MI samples found in test set")
        return
        
    signal, label, ecg_id = dataset[target_idx]
    signal_tensor = torch.tensor(signal).float().unsqueeze(0) # (1, 1000, 12) -> we need (1, 12, 1000)
    signal_tensor = signal_tensor.transpose(1, 2) # (1, 12, 1000)
    
    # Load Models
    cnn_path = PROJECT_ROOT / "checkpoints" / "ecgcnn.pt"
    xgb_dir = LOGS_DIR / "xgb"
    xgb_path = xgb_dir / "xgb_model.json"
    
    if not (cnn_path.exists() and xgb_dir.exists()):
        print("Skipping XAI: Models not found")
        return
        
    # CNN
    cnn_config = ECGCNNConfig()
    model = ECGCNN(cnn_config, num_classes=1)
    state = torch.load(cnn_path, map_location="cpu")
    if "model_state_dict" in state:
        raw_state = state["model_state_dict"]
        # Remap keys: 0. -> backbone., 1. -> head.
        new_state = {}
        for k, v in raw_state.items():
            if k.startswith("0."):
                new_key = k.replace("0.", "backbone.", 1)
            elif k.startswith("1."):
                new_key = k.replace("1.", "head.", 1)
            else:
                new_key = k
            new_state[new_key] = v
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state)
    model.eval()
    
    # XGB - Load manually to handle calibrated wrapper
    xgb_model = None
    calibrated_path = xgb_dir / "xgb_calibrated.joblib"
    if calibrated_path.exists():
        import joblib
        xgb_model = joblib.load(calibrated_path)
    elif xgb_path.exists():
        xgb_model = load_xgb(xgb_path)
        
    if xgb_model is None:
         print("Skipping XAI: Could not load XGBoost model")
         return
    
    # 1. Grad-CAM
    # Determine target layer (last conv block)
    target_layer = model.backbone.features[6] 
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(signal_tensor, class_index=0) 
    
    # Plot Signal + GradCAM
    fig, ax = plt.subplots(figsize=(12, 4))
    lead_idx = 1 # Lead II
    # signal is (1000, 12)
    ax.plot(signal[:, lead_idx], 'b', label="Lead II Signal", alpha=0.7)
    
    # Overlay heatmap
    # Cam is likely (1000,)
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    # Ensure (1, 1000) for imshow
    if cam_normalized.ndim == 1:
        cam_normalized = cam_normalized[np.newaxis, :]
    elif cam_normalized.ndim == 3:
         cam_normalized = cam_normalized.squeeze()
         if cam_normalized.ndim == 1:
              cam_normalized = cam_normalized[np.newaxis, :]
              
    extent = [0, 1000, signal[:, lead_idx].min(), signal[:, lead_idx].max()]
    ax.imshow(cam_normalized, cmap="Reds", aspect="auto", alpha=0.5, extent=extent)
    
    ax.set_title(f"Grad-CAM Attribution (MI Sample, Lead II)")
    ax.set_xlabel("Time (samples)")
    save_fig(fig, "fig_04_gradcam_sample")
    
    # 2. SHAP
    # Get embeddings
    with torch.no_grad():
        embeddings = model.backbone(signal_tensor).numpy()
        
    explanation = explain_xgb(xgb_model, embeddings)
    shap_fig = plot_shap_waterfall(
        explanation["shap_values"],
        explanation["base_value"],
        sample_idx=0,
        features=embeddings,
        title="SHAP Waterfall (Local Explanation)"
    )
    save_fig(shap_fig, "fig_05_shap_waterfall")
    
    # 3. SHAP Summary (Simulate using random noise if full test set is too slow, 
    # but here we use single sample for demo, ideally run on batch)
    # Creating a dummy summary plot with 1 sample looks bad, skipping for now or 
    # we could run on more samples.
    
    print("XAI generation complete.")

def generate_multiclass_plots():
    """Generates comparison plots for Multiclass (Superclass) models."""
    print("\n--- Generating Multiclass Visualizations ---")
    
    # Load Metrics
    cnn_path = LOGS_DIR / "superclass_cnn" / "training_results.json"
    xgb_path = LOGS_DIR / "xgb_superclass" / "training_results.json"
    
    metrics = []
    
    if cnn_path.exists():
        with open(cnn_path) as f:
            data = json.load(f)
            if "test_metrics" in data and "per_class" in data["test_metrics"]:
                for cls, val in data["test_metrics"]["per_class"].items():
                    metrics.append({
                        "Model": "CNN",
                        "Class": cls,
                        "AUC": val.get("auroc", 0),
                        "F1": val.get("f1", 0)
                    })
                    
    if xgb_path.exists():
        with open(xgb_path) as f:
            data = json.load(f)
            if "test_metrics" in data and "per_class" in data["test_metrics"]:
                for cls, val in data["test_metrics"]["per_class"].items():
                    metrics.append({
                        "Model": "XGBoost",
                        "Class": cls,
                        "AUC": val.get("auroc", 0),
                        "F1": val.get("f1", 0)
                    })
    
    if not metrics:
        print("Skipping Multiclass: No metrics found.")
        return

    df = pd.DataFrame(metrics)
    
    # Plot Grouped Bar Chart for AUC
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Class", y="AUC", hue="Model", palette="viridis", ax=ax)
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Multiclass Model Comparison (AUC per Class)")
    save_fig(fig, "fig_06_multiclass_auc_comparison")

    # Plot Grouped Bar Chart for F1
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="Class", y="F1", hue="Model", palette="magma", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multiclass Model Comparison (F1-Score per Class)")
    save_fig(fig, "fig_07_multiclass_f1_comparison")

def generate_localization_visualizations():
    """Generates qualitative Grad-CAM for MI Localization."""
    print("\n--- Generating Localization Visualizations ---")
    
    # 1. Load Data (Reuse setup from XAI)
    config = get_default_config(PROJECT_ROOT)
    try:
        from src.data.loader import load_scp_statements
        scp_df = load_scp_statements(config.scp_statements_path)
        df = load_ptbxl_metadata(config.metadata_path)
        df = add_binary_mi_labels(df, scp_df)
        test_df = df[df.strat_fold == 10]
        
        dataset = SignalDataset(
            test_df,
            config.data_root,
            label_column="label_mi_norm",
            cache_dir=None,
            use_cache=False
        )
    except Exception:
        return

    # 2. Load Localization Model
    loc_path = PROJECT_ROOT / "checkpoints" / "ecgcnn_localization.pt"
    if not loc_path.exists():
        print(f"Skipping Localization: {loc_path} not found")
        return
        
    # Localization has 5 classes: AMI, ASMI, ALMI, IMI, LMI
    from src.data.mi_localization import MI_LOCALIZATION_REGIONS
    num_classes = len(MI_LOCALIZATION_REGIONS) # 5
    
    cnn_config = ECGCNNConfig()
    model = ECGCNN(cnn_config, num_classes=num_classes)
    state = torch.load(loc_path, map_location="cpu")
    
    # Handle state dict wrapped/unwrapped
    if "model_state_dict" in state:
        raw_state = state["model_state_dict"]
        new_state = {}
        for k, v in raw_state.items():
            if k.startswith("0."): new_key = k.replace("0.", "backbone.", 1)
            elif k.startswith("1."): new_key = k.replace("1.", "head.", 1)
            else: new_key = k
            new_state[new_key] = v
        model.load_state_dict(new_state)
    else:
        model.load_state_dict(state)
    model.eval()

    # 3. Find an MI sample (Inferior MI preferred for clear visualization)
    target_idx = -1
    for i in range(len(dataset)):
        _, label, ecg_id = dataset[i]
        if label == 1:
             # Check if it is IMI
             scp = df.loc[ecg_id, "scp_codes"]
             if "IMI" in scp and scp["IMI"] > 50:
                 target_idx = i
                 break
    
    if target_idx == -1: return

    signal, _, _ = dataset[target_idx]
    signal_tensor = torch.tensor(signal).float().unsqueeze(0).transpose(1, 2) # (1, 12, 1000)

    # 4. Grad-CAM for IMI (Index 3 in list relative to MI_LOCALIZATION_REGIONS?)
    # MI_LOCALIZATION_REGIONS = ["AMI", "ASMI", "ALMI", "IMI", "LMI"]
    # IMI is index 3
    target_class_idx = MI_LOCALIZATION_REGIONS.index("IMI")
    
    target_layer = model.backbone.features[6]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(signal_tensor, class_index=target_class_idx)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    lead_idx = 1 # Lead II (sensitive to IMI)
    ax.plot(signal[:, lead_idx], 'g', label="Lead II (IMI Sensitive)", alpha=0.8)
    
    cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    if cam_normalized.ndim == 1: cam_normalized = cam_normalized[np.newaxis, :]
    
    extent = [0, 1000, signal[:, lead_idx].min(), signal[:, lead_idx].max()]
    ax.imshow(cam_normalized, cmap="Oranges", aspect="auto", alpha=0.5, extent=extent)
    ax.set_title("Localization Grad-CAM (Target: Inferior MI)")
    save_fig(fig, "fig_08_localization_gradcam_imi")


    save_fig(fig, "fig_08_localization_gradcam_imi")

def generate_multiclass_inference_assets():
    """Run inference on full test set to generate Multiclass Confusion Matrix."""
    print("\n--- Generating Multiclass Inference Assets ---")
    
    # 1. Load Data
    config = get_default_config(PROJECT_ROOT)
    try:
        from src.data.loader import load_scp_statements
        from src.data.labels import add_5class_labels
        
        scp_df = load_scp_statements(config.scp_statements_path)
        df = load_ptbxl_metadata(config.metadata_path)
        
        # Get Single Label for Confusion Matrix (Primary Label)
        df = add_5class_labels(df, scp_df, multi_label=False)
        test_df = df[df.strat_fold == 10]
        
        # Load Raw Signals
        dataset = SignalDataset(
            test_df,
            config.data_root,
            label_column="label_5class",
            cache_dir=None,
            use_cache=False
        )
    except Exception as e:
        print(f"Skipping Multiclass Inference: Data load error ({e})")
        return

    # 2. Load Model
    cnn_path = PROJECT_ROOT / "checkpoints" / "ecgcnn_superclass.pt"
    if not cnn_path.exists():
        print("Skipping Multiclass Inference: Model not found")
        return
        
    # Superclass config: 5 classes
    from src.config import DIAGNOSTIC_SUPERCLASSES
    # DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
    
    # Check model architecture in file to be safe, usually MultiLabelECGCNN or ECGCNN
    # run_inference_superclass uses MultiLabelECGCNN
    from src.pipeline.train_superclass_cnn import MultiLabelECGCNN
    
    # Try loading
    try:
        # Load state
        state = torch.load(cnn_path, map_location="cpu")
        model = MultiLabelECGCNN(ECGCNNConfig()) # Default config
        
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
            
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except Exception as e:
        print(f"Skipping Multiclass Inference: Model load error ({e})")
        return

    # 3. Run Batch Inference
    y_true = []
    y_pred = []
    
    print(f"Running inference on {len(dataset)} samples...")
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for signals, labels, _ in dataloader:
            signals = signals.float().to(device)
            # Ensure (Batch, Channels, Time)
            if signals.shape[1] == 1000: signals = signals.transpose(1, 2)
            
            logits = model(signals)
            probs = torch.sigmoid(logits)
            
            # Argmax for primary label prediction
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
            
    # 4. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(DIAGNOSTIC_SUPERCLASSES)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DIAGNOSTIC_SUPERCLASSES)
    disp.plot(cmap="Purples", ax=ax, values_format="d", colorbar=False)
    ax.grid(False)
    ax.set_title("Multiclass Confusion Matrix (Primary Label)")
    save_fig(fig, "fig_09_multiclass_confusion_matrix")

    save_fig(fig, "fig_09_multiclass_confusion_matrix")

def generate_localization_metrics():
    """Run inference for Localization model and plot per-region metrics."""
    print("\n--- Generating Localization Metrics ---")
    
    # 1. Load Data
    config = get_default_config(PROJECT_ROOT)
    try:
        from src.data.loader import load_scp_statements
        from src.data.mi_localization import add_mi_localization_labels, MI_LOCALIZATION_REGIONS
        from src.data.labels import add_binary_mi_labels
        
        scp_df = load_scp_statements(config.scp_statements_path)
        df = load_ptbxl_metadata(config.metadata_path)
        
        # Filter for MI samples first to reduce processing or process all then filter
        # We need ground truth localization labels
        df = add_binary_mi_labels(df, scp_df)
        df = add_mi_localization_labels(df)
        
        # Test set and ONLY MI samples (Localization is only relevant for MI)
        test_df = df[(df.strat_fold == 10) & (df.label_mi_norm == 1)].copy()
        
        if len(test_df) == 0:
            print("Skipping Localization Metrics: No MI samples in test set.")
            return

        # Load Raw Signals
        # We need to manually construct dataset for y_loc
        dataset = SignalDataset(
            test_df,
            config.data_root,
            label_column="y_loc", # This will return the localization vector
            cache_dir=None,
            use_cache=False
        )
    except Exception as e:
        print(f"Skipping Localization Metrics: Data load error ({e})")
        return

    # 2. Load Model
    loc_path = PROJECT_ROOT / "checkpoints" / "ecgcnn_localization.pt"
    if not loc_path.exists():
        print("Skipping Localization Metrics: Model not found")
        return
        
    try:
        # Load state
        state = torch.load(loc_path, map_location="cpu")
        cnn_config = ECGCNNConfig()
        model = ECGCNN(cnn_config, num_classes=len(MI_LOCALIZATION_REGIONS))
        
        if "model_state_dict" in state:
            # Need remapping if key mismatch exists (likely yes if same as others)
            raw_state = state["model_state_dict"]
            new_state = {}
            for k, v in raw_state.items():
                if k.startswith("0."): new_key = k.replace("0.", "backbone.", 1)
                elif k.startswith("1."): new_key = k.replace("1.", "head.", 1)
                else: new_key = k
                new_state[new_key] = v
            model.load_state_dict(new_state)
        else:
            model.load_state_dict(state)
            
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except Exception as e:
        print(f"Skipping Localization Metrics: Model load error ({e})")
        return

    # 3. Run Inference
    y_true = []
    y_pred_probs = []
    
    print(f"Running localization inference on {len(dataset)} samples...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for signals, labels, _ in dataloader:
            signals = signals.float().to(device)
            if signals.shape[1] == 1000: signals = signals.transpose(1, 2)
            
            logits = model(signals)
            probs = torch.sigmoid(logits)
            
            y_pred_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.numpy())
            
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # 4. Calculate Metrics
    from sklearn.metrics import f1_score, accuracy_score
    
    metrics = []
    for i, region in enumerate(MI_LOCALIZATION_REGIONS):
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics.append({"Region": region, "Metric": "Accuracy", "Value": acc})
        metrics.append({"Region": region, "Metric": "F1-Score", "Value": f1})
        
    # 5. Plot
    df_metrics = pd.DataFrame(metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_metrics, x="Region", y="Value", hue="Metric", palette="Set2", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("MI Localization Performance per Region (Test Set)")
    save_fig(fig, "fig_10_localization_metrics_per_region")


def main():
    print("Starting Academic Asset Generation...")
    generate_training_curves()
    generate_confusion_matrices()
    generate_xai_visualizations()
    generate_multiclass_plots()
    generate_localization_visualizations()
    generate_multiclass_inference_assets()
    generate_localization_metrics()
    print("\nAll assets generated in 'academic_assets/'.")

if __name__ == "__main__":
    main()
