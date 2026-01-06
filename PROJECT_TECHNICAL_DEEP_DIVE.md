# CardioGuard-AI: Technical Deep Dive & Code Reference

> **Level:** L3 (Code/Implementation)
> **Target Audience:** Backend Engineers, ML Engineers, Maintainers
> **Purpose:** Detailed documentation of internal logic, data structures, and hyperparameters.

---

## 1. Project Specifications

### 1.1. Technology Stack
*   **Language:** Python 3.10+
*   **Deep Learning:** PyTorch (v2.0+)
*   **Classic ML:** XGBoost, Scikit-Learn
*   **Signal Processing:** `wfdb`, `numpy`
*   **API:** FastAPI
*   **Device Support:** CPU (Inference Optimized), CUDA (Training)

### 1.2. Key Constants (`src/config.py`)
*   **`SAMPLING_RATE`:** 100 Hz (Downsampled from 500Hz original)
*   **`INPUT_SHAPE`:** `(12, 1000)` -> (Leads, Timesteps)
*   **`LEAD_NAMES`:** `['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']`
*   **`SUPERCLASS_LABELS`:** `['MI', 'STTC', 'CD', 'HYP', 'NORM']`
*   **`MI_LOCALIZATION_REGIONS`:** `['AMI', 'ASMI', 'ALMI', 'IMI', 'LMI']`

---

## 2. Information Flow Pipeline

### 2.1. Signal Preprocessing (`ensure_channel_first`)
The model expects strict `(Channels, Timesteps)` format. The `run_inference` pipeline enforces this logic:
1.  **Check dims:** If shape is `(1000, 12)`, transpose to `(12, 1000)`.
2.  **Normalization:** Standard Z-score normalization (Mean=0, Std=1) is applied during training via `SignalDataset` but **NOT** strictly enforced in inference (CNN is robust, but scaling `XGBoost` embeddings is critical).

### 2.2. The Backbone (`src/models/cnn.py`)
A custom **1D-ResNet** style backbone (`ECGBackbone`).
*   **Layer 1:** `Conv1d(12 -> 64, k=7)` + BN + ReLU + Dropout(0.3)
*   **Layer 2:** `Conv1d(64 -> 64, k=7)` + BN + ReLU + Dropout(0.3)
*   **Pooling:** `AdaptiveAvgPool1d(1)` -> Collapses time dimension (1000 -> 1).
*   **Output:** **64-dimensional embedding vector**.

---

## 3. Inference Logic (Detailed Trace)
File: `src/pipeline/run_inference_superclass.py`

### Step 1: Model Loading
*   **CNN:** Loads `ecgcnn_superclass.pt`. Map location ensures CPU compatibility.
*   **XGBoost:** Loads 4 discrete JSON models (One-Vs-Rest) + 4 `IsotonicRegression` calibrators + 1 `StandardScaler`.
*   **Localization:** Loads `ecgcnn_localization.pt` (Multi-label, 5 classes).

### Step 2: Prediction Flow
```python
# 1. Forward Pass CNN
logits = cnn_model(signal_tensor) 
cnn_probs = sigmoid(logits)

# 2. Extract Embeddings (Hook/Forward)
embeddings = cnn_model.backbone(signal_tensor) # Shape: (1, 64)

# 3. XGBoost Path
scaled_embeddings = scaler.transform(embeddings)
xgb_probs = {}
for class in ["MI", "STTC", "CD", "HYP"]:
    raw = xgb_models[class].predict_proba(scaled_embeddings)
    calibrated = calibrators[class].predict(raw)
    xgb_probs[class] = calibrated

# 4. Ensemble Fusion
final_prob[cls] = (0.5 * cnn_prob[cls]) + (0.5 * xgb_prob[cls])
```

### Step 3: Conditional Logic
*   **MI Detection:** If `final_prob["MI"] > threshold["MI"]`:
    *   **Trigger:** Run `localization_model(signal_tensor)`.
    *   **Output:** Dict of region probabilities (e.g., `{'IMI': 0.95, 'AMI': 0.12}`).

---

## 4. XAI (Explainable AI) Mechanics

### 4.1. Visual Evidence (Grad-CAM)
*   **Target Layer:** `cnn_model.backbone.features[-3]` (The last Convolutional layer before pooling).
*   **Method:** Gradients of the target class logit flow back to this layer to weigh the activation maps.
*   **Output:** A 1D heatmap (1000 points) showing temporal importance.

### 4.2. Statistical Evidence (SHAP)
*   **Explainer:** `shap.TreeExplainer` or `shap.LinearExplainer` (approximated for speed in production).
*   **Input:** The 64-dim embedding vector.
*   **Output:** Feature contribution scores (e.g., "Feature #32 increased MI risk by 10%").

### 4.3. Unified Synthesis (`src/xai/unified.py`)
*   **Conflict Detection:** Heuristic comparison.
    *   *If* Visual Model says "High Confidence" on Region A
    *   *AND* Clinical Model says "Low Confidence" based on features
    *   *THEN* Flag as "Incoherent".

---

## 5. Directory & File Reference

### `src/backend`
*   `main.py`: **[LEGACY]** FastAPI entry point. Needs update to support new Phase 7 logic.

### `src/config.py`
*   **Config Center:** Defines `MI_CODES`, `DIAGNOSTIC_PRIORITY`. This is the dictionary that maps raw `scp_statements` to `label_MI`.

### `src/data`
*   `loader.py`: `load_ptbxl_metadata()`, `load_scp_statements()`.
*   `labels_superclass.py`: `add_superclass_labels_derived()`. **CRITICAL:** Contains the mapping rules (e.g., "IMI" -> Class 1).
*   `mi_localization.py`: `add_mi_localization_labels()`. Logic to parse the SCP dictionary for region codes.

### `src/models`
*   `cnn.py`: `ECGCNN`, `MultiTaskECGCNN`.
*   `xgb.py`: Helper to load/save XGBoost models.
*   `trainer.py`: Shared training loops (`train_one_epoch`).

### `src/pipeline`
*   `train_superclass_cnn.py`: Training script for the main backbone.
*   `train_superclass_xgb_ovr.py`: Training script for the boosting layer.
*   `train_mi_localization.py`: **[NEW]** specialized trainer for MI regions.
*   `extract_superclass_features.py`: Batch processor to dump embeddings for XGBoost training.
*   `run_inference_superclass.py`: **[PRODUCTION]** The main inference script.

### `src/xai`
*   `gradcam.py`: Custom implementation for 1D signals.
*   `shap_ovr.py`: Wrapper for SHAP on OVR models.
*   `visualize.py`: Plotting utilities (`matplotlib`).
*   `sanity.py`: Unit tests for explanations (Model Parameter Randomization).

---

## 6. Training Hyperparameters (Reference)

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Epochs** | 30 | Early stopping usually triggers around 15-20. |
| **Batch Size** | 64 | Tuned for standard 8GB/16GB GPU memory. |
| **Learning Rate** | 1e-3 | Standard Adam default. |
| **Scheduler** | ReduceLROnPlateau | Factor 0.5, Patience 3. |
| **Loss** | BCEWithLogitsLoss | Supports multi-label. |
| **Pos_Weight** | Dynamic | `(Neg / Pos)` count ratio to handle class imbalance. |
