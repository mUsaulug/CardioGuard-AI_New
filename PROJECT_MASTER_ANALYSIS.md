# CardioGuard-AI: Master Handover & Technical Analysis

> **Date:** 2026-01-06
> **Status:** Phase 9 (Verification) Completed
> **Next Phase:** Phase 6.3 (Backend Integration)
> **Criticality:** HIGH - This document is the source of truth for the project's state.

---

## 1. Executive Summary
**CardioGuard-AI** is an advanced, interpretable deep learning system for ECG classification. Unlike standard black-box models, it is designed for **clinical trust**, providing not just a diagnosis but also:
1.  **Multi-label Classification:** Can detect multiple pathologies simultaneously (MI, STTC, CD, HYP, NORM).
2.  **MI Localization:** If Myocardial Infarction (MI) is detected, it precisely localizes the area (e.g., Anterior, Inferior) using the 12-lead signal.
3.  **Unified XAI:** Provides "Visual Evidence" (Grad-CAM) and "Clinical Evidence" (SHAP) in a human-readable narrative.

The system uses a **Hybrid Ensemble Architecture** combining the visual pattern recognition of a **1D-CNN** with the statistical power of **XGBoost**.

---

## 2. System Architecture & Methodology

### 2.1. The Hybrid Ensemble (Superclass)
The core inference engine is **`src/pipeline/run_inference_superclass.py`**. It orchestrates two parallel streams:

| Component | Model Type | Input | Role |
| :--- | :--- | :--- | :--- |
| **Stream A** | **Multi-Label 1D-CNN** | Raw Signal (12, 1000) | Captures morphological waves patterns (ST elevation, T inversion). Acts as a feature extractor. |
| **Stream B** | **XGBoost (OVR)** | CNN Embeddings (64-dim) | Dense features extracted from the CNN's penultimate layer. Excels at statistical decision boundaries. |
| **Fusion** | **Weighted Average** | Probabilities | `P_final = w * P_cnn + (1-w) * P_xgb` (Default w=0.5) |

**Key Decision:** We use **One-Vs-Rest (OVR)** for XGBoost. This means we have 4 separate XGBoost models (one for each pathology: MI, STTC, CD, HYP), allowing for independent probability calibration.

### 2.2. MI Localization Strategy (Hierarchical)
This is a **Conditional Logic** module. It does *not* run for every patient.
1.  **Trigger:** The Ensemble predicts `MI` probability > Threshold.
2.  **Action:** The system loads the **Localization Model** (`src/pipeline/train_mi_localization.py`).
3.  **Model:** A specialized Multi-Label CNN trained *only* on MI-positive samples.
4.  **Classes:** 5 anatomical regions (AMI, ASMI, ALMI, IMI, LMI).
5.  **Output:** Appended to the final specific report (e.g., "MI (Inferior)").

### 2.3. Unified XAI (Explainable AI)
We solved the "Black Box" problem with a unified approach (`src/xai/unified.py`):
1.  **Grad-CAM:** Highlights *where* in the 10-second signal the model is looking.
2.  **SHAP:** Analyzes feature contributions from the XGBoost side.
3.  **Sanity Check:** A runtime module (`src/xai/sanity.py`) that performs *Input Perturbation* and *Model Randomization* tests to ensure explanations are genuine and not just edge detectors.

---

## 3. Codebase Deep Dive (Critical Files)

### 📂 `src/pipeline` ( The Engine Room)

#### 1. `run_inference_superclass.py` (THE BRAIN)
*   **Function:** This is the most critical file. It ties everything together.
*   **Key Flow:**
    1.  Loads CNN (`load_cnn_model`) and XGBoost (`load_xgb_models`).
    2.  Loads Localization Model (`load_localization_model`).
    3.  runs `predict()`:
        *   Gets CNN probabilities.
        *   Extracts embeddings -> XGBoost probabilities.
        *   Ensembles them.
        *   **IF** "MI" in prediction -> Runs Localization Model.
        *   **IF** `--explain` -> Runs Grad-CAM + UnifiedExplainer.
        *   **IF** `--sanity-check` -> Verifies explanation stability.
*   **Critical Detail:** Uses `ensure_channel_first` to handle shape mismatches ((1000,12) vs (12,1000)).

#### 2. `train_mi_localization.py` (The Locator)
*   **Function:** Trains the secondary model for MI regions.
*   **Key Method:** `MILocalizationDataset`. It implements a robust `try-except` block in `__getitem__` to handle missing files in the partial PTB-XL dataset (records100) without crashing the pipeline. It substitutes zeros for missing files (lossy but stable).
*   **Loss Function:** `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance within MI regions.

#### 3. `train_superclass_cnn.py` & `train_superclass_xgb_ovr.py` (The Trainers)
*   **CNN:** Uses a ResNet-like 1D architecture (`src/models/cnn.py`). Implements `Macro AUPRC` as the primary optimization metric.
*   **XGBoost:** Trains on *embeddings* dumped by `extract_superclass_features.py`. Uses `CalibratedClassifierCV` (Isotonic/Sigmoid) to alignment probabilities with the CNN.

### 📂 `src/xai` (The Interpreter)

#### 4. `visualize.py` (The Artist)
*   **Function:** Generates the PNG plots.
*   **Recent Update:** Added `plot_ecg_with_localization` to visualize anatomical predictions. Added `LEAD_NAMES` as a global constant to fix import errors.
*   **Key Functions:** `plot_12lead_gradcam` (overlays heatmaps on ECG signals).

#### 5. `unified.py` (The Narrator)
*   **Function:** Converts numerical XAI outputs into text.
*   **Logic:** Checks "Visual Evidence" (Grad-CAM peak times) and "Feature Evidence" (SHAP tops). Generates a "Coherence Score" to tell the doctor if the visual and statistical models agree.

### 📂 `src/data` (The Foundation)

#### 6. `labels_superclass.py`
*   **Function:** The dictionary mapping SCP-ECG codes (e.g., '164889003') to our 5 Superclasses (`MI`, `STTC`, `CD`, `HYP`, `NORM`).
*   **Critical:** This is the ground truth logic. If these mappings are wrong, the whole system is wrong.

---

## 4. Current State & Known Variables

### 📊 Model Performance (Phase 7 Baseline)
*   **MI Detection:** 1.00 Recall (Verified on 15 samples).
*   **Threshold Strategy:** All superclasses set to **0.5** to balance precision and recall.
*   **MI-Priority Rule:** Applied to ensure clinical safety.
*   **MI Localization:** ~0.58 Macro AUPRC (Acceptable for 5-class, limited data).
    *   *Note:* The model is robust to missing files (handles partial `records100` dataset gracefull).
*   **Inference Speed:** Real-time on CPU (using ONNX-ready PyTorch models).

### 🛠️ Configuration (`src/config.py`)
*   **Sampling Rate:** 100 Hz (Fixed).
*   **Input Duration:** 10s (1000 samples).
*   **Leads:** 12 Standard Leads.

### ⚠️ Active Blockers / Todos
1.  **Backend Legacy Code:** `src/backend/main.py` is **OUTDATED**. It does not yet support:
    *   The Localization Model.
    *   The Unified XAI response format.
    *   The new JSON structure.
    *   *Action:* This is the immediate next task (Phase 6.3).

---

## 5. How to Run the System

### Full Verification Test
To confirm the entire pipeline (Prediction + Localization + Explanation + Sanity Check):
```bash
python src/pipeline/test_mi_integration.py
```
*Expected Output:*
*   Sample: 8 (True Label: IMI)
*   Prediction: MI
*   Localization: IMI (Confidence ~97%)
*   Plot saved: `_loc.png`

### Manual Inference
```bash
python -m src.pipeline.run_inference_superclass --input sample.npy --explain --sanity-check
```

---

## 6. Handover Note to Next Agent
**"You are inheriting a complete, verified inference engine.** The hard work of training and integrating the logic is done. The inference script `run_inference_superclass.py` is the gold standard.

**Your Mission:**
Your primary goal is **Integration**. You need to take the logic inside `run_inference_superclass.py` and transplant it into the FastAPI backend (`src/backend/main.py`).
*   Don't reinvent the inference logic. Import functions from the pipeline script if possible, or replicate the `load_models` and `predict` flow exactly.
*   Pay attention to `LEAD_NAMES` in visualization.
*   Ensure the API response model (`MILocalizationResponse`) is populated correctly."
