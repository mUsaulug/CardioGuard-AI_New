# CardioGuard-AI: Binary Model Analysis (Legacy Phase)

> **Context:** This document analyzes the **Binary Classification Phase** (MI vs. NORM) which preceded the current Multi-Label Superclass phase.
> **Status:** Archived / Legacy Support.
> **Key Goal:** Detect Myocardial Infarction (MI) vs. Normal/Other with high sensitivity.

---

## 1. System Architecture (Binary)

The binary phase utilized the same **Hybrid Ensemble** philosophy as the modern system, but simplified for a single-output task.

| Component | Model Type | Input | Output |
| :--- | :--- | :--- | :--- |
| **Stream A** | **Binary CNN** | Raw Signal (12, 1000) | Logit (Scalar) -> Sigmoid -> Probability |
| **Stream B** | **XGBoost (Binary)** | CNN Embeddings (Low-dim) | Probability (Calibrated) |
| **Ensemble** | **Weighted Average** | Probabilities | `P_final = alpha * P_cnn + (1-alpha) * P_xgb` |

### 1.1. Experimental "Bound" Localization
Unlike the Phase 7 region classification (Anterior, Inferior, etc.), the binary phase attempted a regression-based localization:
*   **Method:** Predicting `start_index` and `end_index` of the MI event within the signal.
*   **Loss:** `SmoothL1Loss` (Regression).
*   **Status:** This appears to be an experimental feature controlled by `--localization-column` in training.

---

## 2. Codebase Analysis (Binary Files)

### 📂 `src/pipeline`

#### 1. `run_inference_binary.py` (The Binary Engine)
*   **Function:** Performs end-to-end inference for the binary task.
*   **Key Logic:**
    1.  Loads `ecgcnn.pt` (Binary Mode).
    2.  Loads `xgb_model.json` + `xgb_scaler.joblib`.
    3.  **Ensemble:** Combines scores using a loaded `ensemble_config.json` (or default alpha=0.15).
    4.  **Thresholding:** Applies a threshold (default 0.5) to decide `MI` vs `NORM`.
    5.  **XAI:** Generates Grad-CAM and SHAP Waterfall plots.
*   **Output:** JSON payload with binary prediction, probabilities, and XAI image paths.

#### 2. `run_experiment.py` (The Trainer)
*   **Function:** Generic training script that handles both Binary and Multiclass tasks.
*   **Key Logic:**
    *   `--task binary`: Selects `BCEWithLogitsLoss`.
    *   `--strategy cnn`: Uses the standard CNN backbone.
    *   **Multitask:** If `--localization-column` is provided, adds a regression head for localization.

### 📂 `src/models`

#### 3. `trainer.py`
*   **Function:** Contains the training loop logic.
*   **Binary Specifics:**
    *   `train_one_epoch`: Explicitly checks `if task == "binary": criterion = nn.BCEWithLogitsLoss()`.
    *   `train_one_epoch_multitask`: Handles the combined `BCE + SmoothL1` loss for the experimental localization.

---

## 3. Key Differences: Binary vs. Superclass

| Feature | Binary Phase (Legacy) | Superclass Phase (Current) |
| :--- | :--- | :--- |
| **Classification** | Single-Label (MI or Not) | Multi-Label (MI, STTC, CD, HYP, NORM) |
| **Localization** | Regression (Start/End Time) | Classification (Anatomical Region) |
| **xgboost** | Single Binary Classifier | 4-OVR Classifiers (One-vs-Rest) |
| **Loss** | BCE | BCE (per class) |
| **Inference Script** | `run_inference_binary.py` | `run_inference_superclass.py` |

---

## 4. How to Run (Legacy)

To interact with the binary model (assuming checkpoints exist):

```bash
# Inference
python -m src.pipeline.run_inference_binary --input sample.npz --cnn-path checkpoints/ecgcnn.pt

# Training
python -m src.pipeline.run_experiment --task binary --epochs 10
```

## 5. Conclusion
The Binary Phase established the foundational **Hybrid Architecture** (CNN + XGBoost + XAI) that is currently used. The move to Superclass extended this foundation to support multi-label diagnosis and a more clinically relevant (anatomical) localization strategy.
