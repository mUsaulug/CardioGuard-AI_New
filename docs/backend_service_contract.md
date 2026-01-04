# Backend Service Contract (CardioGuard-AI)

## Endpoint
`POST /predict`

### Modes
- `binary`
- `multiclass`
- `localization`

## Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `file` | file | ✅ | ECG signal file (`.npz` or raw waveform). |
| `mode` | string | ✅ | `binary`, `multiclass`, or `localization`. |

### File formats
- **NPZ**: `signal` (or `signals`) array, shape `(T, 12)` or `(12, T)`.
- **Raw**: waveform file handled by backend ingestion (e.g., `.npy`/`.csv` or WFDB), converted to `(T, 12)`.

## Response

```json
{
  "mode": "binary",
  "prediction": {
    "label": "MI",
    "value": 1,
    "threshold": 0.5
  },
  "probabilities": {
    "cnn": 0.84,
    "xgb": 0.79,
    "ensemble": 0.81
  },
  "xai_images": {
    "gradcam": "https://<host>/reports/xai/inference_gradcam.png",
    "shap": "https://<host>/reports/xai/inference_shap_waterfall.png",
    "lead_attention": "https://<host>/reports/xai/inference_lead_attention.png"
  },
  "explanation_text": "Binary ensemble inference complete. CNN=0.840, XGB=0.790, ensemble=0.810, threshold=0.50."
}
```

### Response fields
- `prediction`: final label + threshold decision.
- `probabilities`: CNN, XGBoost, and ensemble probabilities.
- `xai_images`: **PNG links** to Grad-CAM, SHAP, and lead-attention artifacts.
- `explanation_text`: short human-readable summary.

## Artifact contract (stable paths)

```
checkpoints/ecgcnn.pt
logs/xgb/*
reports/xai/*
```

## Binary inference entrypoint

```bash
python -m src.pipeline.run_inference_binary \
  --input <path_to_ecg.npz> \
  --cnn-path checkpoints/ecgcnn.pt \
  --xgb-dir logs/xgb \
  --xgb-path logs/xgb/xgb_model.json
```
