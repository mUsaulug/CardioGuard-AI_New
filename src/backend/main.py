"""
CardioGuard-AI FastAPI Backend.

REST API for multi-label superclass ECG prediction.

Endpoints:
- POST /predict/superclass - Multi-label prediction
- POST /predict/mi-localization - MI localization (if MI detected)
- GET /health - Health check
- GET /ready - Readiness check

Usage:
    uvicorn src.backend.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionProbabilities(BaseModel):
    """Probabilities for each class."""
    MI: float = Field(..., description="MI probability")
    STTC: float = Field(..., description="STTC probability")
    CD: float = Field(..., description="CD probability")
    HYP: float = Field(..., description="HYP probability")
    NORM: float = Field(..., description="Derived NORM probability")


class PrimaryPrediction(BaseModel):
    """Primary (single) label prediction."""
    label: str = Field(..., description="Primary predicted label")
    confidence: float = Field(..., description="Confidence score")
    rule: str = Field(default="MI-first-then-priority", description="Selection rule")


class SourceProbabilities(BaseModel):
    """Probabilities from each model source."""
    cnn: Dict[str, float] = Field(..., description="CNN probabilities")
    xgb: Optional[Dict[str, float]] = Field(None, description="XGBoost probabilities")
    ensemble: Dict[str, float] = Field(..., description="Ensemble probabilities")


class VersionInfo(BaseModel):
    """Model version information."""
    model_hash: str = Field(..., description="Hash of model checkpoint")
    threshold_hash: str = Field(..., description="Hash of threshold config")
    api_version: str = Field(default="1.0.0", description="API version")
    timestamp: str = Field(..., description="Prediction timestamp")


class SuperclassPredictionResponse(BaseModel):
    """Full superclass prediction response."""
    mode: str = Field(default="multilabel-superclass", description="Prediction mode")
    probabilities: PredictionProbabilities
    predicted_labels: List[str] = Field(..., description="Labels exceeding threshold")
    thresholds: Dict[str, float] = Field(..., description="Per-class thresholds")
    primary: PrimaryPrediction
    sources: SourceProbabilities
    versions: VersionInfo


class MILocalizationResponse(BaseModel):
    """MI localization prediction response."""
    mi_detected: bool = Field(..., description="Whether MI was detected")
    regions: List[str] = Field(default=[], description="Predicted MI regions")
    probabilities: Dict[str, float] = Field(default={}, description="Per-region probabilities")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    models_loaded: Dict[str, bool]
    message: str


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state for loaded models."""
    
    def __init__(self):
        self.cnn_model = None
        self.xgb_models = {}
        self.calibrators = {}
        self.scaler = None
        self.thresholds = {}
        self.norm_stats = None
        self.model_hash = ""
        self.threshold_hash = ""
        self.loaded = False
    
    def load_models(
        self,
        cnn_checkpoint: Path = Path("checkpoints/ecgcnn_superclass.pt"),
        xgb_dir: Path = Path("logs/xgb_superclass"),
        thresholds_path: Path = Path("artifacts/thresholds_superclass.json"),
    ):
        """Load all models."""
        import torch
        import joblib
        from xgboost import XGBClassifier
        
        device = torch.device("cpu")  # Use CPU for API serving
        
        # Load CNN
        if cnn_checkpoint.exists():
            from src.pipeline.training.train_superclass_cnn import MultiLabelECGCNN
            from src.models.cnn import ECGCNNConfig
            
            checkpoint = torch.load(cnn_checkpoint, map_location=device)
            config = ECGCNNConfig()
            self.cnn_model = MultiLabelECGCNN(config)
            self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
            self.cnn_model.eval()
            
            # Compute hash
            with open(cnn_checkpoint, "rb") as f:
                self.model_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        # Load XGBoost models
        if xgb_dir.exists():
            for cls in ["MI", "STTC", "CD", "HYP"]:
                model_path = xgb_dir / cls / "xgb_model.json"
                if model_path.exists():
                    model = XGBClassifier()
                    model.load_model(model_path)
                    self.xgb_models[cls] = model
                
                calibrator_path = xgb_dir / cls / "calibrator.joblib"
                if calibrator_path.exists():
                    self.calibrators[cls] = joblib.load(calibrator_path)
            
            scaler_path = xgb_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
        
        # Load thresholds
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                data = json.load(f)
            self.thresholds = data.get("thresholds", {})
            
            with open(thresholds_path, "rb") as f:
                self.threshold_hash = hashlib.md5(f.read()).hexdigest()[:8]
        else:
            # Default thresholds
            self.thresholds = {"MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
            self.threshold_hash = "default"
        
        self.loaded = True
        print(f"Models loaded: CNN={self.cnn_model is not None}, XGB={len(self.xgb_models)} models")


# Global state
state = AppState()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="CardioGuard-AI",
    description="Multi-label ECG Classification API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    print("Loading models...")
    try:
        state.load_models()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("API will start but predictions will fail until models are loaded.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Readiness check - are models loaded?"""
    models_status = {
        "cnn": state.cnn_model is not None,
        "xgb_MI": "MI" in state.xgb_models,
        "xgb_STTC": "STTC" in state.xgb_models,
        "xgb_CD": "CD" in state.xgb_models,
        "xgb_HYP": "HYP" in state.xgb_models,
        "thresholds": len(state.thresholds) > 0,
    }
    
    ready = all(models_status.values())
    
    return ReadyResponse(
        ready=ready,
        models_loaded=models_status,
        message="All models loaded" if ready else "Some models not loaded",
    )


def parse_ecg_file(file_content: bytes, filename: str) -> np.ndarray:
    """Parse uploaded ECG file."""
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
        tmp.write(file_content)
        tmp_path = Path(tmp.name)
    
    try:
        if filename.endswith(".npz"):
            data = np.load(tmp_path)
            if "signal" in data:
                signal = data["signal"]
            elif "X" in data:
                signal = data["X"]
            else:
                signal = data[list(data.keys())[0]]
        elif filename.endswith(".npy"):
            signal = np.load(tmp_path)
        else:
            raise HTTPException(400, f"Unsupported file format: {filename}")
    finally:
        tmp_path.unlink()
    
    # Ensure (channels, timesteps) format
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if signal.shape[0] != 12:
        if signal.shape[1] == 12:
            signal = signal.T
        elif signal.shape[0] > signal.shape[1]:
            signal = signal.T
    
    return signal.astype(np.float32)


def get_primary_label(probs: Dict[str, float]) -> tuple:
    """Determine primary label using MI-first rule."""
    thresholds = state.thresholds
    
    if probs.get("MI", 0) >= thresholds.get("MI", 0.5):
        return "MI", probs["MI"]
    
    for cls in ["STTC", "CD", "HYP"]:
        if probs.get(cls, 0) >= thresholds.get(cls, 0.5):
            return cls, probs[cls]
    
    # NORM
    max_pathology = max(probs.get(c, 0) for c in ["MI", "STTC", "CD", "HYP"])
    return "NORM", 1.0 - max_pathology


@app.post("/predict/superclass", response_model=SuperclassPredictionResponse)
async def predict_superclass(
    file: UploadFile = File(...),
    ensemble_weight: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Multi-label superclass prediction.
    
    Accepts ECG signal file (.npz or .npy format).
    Returns multi-label probabilities and primary label.
    """
    import torch
    
    if not state.loaded:
        raise HTTPException(503, "Models not loaded")
    
    if state.cnn_model is None:
        raise HTTPException(503, "CNN model not loaded")
    
    # Validate file size (10MB max)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    
    # Parse file
    try:
        signal = parse_ecg_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    
    # CNN prediction
    device = torch.device("cpu")
    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0)
        cnn_logits = state.cnn_model(signal_tensor)
        cnn_probs = torch.sigmoid(cnn_logits).numpy()[0]
    
    cnn_probs_dict = {
        "MI": float(cnn_probs[0]),
        "STTC": float(cnn_probs[1]),
        "CD": float(cnn_probs[2]),
        "HYP": float(cnn_probs[3]),
    }
    
    # XGBoost prediction
    xgb_probs_dict = {}
    if state.xgb_models:
        embeddings = state.cnn_model.backbone(signal_tensor).numpy()
        if state.scaler:
            embeddings = state.scaler.transform(embeddings)
        
        for cls in ["MI", "STTC", "CD", "HYP"]:
            if cls in state.xgb_models:
                raw_prob = state.xgb_models[cls].predict_proba(embeddings)[0, 1]
                if cls in state.calibrators:
                    prob = state.calibrators[cls].predict_proba([[raw_prob]])[0, 1]
                else:
                    prob = raw_prob
                xgb_probs_dict[cls] = float(prob)
    
    # Ensemble
    w = ensemble_weight
    ensemble_probs = {}
    for cls in ["MI", "STTC", "CD", "HYP"]:
        cnn_p = cnn_probs_dict[cls]
        xgb_p = xgb_probs_dict.get(cls, cnn_p)
        ensemble_probs[cls] = w * cnn_p + (1 - w) * xgb_p
    
    # NORM (derived)
    norm_prob = 1.0 - max(ensemble_probs.values())
    
    # Predicted labels
    predicted_labels = [
        cls for cls in ["MI", "STTC", "CD", "HYP"]
        if ensemble_probs[cls] >= state.thresholds.get(cls, 0.5)
    ]
    if not predicted_labels:
        predicted_labels = ["NORM"]
    
    # Primary label
    primary_label, primary_conf = get_primary_label(ensemble_probs)
    
    return SuperclassPredictionResponse(
        mode="multilabel-superclass",
        probabilities=PredictionProbabilities(
            **ensemble_probs,
            NORM=norm_prob,
        ),
        predicted_labels=predicted_labels,
        thresholds=state.thresholds,
        primary=PrimaryPrediction(
            label=primary_label,
            confidence=primary_conf,
        ),
        sources=SourceProbabilities(
            cnn=cnn_probs_dict,
            xgb=xgb_probs_dict if xgb_probs_dict else None,
            ensemble=ensemble_probs,
        ),
        versions=VersionInfo(
            model_hash=state.model_hash,
            threshold_hash=state.threshold_hash,
            timestamp=datetime.utcnow().isoformat(),
        ),
    )


@app.post("/predict/mi-localization", response_model=MILocalizationResponse)
async def predict_mi_localization(
    file: UploadFile = File(...),
):
    """
    MI localization prediction.
    
    Only returns localization if MI is detected above threshold.
    """
    # TODO: Implement after MI localization model is trained
    raise HTTPException(501, "MI localization not yet implemented")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
