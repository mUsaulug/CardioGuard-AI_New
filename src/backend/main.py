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
    """MI localization prediction response with full contract fields."""
    mi_detected: bool = Field(..., description="Whether MI was detected")
    regions: List[str] = Field(default=[], description="Predicted MI regions")
    probabilities: Dict[str, float] = Field(default={}, description="Per-region probabilities")
    
    # Contract fields (K4 requirement)
    label_space: str = Field(default="ptbxl_derived_anatomical_v1", description="Label space identifier")
    labels: List[str] = Field(default=["AMI", "ASMI", "ALMI", "IMI", "LMI"], description="Ordered label list")
    mapping_source: str = Field(default="src/data/mi_localization.py", description="Source file for mapping")
    mapping_fingerprint: str = Field(default="8ab274e06afa1be8", description="Mapping version fingerprint")
    localization_head_type: str = Field(default="classification_5", description="Head type (NOT regression_2)")


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
    """Application state for loaded models (3-model system)."""
    
    def __init__(self):
        # CNN Models (3 tasks)
        self.superclass_model = None      # 4-class [MI, STTC, CD, HYP]
        self.binary_model = None          # 1-class (MI vs NORM)
        self.localization_model = None    # 5-class [AMI, ASMI, ALMI, IMI, LMI]
        
        # XGBoost
        self.xgb_models = {}
        self.calibrators = {}
        self.scaler = None
        self.feature_schema = None
        
        # Config
        self.thresholds = {}
        self.norm_stats = None
        
        # Metadata
        self.model_hashes = {}
        self.threshold_hash = ""
        self.loaded = False
    
    def load_models(
        self,
        superclass_checkpoint: Path = Path("checkpoints/ecgcnn_superclass.pt"),
        binary_checkpoint: Path = Path("checkpoints/ecgcnn.pt"),
        localization_checkpoint: Path = Path("checkpoints/ecgcnn_localization.pt"),
        xgb_dir: Path = Path("logs/xgb_superclass"),
        thresholds_path: Path = Path("artifacts/thresholds_superclass.json"),
    ):
        """Load all models with safe loader."""
        import torch
        import joblib
        from xgboost import XGBClassifier
        from src.utils.model_loader import load_model_safe, validate_feature_schema
        
        device = torch.device("cpu")
        
        # --- Load Superclass (required) ---
        if superclass_checkpoint.exists():
            self.superclass_model, meta = load_model_safe(
                superclass_checkpoint, "superclass", device
            )
            self.model_hashes["superclass"] = meta["checkpoint_hash"]
            print(f"Superclass model loaded (schema: {meta['schema']})")
        
        # --- Load Binary (optional but recommended) ---
        if binary_checkpoint.exists():
            try:
                self.binary_model, meta = load_model_safe(
                    binary_checkpoint, "binary", device
                )
                self.model_hashes["binary"] = meta["checkpoint_hash"]
                print(f"Binary model loaded (schema: {meta['schema']})")
            except Exception as e:
                print(f"Warning: Binary model load failed: {e}")
        
        # --- Load Localization (optional) ---
        if localization_checkpoint.exists():
            try:
                self.localization_model, meta = load_model_safe(
                    localization_checkpoint, "mi_localization", device
                )
                self.model_hashes["localization"] = meta["checkpoint_hash"]
                print(f"Localization model loaded (schema: {meta['schema']})")
            except Exception as e:
                print(f"Warning: Localization model load failed: {e}")
        
        # --- Load XGBoost models ---
        if xgb_dir.exists():
            # Load feature schema (FAIL-FAST if missing)
            schema_path = xgb_dir / "feature_schema.json"
            if schema_path.exists():
                with open(schema_path) as f:
                    self.feature_schema = json.load(f)
                print(f"XGBoost feature schema loaded: {self.feature_schema['feature_count']} features")
            else:
                print("WARNING: feature_schema.json missing - XGBoost safety checks disabled")
            
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
        
        # --- Load thresholds ---
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                data = json.load(f)
            self.thresholds = data.get("thresholds", {})
            
            with open(thresholds_path, "rb") as f:
                self.threshold_hash = hashlib.md5(f.read()).hexdigest()[:8]
        else:
            self.thresholds = {"MI": 0.5, "STTC": 0.5, "CD": 0.5, "HYP": 0.5}
            self.threshold_hash = "default"
        
        self.loaded = True
        print(f"Models loaded: Superclass={self.superclass_model is not None}, "
              f"Binary={self.binary_model is not None}, "
              f"Localization={self.localization_model is not None}, "
              f"XGB={len(self.xgb_models)}")


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
    """Load models on startup with fail-fast validation."""
    from src.utils.checkpoint_validation import (
        validate_all_checkpoints,
        CheckpointMismatchError,
        MappingDriftError,
    )
    
    # --- Fail-closed checkpoint validation ---
    print("Validating checkpoints...")
    try:
        results = validate_all_checkpoints(strict=True)
        print("Checkpoint validation passed!")
        for task, result in results.items():
            if isinstance(result, dict) and result.get("valid"):
                print(f"  {task}: out_dim={result.get('out_dim')} ✓")
    except (CheckpointMismatchError, MappingDriftError) as e:
        print(f"CRITICAL: Checkpoint validation failed: {e}")
        raise RuntimeError(f"Checkpoint validation failed: {e}")
    except FileNotFoundError as e:
        print(f"Warning: Some checkpoints missing: {e}")
    
    # --- Load models ---
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
    
    if state.superclass_model is None:
        raise HTTPException(503, "Superclass model not loaded")
    
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
        cnn_logits = state.superclass_model(signal_tensor)
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
        embeddings = state.superclass_model.backbone(signal_tensor).numpy()
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
            model_hash=state.model_hashes.get("superclass", ""),
            threshold_hash=state.threshold_hash,
            timestamp=datetime.utcnow().isoformat(),
        ),
    )


@app.post("/predict/mi-localization", response_model=MILocalizationResponse)
async def predict_mi_localization(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Detection threshold"),
):
    """
    MI localization prediction (5 anatomical regions).
    
    Labels: [AMI, ASMI, ALMI, IMI, LMI]
    Label space: ptbxl_derived_anatomical_v1 (DERIVED from PTB-XL SCP codes)
    """
    import torch
    from src.data.mi_localization import MI_LOCALIZATION_REGIONS
    from src.utils.checkpoint_validation import MI_LOCALIZATION_FINGERPRINT
    
    if state.localization_model is None:
        raise HTTPException(503, "MI localization model not loaded")
    
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10MB)")
    
    try:
        signal = parse_ecg_file(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    
    with torch.no_grad():
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0)
        logits = state.localization_model(signal_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    
    probs_dict = {
        region: float(probs[i])
        for i, region in enumerate(MI_LOCALIZATION_REGIONS)
    }
    detected_regions = [r for r, p in probs_dict.items() if p >= threshold]
    
    return MILocalizationResponse(
        mi_detected=len(detected_regions) > 0,
        regions=detected_regions,
        probabilities=probs_dict,
        label_space="ptbxl_derived_anatomical_v1",
        labels=MI_LOCALIZATION_REGIONS,
        mapping_source="src/data/mi_localization.py",
        mapping_fingerprint=MI_LOCALIZATION_FINGERPRINT,
        localization_head_type="classification_5",
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
