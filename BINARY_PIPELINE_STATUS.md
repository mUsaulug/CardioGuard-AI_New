`run_xai_demo.py` pipeline’a uyumlandı:

- Checkpoint yükleme standardize edildi
- Train stats ile normalize ediliyor
- XGB scaler & calibrated model kullanılıyor
- SHAP için base model kullanılıyor

---

## 3) Güncel Metrikler (Örnek)

### XGBoost (calibrated)
**Validation:**
- ROC-AUC: **0.9839**
- PR-AUC: **0.9652**
- F1: **0.9055**
- Best threshold: **0.80**

**Test:**
- ROC-AUC: **0.9763**
- PR-AUC: **0.9497**
- F1: **0.8664**

### Ensemble (compare_models çıktısı)
Örnek sonuçlar:

- CNN: AUC 0.9380 / PR_AUC 0.9215 / F1 0.7436 / Acc 0.8438  
- XGB: AUC 0.9408 / PR_AUC 0.9254 / F1 0.7850 / Acc 0.8629  
- Ensemble (α=0.15): AUC 0.9420 / PR_AUC 0.9268 / F1 0.8132 / Acc 0.8765

> **Ensemble artık daha iyi**, `best_alpha < 1.0` → hedef başarıyla sağlandı.

---

## 4) Artifact Sözleşmesi (Backend İçin Minimum)

```
checkpoints/ecgcnn.pt
features_out/train.npz
features_out/val.npz
features_out/test.npz

logs/xgb/xgb_model.json
logs/xgb/xgb_scaler.joblib
logs/xgb/xgb_calibrated.joblib
logs/xgb/metrics.json
logs/xgb/xgb_config.json

reports/comparison_report.md
reports/comparison_report.csv
reports/ensemble_config.json
reports/xai/*.png
```

---

## 5) Çalıştırma Komutları (PowerShell Tek Satır)

### (1) Feature Extraction
```
python -m src.pipeline.run_feature_extraction --checkpoint checkpoints/ecgcnn.pt --output-dir features_out --batch-size 32
```

### (2) XGBoost Training + Calibration
```
python -m src.pipeline.run_xgb --train features_out/train.npz --val features_out/val.npz --test features_out/test.npz --output-dir logs/xgb --calibration sigmoid
```

### (3) Compare / Ensemble
```
python -m src.pipeline.compare_models --cnn-path checkpoints/ecgcnn.pt --xgb-path logs/xgb/xgb_model.json --xgb-metrics logs/xgb/metrics.json --output-dir reports --batch-size 32
```

### (4) XAI Demo
```
python -m src.pipeline.run_xai_demo --cnn-path checkpoints/ecgcnn.pt --xgb-dir logs/xgb --xgb-path logs/xgb/xgb_model.json --output-dir reports/xai --num-samples 4
```

---

## 6) Mevcut Durumun Özeti
- ✅ CNN checkpoint ile embedding extraction garanti
- ✅ XGB performansı güçlü (AUC ~0.97 test)
- ✅ Ensemble faydalı ve `best_alpha < 1.0`
- ✅ XAI pipeline (Grad-CAM + SHAP) çalışıyor

**Binary Hybrid hattı şu anda stabil, tekrarlanabilir ve backend’e hazır.**

---

## 7) Bundan Sonraki Adımlar
1. **Backend inference entrypoint** ekle (tek giriş noktası)
2. Artefact’ları API contract’a bağla
3. Sonrasında 5-class + localization’a geç
