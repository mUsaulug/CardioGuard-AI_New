# CardioGuard-AI
# Sınıf Diyagramı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sınıf Diyagramı (Class Diagram)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## İçindekiler

1. [Genel Bakış](#1-genel-bakış)
2. [Model Paketi](#2-model-paketi)
3. [Veri Paketi](#3-veri-paketi)
4. [Pipeline Paketi](#4-pipeline-paketi)
5. [XAI Paketi](#5-xai-paketi)
6. [Kontrat Paketi](#6-kontrat-paketi)
7. [Backend Paketi](#7-backend-paketi)
8. [Yardımcı Paket](#8-yardımcı-paket)
9. [Bağımlılık Grafiği](#9-bağımlılık-grafiği)
10. [Tasarım Prensipleri](#10-tasarım-prensipleri)

---

## 1. Genel Bakış

CardioGuard-AI sistemi aşağıdaki ana paketlerden oluşmaktadır:

```mermaid
graph TB
    subgraph Paketler
        MODELS["src.models - Model Tanımları"]
        DATA["src.data - Veri İşleme"]
        PIPELINE["src.pipeline - İş Akışları"]
        XAI["src.xai - Açıklanabilirlik"]
        CONTRACTS["src.contracts - API Kontratları"]
        BACKEND["src.backend - Web Servisi"]
        UTILS["src.utils - Yardımcı Fonksiyonlar"]
    end
    
    BACKEND --> MODELS
    BACKEND --> PIPELINE
    BACKEND --> CONTRACTS
    PIPELINE --> MODELS
    PIPELINE --> DATA
    PIPELINE --> XAI
    XAI --> MODELS
    CONTRACTS --> DATA
```

---

## 2. Model Paketi (src.models)

### 2.1 Sınıf Diyagramı

```mermaid
classDiagram
    direction TB
    
    class ECGCNNConfig {
        +int in_channels = 12
        +int num_filters = 64
        +int kernel_size = 7
        +float dropout = 0.3
        +int localization_output_dim = 2
    }
    
    class ECGBackbone {
        -Sequential features
        +__init__(config: ECGCNNConfig)
        +forward(x: Tensor) Tensor
    }
    
    class BinaryHead {
        -Linear classifier
        +__init__(in_features: int)
        +forward(x: Tensor) Tensor
    }
    
    class MultiClassHead {
        -Linear classifier
        +__init__(in_features: int, num_classes: int)
        +forward(x: Tensor) Tensor
    }
    
    class FiveClassHead {
        +__init__(in_features: int)
    }
    
    class LocalizationHead {
        -Linear regressor
        +__init__(in_features: int, output_dim: int)
        +forward(x: Tensor) Tensor
    }
    
    class ECGCNN {
        +ECGBackbone backbone
        +Module head
        +__init__(config: ECGCNNConfig, num_classes: int)
        +forward(x: Tensor) Tensor
    }
    
    class MultiTaskECGCNN {
        +ECGBackbone backbone
        +Module head
        +LocalizationHead localization_head
        +__init__(config: ECGCNNConfig, num_classes: int)
        +forward(x: Tensor) Dict
    }
    
    ECGBackbone --> ECGCNNConfig : kullanır
    ECGCNN --> ECGBackbone : içerir
    ECGCNN --> BinaryHead : kullanır
    ECGCNN --> MultiClassHead : kullanır
    MultiTaskECGCNN --> ECGBackbone : içerir
    MultiTaskECGCNN --> LocalizationHead : içerir
    FiveClassHead --|> MultiClassHead : genişletir
```

### 2.2 Model Varyantları

| Model | Çıktı Boyutu | Kafa Tipi | Kullanım Amacı |
|-------|--------------|-----------|----------------|
| Binary MI | 1 | BinaryHead | MI ve Normal sınıflandırma |
| Superclass | 4 | MultiClassHead | MI, STTC, CD, HYP çoklu etiket |
| Lokalizasyon | 5 | FiveClassHead | AMI, ASMI, ALMI, IMI, LMI bölge tespiti |

### 2.3 Sınıf Açıklamaları

| Sınıf | Açıklama |
|-------|----------|
| ECGCNNConfig | Model yapılandırma parametrelerini tutan veri sınıfı |
| ECGBackbone | Evrişimsel sinir ağı omurgası, 64 boyutlu gömme vektörü üretir |
| BinaryHead | İkili sınıflandırma için tek çıktılı doğrusal katman |
| MultiClassHead | Çoklu sınıf sınıflandırma için n çıktılı doğrusal katman |
| FiveClassHead | Beş sınıflı sınıflandırma için özelleştirilmiş kafa |
| LocalizationHead | Regresyon çıktısı için doğrusal katman |
| ECGCNN | Tam EKG sınıflandırma modeli |
| MultiTaskECGCNN | Çoklu görev modeli (sınıflandırma ve lokalizasyon) |

---

## 3. Veri Paketi (src.data)

### 3.1 Sınıf Diyagramı

```mermaid
classDiagram
    direction TB
    
    class SignalDataset {
        -DataFrame df
        -Path base_path
        -str filename_column
        -str label_column
        -callable transform
        -Path cache_dir
        +__init__(df, base_path, ...)
        +__len__() int
        +__getitem__(idx: int) Tuple
        +__iter__() Iterator
    }
    
    class CachedSignalDataset {
        -ndarray signals
        -ndarray ecg_ids
        -Dict labels
        -callable transform
        +__init__(signals, ecg_ids, labels, transform)
        +__len__() int
        +__getitem__(idx: int) Tuple
    }
    
    class PTBXLConfig {
        +Path data_dir
        +str metadata_file
        +str scp_statements_file
        +int sampling_rate = 100
        +int signal_length = 1000
        +int num_leads = 12
    }
    
    class LabelProcessor {
        +add_binary_mi_labels(df, scp_df, min_likelihood) DataFrame
        +add_superclass_labels(df, scp_df) DataFrame
        +add_5class_labels(df, scp_df, multi_label) DataFrame
        +extract_codes_above_threshold(scp_codes, threshold) Set
    }
    
    class MILocalizationProcessor {
        +List MI_LOCALIZATION_REGIONS
        +Dict MI_CODE_TO_REGIONS
        +List EXCLUDED_MI_CODES
        +extract_mi_regions(scp_codes, min_likelihood) List
        +extract_mi_localization_labels(df) ndarray
        +add_mi_localization_labels(df) DataFrame
        +get_mi_localization_mask(y_multi4) ndarray
    }
    
    class SignalLoader {
        +load_single_signal(filename, base_path, cache_dir) ndarray
        +load_signals_batch(df, base_path, ...) ndarray
        +build_npz_cache(df, base_path, cache_path)
        +load_npz_cache(cache_path) Tuple
    }
    
    class Normalizer {
        +min_max_normalize(signal) ndarray
        +z_score_normalize(signal) ndarray
        +per_lead_normalize(signal) ndarray
    }
    
    SignalDataset --> PTBXLConfig : kullanır
    SignalDataset --> SignalLoader : kullanır
    SignalDataset --> LabelProcessor : kullanır
    CachedSignalDataset --> Normalizer : kullanabilir
    MILocalizationProcessor --> LabelProcessor : genişletir
```

### 3.2 Veri Akışı

```mermaid
graph LR
    subgraph Ham_Veri["Ham Veri"]
        PTBXL["PTB-XL Veritabanı (21,837 kayıt)"]
    end
    
    subgraph Yukleme["Yükleme"]
        LOADER["SignalLoader"]
        META["Metadata DataFrame"]
    end
    
    subgraph Isleme["İşleme"]
        LABEL["LabelProcessor"]
        MILOC["MILocalizationProcessor"]
        NORM["Normalizer"]
    end
    
    subgraph Cikti["Çıktı"]
        DS["SignalDataset veya CachedSignalDataset"]
    end
    
    PTBXL --> LOADER --> META
    META --> LABEL --> MILOC --> DS
    LOADER --> NORM --> DS
```

---

## 4. Pipeline Paketi (src.pipeline)

### 4.1 Çıkarım Pipeline Sınıfları

```mermaid
classDiagram
    direction TB
    
    class AgreementType {
        <<enumeration>>
        AGREE_MI
        AGREE_NO_MI
        DISAGREE_TYPE_1
        DISAGREE_TYPE_2
    }
    
    class ConsistencyResult {
        +float superclass_mi_prob
        +float binary_mi_prob
        +bool superclass_mi_decision
        +bool binary_mi_decision
        +AgreementType agreement
        +str triage_level
        +List warnings
        +to_dict() Dict
    }
    
    class ConsistencyGuard {
        +check_consistency(superclass_mi_prob, binary_mi_prob, thresholds) ConsistencyResult
        +should_run_localization(consistency, gate_mode) bool
        +derive_norm_from_superclass(superclass_probs, threshold) Dict
    }
    
    ConsistencyResult --> AgreementType : içerir
    ConsistencyGuard --> ConsistencyResult : döndürür
```

### 4.2 Eğitim Pipeline Sınıfları

```mermaid
classDiagram
    direction TB
    
    class Trainer {
        +Model model
        +Optimizer optimizer
        +Loss criterion
        +DataLoader train_loader
        +DataLoader val_loader
        +train_epoch() float
        +validate() Dict
        +save_checkpoint(path)
        +load_checkpoint(path)
    }
    
    class SuperclassTrainer {
        +train_superclass_cnn(epochs, lr) Model
        +compute_multilabel_metrics(preds, labels) Dict
    }
    
    class BinaryTrainer {
        +train_binary_cnn(epochs, lr) Model
        +compute_binary_metrics(preds, labels) Dict
    }
    
    class LocalizationTrainer {
        +train_localization(pretrained_path) Model
        +compute_region_metrics(preds, labels) Dict
    }
    
    class XGBoostTrainer {
        +train_xgb_ovr(embeddings, labels) Dict
        +calibrate_models(models, X_val, y_val) Dict
        +save_models(models, output_dir)
    }
    
    SuperclassTrainer --|> Trainer
    BinaryTrainer --|> Trainer
    LocalizationTrainer --|> Trainer
```

### 4.3 Uyum Tipi Açıklamaları

| Uyum Tipi | Super MI | Binary MI | Triaj | Yorum |
|-----------|----------|-----------|-------|-------|
| AGREE_MI | Pozitif | Pozitif | Yüksek | Her iki model MI tespit etti |
| AGREE_NO_MI | Negatif | Negatif | Düşük | Her iki model MI tespit etmedi |
| DISAGREE_TYPE_1 | Pozitif | Negatif | İnceleme | Düşük güvenlikli MI |
| DISAGREE_TYPE_2 | Negatif | Pozitif | İnceleme | Superclass kaçırmış olabilir |

---

## 5. XAI Paketi (src.xai)

### 5.1 Sınıf Diyagramı

```mermaid
classDiagram
    direction TB
    
    class GradCAM {
        -Module model
        -Module target_layer
        -Tensor gradients
        -Tensor activations
        +__init__(model, target_layer)
        +generate(inputs, class_index) ndarray
    }
    
    class SHAPExplainer {
        <<abstract>>
        +explain(model, X) ndarray
    }
    
    class SHAPXGBExplainer {
        -TreeExplainer explainer
        +__init__(model)
        +explain_single(x) ndarray
        +explain_batch(X) ndarray
        +get_feature_importance() Dict
    }
    
    class SHAPOVRExplainer {
        -Dict explainers
        +__init__(models_dict)
        +explain_per_class(x) Dict
    }
    
    class SanityChecker {
        +randomize_model_weights(model, layer_name) Module
        +compare_explanations(original, randomized) Dict
        +run_sanity_check(model, inputs) bool
    }
    
    class XAIVisualizer {
        +plot_gradcam_overlay(signal, cam, class_name) Figure
        +plot_shap_summary(shap_values, feature_names) Figure
        +save_artifacts(run_dir)
    }
    
    class XAIReporter {
        +generate_narrative(predictions, explanations) str
        +create_report(run_dir) Path
    }
    
    SHAPXGBExplainer --|> SHAPExplainer
    SHAPOVRExplainer --|> SHAPExplainer
    XAIVisualizer --> GradCAM : görselleştirir
    XAIVisualizer --> SHAPExplainer : görselleştirir
    XAIReporter --> XAIVisualizer : kullanır
```

### 5.2 Açıklama Akışı

```mermaid
graph TB
    subgraph Girdi
        SIGNAL["EKG Sinyali"]
        PREDS["Tahminler"]
    end
    
    subgraph Ureticiler["Açıklama Üreticileri"]
        GCAM["GradCAM"]
        SHAP["SHAP Açıklayıcı"]
    end
    
    subgraph Dogrulayicilar["Doğrulayıcılar"]
        SANITY["SanityChecker"]
    end
    
    subgraph Cikti["Çıktılar"]
        VIZ["Görselleştirmeler (PNG)"]
        TEXT["Metin Açıklamaları (MD)"]
    end
    
    SIGNAL --> GCAM
    SIGNAL --> SHAP
    PREDS --> GCAM
    PREDS --> SHAP
    GCAM --> SANITY --> VIZ
    SHAP --> VIZ
    VIZ --> TEXT
```

---

## 6. Kontrat Paketi (src.contracts)

### 6.1 Sınıf Diyagramı

```mermaid
classDiagram
    direction TB
    
    class AIResultMapper {
        +clamp(value, min_val, max_val) float
        +compute_triage(predictions, input_meta) Dict
        +derive_input_meta(signal_path, request_payload) Dict
        +map_predict_output_to_airesult(predict_out, case_id, ...) Dict
    }
    
    class AIResultSchema {
        +Dict identity
        +str mode
        +Dict input
        +Dict predictions
        +Dict localization
        +Dict triage
        +Dict sources
        +Dict explanations
        +Dict versions
    }
    
    class ArtifactDiscovery {
        +discover_xai_artifacts(run_dir) List
        +validate_artifact_path(path) bool
        +get_artifact_metadata(path) Dict
    }
    
    class TriageLevel {
        <<enumeration>>
        HIGH
        MEDIUM
        LOW
        REVIEW
    }
    
    AIResultMapper --> AIResultSchema : üretir
    AIResultMapper --> TriageLevel : kullanır
    AIResultMapper --> ArtifactDiscovery : kullanır
```

---

## 7. Backend Paketi (src.backend)

### 7.1 Sınıf Diyagramı

```mermaid
classDiagram
    direction TB
    
    class FastAPIApp {
        +str title
        +str version
        +CORSMiddleware cors
    }
    
    class AppState {
        +Module superclass_model
        +Module binary_model
        +Module localization_model
        +Dict xgb_models
        +Dict calibrators
        +StandardScaler scaler
        +Dict thresholds
        +bool is_loaded
        +__init__()
        +load_models(checkpoint_paths, xgb_dir, thresholds_path)
    }
    
    class PredictionProbabilities {
        +float MI
        +float STTC
        +float CD
        +float HYP
        +float NORM
    }
    
    class PrimaryPrediction {
        +str label
        +float confidence
        +str rule
    }
    
    class SourceProbabilities {
        +Dict cnn
        +Dict xgb
        +Dict ensemble
    }
    
    class VersionInfo {
        +str model_hash
        +str threshold_hash
        +str api_version
        +str timestamp
    }
    
    class SuperclassPredictionResponse {
        +str mode
        +PredictionProbabilities probabilities
        +List predicted_labels
        +Dict thresholds
        +PrimaryPrediction primary
        +SourceProbabilities sources
        +VersionInfo versions
    }
    
    class MILocalizationResponse {
        +bool mi_detected
        +Dict region_probabilities
        +List detected_regions
        +str label_space
        +str mapping_fingerprint
        +str localization_head_type
    }
    
    class HealthResponse {
        +str status
        +str timestamp
    }
    
    class ReadyResponse {
        +bool ready
        +Dict models_loaded
        +str message
    }
    
    FastAPIApp --> AppState : içerir
    SuperclassPredictionResponse --> PredictionProbabilities
    SuperclassPredictionResponse --> PrimaryPrediction
    SuperclassPredictionResponse --> SourceProbabilities
    SuperclassPredictionResponse --> VersionInfo
```

### 7.2 API Uç Noktaları

| Uç Nokta | Metod | Yanıt Sınıfı | Açıklama |
|----------|-------|--------------|----------|
| /predict/superclass | POST | SuperclassPredictionResponse | Çoklu etiket patoloji tahmini |
| /predict/mi-localization | POST | MILocalizationResponse | MI anatomik lokalizasyonu |
| /health | GET | HealthResponse | Canlılık kontrolü |
| /ready | GET | ReadyResponse | Hazırlık kontrolü |

---

## 8. Yardımcı Paket (src.utils)

### 8.1 Sınıf Diyagramı

```mermaid
classDiagram
    direction TB
    
    class CheckpointValidator {
        +validate_checkpoint(path, expected_dim) bool
        +validate_all_checkpoints(checkpoint_dict, strict) Dict
        +compute_mapping_fingerprint() str
    }
    
    class SafeModelLoader {
        +load_model_safe(path, model_class, config) Module
        +normalize_state_dict(state_dict) Dict
        +get_output_dimension(state_dict) int
    }
    
    class Metrics {
        +compute_auroc(y_true, y_prob) float
        +compute_auprc(y_true, y_prob) float
        +compute_f1(y_true, y_pred) float
        +compute_multilabel_metrics(y_true, y_pred) Dict
    }
    
    CheckpointValidator --> SafeModelLoader : kullanır
```

### 8.2 İstisna Sınıfları

| İstisna | Açıklama |
|---------|----------|
| CheckpointMismatchError | Kontrol noktası boyut uyuşmazlığı |
| MappingDriftError | MI eşleme parmak izi değişikliği |

---

## 9. Bağımlılık Grafiği

```mermaid
graph TB
    subgraph Harici_Kutuphaneler["Harici Kütüphaneler"]
        TORCH["PyTorch"]
        NUMPY["NumPy"]
        PANDAS["Pandas"]
        FASTAPI["FastAPI"]
        XGBOOST["XGBoost"]
        SHAP["SHAP"]
    end
    
    subgraph Ic_Paketler["İç Paketler"]
        MODELS["src.models"]
        DATA["src.data"]
        PIPELINE["src.pipeline"]
        XAI["src.xai"]
        BACKEND["src.backend"]
        CONTRACTS["src.contracts"]
    end
    
    MODELS --> TORCH
    DATA --> NUMPY
    DATA --> PANDAS
    PIPELINE --> MODELS
    PIPELINE --> DATA
    XAI --> TORCH
    XAI --> SHAP
    BACKEND --> FASTAPI
    BACKEND --> PIPELINE
```

---

## 10. Tasarım Prensipleri

### 10.1 SOLID Prensipleri Uygulaması

| Prensip | Açıklama | Uygulama Örneği |
|---------|----------|-----------------|
| Tek Sorumluluk | Her sınıf tek bir sorumluluğa sahiptir | GradCAM yalnızca ısı haritası üretir |
| Açık/Kapalı | Sınıflar genişletmeye açık, değişikliğe kapalıdır | FiveClassHead, MultiClassHead sınıfını genişletir |
| Liskov Yerine Koyma | Alt sınıflar üst sınıfların yerine kullanılabilir | Tüm Head sınıfları nn.Module türevlidir |
| Arayüz Ayrımı | Küçük, odaklı arayüzler | SHAPExplainer soyut sınıfı |
| Bağımlılık Tersine Çevirme | Yüksek seviye modüller soyutlamalara bağımlıdır | Pipeline, Model arayüzlerine bağımlıdır |

### 10.2 Uygulanan Tasarım Desenleri

| Desen | Kullanım Yeri | Açıklama |
|-------|---------------|----------|
| Fabrika (Factory) | build_classification_head() | Tip parametresine göre uygun head sınıfı oluşturur |
| Strateji (Strategy) | Head sınıfları | Farklı sınıflandırma stratejileri |
| Tekil (Singleton) | AppState | Global uygulama durumu |
| Gözlemci (Observer) | GradCAM hook mekanizması | İleri ve geri yayılım olaylarını dinler |
| Cephe (Facade) | AIResultMapper | Karmaşık eşleme işlemini basitleştirir |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |
| Kalite Güvence Mühendisi | | | |

---

**Doküman Sonu**

*Bu sınıf diyagramı CardioGuard-AI v1.0.0 mimarisini yansıtmaktadır. Gelecek versiyonlarda Transformer tabanlı modeller ve RAG entegrasyonu için yeni sınıflar eklenecektir.*
