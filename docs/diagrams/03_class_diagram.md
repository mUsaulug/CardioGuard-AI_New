# CardioGuard-AI: Class Diyagramı
## (Sınıf Diyagramı)

---

## 📋 Doküman Bilgileri

| Özellik | Değer |
|---------|-------|
| **Proje Adı** | CardioGuard-AI |
| **Doküman Tipi** | Class Diyagramı |
| **Versiyon** | 1.0.0 |
| **Tarih** | 2026-01-21 |

---

## 1. Genel Bakış

CardioGuard-AI sistemi aşağıdaki ana paketlerden oluşur:

```mermaid
graph TB
    subgraph Packages["📦 Paketler"]
        MODELS["src.models<br/>🧠 Model Tanımları"]
        DATA["src.data<br/>📊 Veri İşleme"]
        PIPELINE["src.pipeline<br/>🔄 Pipeline'lar"]
        XAI["src.xai<br/>💡 Açıklanabilirlik"]
        CONTRACTS["src.contracts<br/>📄 API Kontratları"]
        BACKEND["src.backend<br/>🌐 Web Servisi"]
        UTILS["src.utils<br/>🔧 Yardımcılar"]
    end
    
    BACKEND --> MODELS
    BACKEND --> PIPELINE
    BACKEND --> CONTRACTS
    PIPELINE --> MODELS
    PIPELINE --> DATA
    PIPELINE --> XAI
    XAI --> MODELS
    CONTRACTS --> DATA
    
    style MODELS fill:#e3f2fd
    style DATA fill:#fff3e0
    style PIPELINE fill:#e8f5e9
    style XAI fill:#f3e5f5
    style CONTRACTS fill:#fce4ec
    style BACKEND fill:#e0f2f1
    style UTILS fill:#f5f5f5
```

---

## 2. Model Paketi (src.models)

### 2.1 Tam Class Diyagramı

```mermaid
classDiagram
    direction TB
    
    class ECGCNNConfig {
        <<dataclass>>
        +int in_channels = 12
        +int num_filters = 64
        +int kernel_size = 7
        +float dropout = 0.3
        +int localization_output_dim = 2
    }
    
    class ECGBackbone {
        <<nn.Module>>
        -Sequential features
        +__init__(config: ECGCNNConfig)
        +forward(x: Tensor) Tensor
        ~Conv1d layer1
        ~BatchNorm1d bn1
        ~ReLU relu1
        ~Dropout drop1
        ~Conv1d layer2
        ~BatchNorm1d bn2
        ~ReLU relu2
        ~Dropout drop2
        ~AdaptiveAvgPool1d pool
    }
    
    class BinaryHead {
        <<nn.Module>>
        -Linear classifier
        +__init__(in_features: int)
        +forward(x: Tensor) Tensor
    }
    
    class MultiClassHead {
        <<nn.Module>>
        -Linear classifier
        +__init__(in_features: int, num_classes: int)
        +forward(x: Tensor) Tensor
    }
    
    class FiveClassHead {
        <<nn.Module>>
        +__init__(in_features: int)
    }
    
    class LocalizationHead {
        <<nn.Module>>
        -Linear regressor
        +__init__(in_features: int, output_dim: int = 2)
        +forward(x: Tensor) Tensor
    }
    
    class ECGCNN {
        <<nn.Module>>
        +ECGBackbone backbone
        +Module head
        +__init__(config: ECGCNNConfig, num_classes: int = 1)
        +forward(x: Tensor) Tensor
    }
    
    class MultiTaskECGCNN {
        <<nn.Module>>
        +ECGBackbone backbone
        +Module head
        +LocalizationHead localization_head
        +__init__(config: ECGCNNConfig, num_classes: int = 1)
        +forward(x: Tensor) Dict~str, Tensor~
    }
    
    class CNNEncoder {
        <<deprecated>>
        ~Alias for ECGBackbone
    }
    
    %% İlişkiler
    ECGBackbone --> ECGCNNConfig : uses
    ECGCNN --> ECGBackbone : contains
    ECGCNN --> BinaryHead : uses (num_classes=1)
    ECGCNN --> MultiClassHead : uses (num_classes>1)
    MultiTaskECGCNN --> ECGBackbone : contains
    MultiTaskECGCNN --> LocalizationHead : contains
    FiveClassHead --|> MultiClassHead : extends
    CNNEncoder --|> ECGBackbone : alias
```

### 2.2 Model Varyantları

| Model | Çıktı Boyutu | Head Tipi | Kullanım |
|-------|--------------|-----------|----------|
| **Binary MI** | 1 | BinaryHead | MI vs NORM sınıflandırması |
| **Superclass** | 4 | MultiClassHead | [MI, STTC, CD, HYP] çoklu-etiket |
| **Localization** | 5 | FiveClassHead | [AMI, ASMI, ALMI, IMI, LMI] bölge tespiti |

---

## 3. Data Paketi (src.data)

### 3.1 Class Diyagramı

```mermaid
classDiagram
    direction TB
    
    class SignalDataset {
        <<torch.utils.data.Dataset>>
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
        <<torch.utils.data.Dataset>>
        -ndarray signals
        -ndarray ecg_ids
        -Dict labels
        -callable transform
        +__init__(signals, ecg_ids, labels, transform)
        +__len__() int
        +__getitem__(idx: int) Tuple
    }
    
    class PTBXLConfig {
        <<dataclass>>
        +Path data_dir
        +str metadata_file
        +str scp_statements_file
        +int sampling_rate = 100
        +int signal_length = 1000
        +int num_leads = 12
    }
    
    class DataSplitter {
        +split_by_fold(df, train_folds, val_fold, test_fold) Tuple
        +get_stratified_split(df, label_col, test_size) Tuple
    }
    
    class LabelProcessor {
        +add_binary_mi_labels(df, scp_df, min_likelihood) DataFrame
        +add_superclass_labels(df, scp_df) DataFrame
        +add_5class_labels(df, scp_df, multi_label) DataFrame
        +extract_codes_above_threshold(scp_codes, threshold) Set
    }
    
    class MILocalizationProcessor {
        +List~str~ MI_LOCALIZATION_REGIONS$
        +Dict MI_CODE_TO_REGIONS$
        +List~str~ EXCLUDED_MI_CODES$
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
    
    %% İlişkiler
    SignalDataset --> PTBXLConfig : uses
    SignalDataset --> SignalLoader : uses
    SignalDataset --> LabelProcessor : uses
    CachedSignalDataset --> Normalizer : may use
    MILocalizationProcessor --> LabelProcessor : extends
```

### 3.2 Veri Akışı

```mermaid
graph LR
    subgraph Raw["Ham Veri"]
        PTBXL["PTB-XL<br/>21,837 kayıt"]
    end
    
    subgraph Loading["Yükleme"]
        LOADER["SignalLoader"]
        META["Metadata<br/>DataFrame"]
    end
    
    subgraph Processing["İşleme"]
        LABEL["LabelProcessor"]
        MILOC["MILocalizationProcessor"]
        NORM["Normalizer"]
    end
    
    subgraph Output["Çıktı"]
        DS["SignalDataset<br/>veya<br/>CachedSignalDataset"]
    end
    
    PTBXL --> LOADER --> META
    META --> LABEL --> MILOC --> DS
    LOADER --> NORM --> DS
    
    style PTBXL fill:#e3f2fd
    style DS fill:#e8f5e9
```

---

## 4. Pipeline Paketi (src.pipeline)

### 4.1 Inference Pipeline

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
        <<dataclass>>
        +float superclass_mi_prob
        +float binary_mi_prob
        +bool superclass_mi_decision
        +bool binary_mi_decision
        +AgreementType agreement
        +str triage_level
        +List~str~ warnings
        +to_dict() Dict
    }
    
    class ConsistencyGuard {
        <<module>>
        +check_consistency(superclass_mi_prob, binary_mi_prob, thresholds) ConsistencyResult
        +should_run_localization(consistency, gate_mode) bool
        +derive_norm_from_superclass(superclass_probs, threshold) Dict
    }
    
    class InferencePipeline {
        <<abstract>>
        +Model superclass_model
        +Model binary_model
        +Model localization_model
        +Dict xgb_models
        +Dict thresholds
        +predict(signal) Dict
    }
    
    class SuperclassInference {
        +predict_superclass(signal) Dict
        +extract_embeddings(signal) ndarray
        +ensemble_predictions(cnn_probs, xgb_probs, alpha) Dict
    }
    
    class BinaryInference {
        +predict_binary(signal) float
    }
    
    ConsistencyResult --> AgreementType : contains
    ConsistencyGuard --> ConsistencyResult : returns
    InferencePipeline --> ConsistencyGuard : uses
    SuperclassInference --|> InferencePipeline : implements
    BinaryInference --|> InferencePipeline : implements
```

### 4.2 Training Pipeline

```mermaid
classDiagram
    direction TB
    
    class Trainer {
        <<base>>
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
        +train_xgb_ovr(embeddings, labels) Dict~str, XGBClassifier~
        +calibrate_models(models, X_val, y_val) Dict
        +save_models(models, output_dir)
    }
    
    SuperclassTrainer --|> Trainer
    BinaryTrainer --|> Trainer
    LocalizationTrainer --|> Trainer
```

---

## 5. XAI Paketi (src.xai)

### 5.1 Class Diyagramı

```mermaid
classDiagram
    direction TB
    
    class GradCAM {
        -Module model
        -Module target_layer
        -Tensor gradients
        -Tensor activations
        +__init__(model, target_layer)
        -_register_hooks()
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
        -Dict~str, TreeExplainer~ explainers
        +__init__(models_dict)
        +explain_per_class(x) Dict~str, ndarray~
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
    XAIVisualizer --> GradCAM : visualizes
    XAIVisualizer --> SHAPExplainer : visualizes
    XAIReporter --> XAIVisualizer : uses
```

### 5.2 Açıklama Akışı

```mermaid
graph TB
    subgraph Input
        SIGNAL["EKG Sinyali"]
        PREDS["Tahminler"]
    end
    
    subgraph Generators["Açıklama Üreticileri"]
        GCAM["GradCAM"]
        SHAP["SHAP Explainer"]
    end
    
    subgraph Validators["Doğrulayıcılar"]
        SANITY["SanityChecker"]
    end
    
    subgraph Output["Çıktılar"]
        VIZ["Görselleştirmeler<br/>(PNG)"]
        TEXT["Narratives<br/>(MD)"]
    end
    
    SIGNAL --> GCAM & SHAP
    PREDS --> GCAM & SHAP
    GCAM --> SANITY --> VIZ
    SHAP --> VIZ
    VIZ --> TEXT
    
    style GCAM fill:#ffecb3
    style SHAP fill:#ffecb3
    style SANITY fill:#ffcdd2
```

---

## 6. Contracts Paketi (src.contracts)

### 6.1 Class Diyagramı

```mermaid
classDiagram
    direction TB
    
    class AIResultMapper {
        <<module>>
        +clamp(value, min_val, max_val) float
        +compute_triage(predictions, input_meta) Dict
        +derive_input_meta(signal_path, request_payload) Dict
        +map_predict_output_to_airesult(predict_out, case_id, ...) Dict
    }
    
    class AIResultSchema {
        <<TypedDict>>
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
        <<module>>
        +discover_xai_artifacts(run_dir) List~Dict~
        +validate_artifact_path(path) bool
        +get_artifact_metadata(path) Dict
    }
    
    class TriageLevel {
        <<enumeration>>
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
        REVIEW = "REVIEW"
    }
    
    AIResultMapper --> AIResultSchema : produces
    AIResultMapper --> TriageLevel : uses
    AIResultMapper --> ArtifactDiscovery : uses
```

---

## 7. Backend Paketi (src.backend)

### 7.1 Class Diyagramı

```mermaid
classDiagram
    direction TB
    
    class FastAPIApp {
        <<FastAPI>>
        +str title = "CardioGuard-AI"
        +str version = "1.0.0"
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
        <<Pydantic BaseModel>>
        +float MI
        +float STTC
        +float CD
        +float HYP
        +float NORM
    }
    
    class PrimaryPrediction {
        <<Pydantic BaseModel>>
        +str label
        +float confidence
        +str rule = "MI-first-then-priority"
    }
    
    class SourceProbabilities {
        <<Pydantic BaseModel>>
        +Dict~str, float~ cnn
        +Dict~str, float~ xgb
        +Dict~str, float~ ensemble
    }
    
    class VersionInfo {
        <<Pydantic BaseModel>>
        +str model_hash
        +str threshold_hash
        +str api_version = "1.0.0"
        +str timestamp
    }
    
    class SuperclassPredictionResponse {
        <<Pydantic BaseModel>>
        +str mode = "multilabel-superclass"
        +PredictionProbabilities probabilities
        +List~str~ predicted_labels
        +Dict~str, float~ thresholds
        +PrimaryPrediction primary
        +SourceProbabilities sources
        +VersionInfo versions
    }
    
    class MILocalizationResponse {
        <<Pydantic BaseModel>>
        +bool mi_detected
        +Dict~str, float~ region_probabilities
        +List~str~ detected_regions
        +str label_space
        +str mapping_fingerprint
        +str localization_head_type
    }
    
    class HealthResponse {
        <<Pydantic BaseModel>>
        +str status
        +str timestamp
    }
    
    class ReadyResponse {
        <<Pydantic BaseModel>>
        +bool ready
        +Dict~str, bool~ models_loaded
        +str message
    }
    
    %% İlişkiler
    FastAPIApp --> AppState : contains
    SuperclassPredictionResponse --> PredictionProbabilities
    SuperclassPredictionResponse --> PrimaryPrediction
    SuperclassPredictionResponse --> SourceProbabilities
    SuperclassPredictionResponse --> VersionInfo
```

### 7.2 API Endpoint'leri

```mermaid
graph LR
    subgraph Endpoints
        E1["POST /predict/superclass"]
        E2["POST /predict/mi-localization"]
        E3["GET /health"]
        E4["GET /ready"]
    end
    
    subgraph Responses
        R1["SuperclassPredictionResponse"]
        R2["MILocalizationResponse"]
        R3["HealthResponse"]
        R4["ReadyResponse"]
    end
    
    E1 --> R1
    E2 --> R2
    E3 --> R3
    E4 --> R4
    
    style E1 fill:#e8f5e9
    style E2 fill:#fff3e0
    style E3 fill:#e3f2fd
    style E4 fill:#e3f2fd
```

---

## 8. Utils Paketi (src.utils)

### 8.1 Class Diyagramı

```mermaid
classDiagram
    direction TB
    
    class CheckpointValidator {
        <<module>>
        +validate_checkpoint(path, expected_dim) bool
        +validate_all_checkpoints(checkpoint_dict, strict) Dict
        +compute_mapping_fingerprint() str
        +CheckpointMismatchError
        +MappingDriftError
    }
    
    class SafeModelLoader {
        <<module>>
        +load_model_safe(path, model_class, config) Module
        +normalize_state_dict(state_dict) Dict
        +get_output_dimension(state_dict) int
    }
    
    class Metrics {
        <<module>>
        +compute_auroc(y_true, y_prob) float
        +compute_auprc(y_true, y_prob) float
        +compute_f1(y_true, y_pred) float
        +compute_multilabel_metrics(y_true, y_pred) Dict
    }
    
    CheckpointValidator --> SafeModelLoader : uses
```

---

## 9. Bağımlılık Grafiği

```mermaid
graph TB
    subgraph External["Harici Kütüphaneler"]
        TORCH["PyTorch"]
        NUMPY["NumPy"]
        PANDAS["Pandas"]
        FASTAPI["FastAPI"]
        XGBOOST["XGBoost"]
        SHAP["SHAP"]
    end
    
    subgraph Internal["İç Paketler"]
        MODELS --> TORCH
        DATA --> NUMPY & PANDAS
        PIPELINE --> MODELS & DATA
        XAI --> TORCH & SHAP
        BACKEND --> FASTAPI & PIPELINE
        CONTRACTS --> DATA
    end
    
    style TORCH fill:#ee6c4d
    style NUMPY fill:#3d5a80
    style PANDAS fill:#3d5a80
    style FASTAPI fill:#457b9d
    style XGBOOST fill:#2a9d8f
    style SHAP fill:#e9c46a
```

---

## 10. Tasarım Prensipleri

### 10.1 SOLID Prensipleri

| Prensip | Uygulama |
|---------|----------|
| **Single Responsibility** | Her sınıf tek bir sorumluluğa sahip (ör. GradCAM sadece heatmap üretir) |
| **Open/Closed** | Head sınıfları genişletilebilir (FiveClassHead extends MultiClassHead) |
| **Liskov Substitution** | Tüm Head sınıfları nn.Module'den türetilmiş ve değiştirilebilir |
| **Interface Segregation** | Küçük, odaklı arayüzler (SHAPExplainer abstract class) |
| **Dependency Inversion** | Üst seviye modüller soyutlamalara bağımlı |

### 10.2 Tasarım Desenleri

| Desen | Kullanım Yeri |
|-------|---------------|
| **Factory** | `build_classification_head()`, `build_sequential_cnn()` |
| **Strategy** | Farklı Head tipleri (BinaryHead, MultiClassHead) |
| **Singleton** | AppState (global state) |
| **Observer** | GradCAM hook mekanizması |
| **Facade** | AIResultMapper (karmaşık mapping işlemini basitleştirir) |

---

> **Not:** Bu class diyagramı CardioGuard-AI v1.0.0 mimarisini yansıtmaktadır. Gelecek versiyonlarda Transformer-based modeller ve RAG entegrasyonu için yeni sınıflar eklenecektir.
