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

1. [Model Paketi](#1-model-paketi)
2. [Veri Paketi](#2-veri-paketi)
3. [Pipeline Paketi](#3-pipeline-paketi)
4. [XAI Paketi](#4-xai-paketi)
5. [Backend Paketi](#5-backend-paketi)
6. [Paket Bağımlılıkları](#6-paket-bağımlılıkları)
7. [Planlanan Sınıflar](#7-planlanan-sınıflar)

---

## 1. Model Paketi (src.models)

```mermaid
classDiagram
    direction TB
    
    class ECGCNNConfig {
        <<dataclass>>
        +int in_channels = 12
        +int num_filters = 64
        +int kernel_size = 7
        +float dropout = 0.3
    }
    
    class ECGBackbone {
        <<nn.Module>>
        -Sequential features
        +__init__(config)
        +forward(x) Tensor
    }
    
    class BinaryHead {
        <<nn.Module>>
        -Linear classifier
        +__init__(in_features)
        +forward(x) Tensor
    }
    
    class MultiClassHead {
        <<nn.Module>>
        -Linear classifier
        +__init__(in_features, num_classes)
        +forward(x) Tensor
    }
    
    class FiveClassHead {
        <<nn.Module>>
        +__init__(in_features)
        +forward(x) Tensor
    }
    
    class ECGCNN {
        <<nn.Module>>
        +ECGBackbone backbone
        +Module head
        +__init__(config, num_classes)
        +forward(x) Tensor
    }
    
    ECGBackbone ..> ECGCNNConfig
    ECGCNN *-- ECGBackbone
    ECGCNN o-- BinaryHead
    ECGCNN o-- MultiClassHead
    FiveClassHead --|> MultiClassHead
```

### Model Varyantları

| Model | Çıktı | Head Tipi | Kullanım |
|-------|-------|-----------|----------|
| Binary MI | 1 | BinaryHead | MI vs Normal sınıflandırma |
| Superclass | 4 | MultiClassHead | MI, STTC, CD, HYP çoklu-etiket |
| Lokalizasyon | 5 | FiveClassHead | AMI, ASMI, ALMI, IMI, LMI bölge tespiti |

---

## 2. Veri Paketi (src.data)

```mermaid
classDiagram
    direction TB
    
    class SignalDataset {
        <<Dataset>>
        -DataFrame df
        -Path base_path
        -str label_column
        -callable transform
        +__init__(df, base_path, ...)
        +__len__() int
        +__getitem__(idx) Tuple
    }
    
    class CachedSignalDataset {
        <<Dataset>>
        -ndarray signals
        -ndarray ecg_ids
        -Dict labels
        +__init__(signals, ecg_ids, labels)
        +__len__() int
        +__getitem__(idx) Tuple
    }
    
    class LabelProcessor {
        +add_binary_mi_labels(df, scp_df) DataFrame
        +add_superclass_labels(df, scp_df) DataFrame
        +extract_codes_above_threshold(scp_codes, threshold) Set
    }
    
    class MILocalizationProcessor {
        +List MI_LOCALIZATION_REGIONS
        +Dict MI_CODE_TO_REGIONS
        +extract_mi_regions(scp_codes) List
        +extract_mi_localization_labels(df) ndarray
        +get_mi_localization_mask(y_multi4) ndarray
    }
    
    class SignalLoader {
        +load_single_signal(filename, base_path) ndarray
        +load_signals_batch(df, base_path) ndarray
        +build_npz_cache(df, base_path, cache_path)
    }
    
    class Normalizer {
        +min_max_normalize(signal) ndarray
        +per_lead_normalize(signal) ndarray
    }
    
    SignalDataset ..> SignalLoader
    SignalDataset ..> LabelProcessor
    CachedSignalDataset ..> Normalizer
    MILocalizationProcessor --|> LabelProcessor
```

---

## 3. Pipeline Paketi (src.pipeline)

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
        +List warnings
        +to_dict() Dict
    }
    
    class ConsistencyGuard {
        +check_consistency(super_prob, binary_prob) ConsistencyResult
        +should_run_localization(consistency) bool
        +derive_norm_from_superclass(probs) Dict
    }
    
    ConsistencyResult *-- AgreementType
    ConsistencyGuard ..> ConsistencyResult
```

### Uyum Tipleri Açıklaması

| Tip | Super MI | Binary MI | Triaj | Yorum |
|-----|----------|-----------|-------|-------|
| AGREE_MI | Pozitif | Pozitif | Yüksek | Her iki model MI tespit etti |
| AGREE_NO_MI | Negatif | Negatif | Düşük | Her iki model normal buldu |
| DISAGREE_TYPE_1 | Pozitif | Negatif | İnceleme | Düşük güvenlikli MI |
| DISAGREE_TYPE_2 | Negatif | Pozitif | İnceleme | Olası kaçırılmış MI |

---

## 4. XAI Paketi (src.xai)

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
    
    class XAIVisualizer {
        +plot_gradcam_overlay(signal, cam, class_name) Figure
        +plot_shap_summary(shap_values, feature_names) Figure
        +save_artifacts(run_dir)
    }
    
    SHAPXGBExplainer --|> SHAPExplainer
    SHAPOVRExplainer --|> SHAPExplainer
    XAIVisualizer ..> GradCAM
    XAIVisualizer ..> SHAPExplainer
```

---

## 5. Backend Paketi (src.backend)

```mermaid
classDiagram
    direction TB
    
    class AppState {
        +Module superclass_model
        +Module binary_model
        +Module localization_model
        +Dict xgb_models
        +Dict calibrators
        +StandardScaler scaler
        +Dict thresholds
        +bool is_loaded
        +load_models(paths)
    }
    
    class PredictionProbabilities {
        <<Pydantic>>
        +float MI
        +float STTC
        +float CD
        +float HYP
        +float NORM
    }
    
    class SuperclassPredictionResponse {
        <<Pydantic>>
        +str mode
        +PredictionProbabilities probabilities
        +List predicted_labels
        +Dict thresholds
        +str primary_label
        +str triage_level
    }
    
    class MILocalizationResponse {
        <<Pydantic>>
        +bool mi_detected
        +Dict region_probabilities
        +List detected_regions
        +str mapping_fingerprint
    }
    
    SuperclassPredictionResponse *-- PredictionProbabilities
    AppState ..> SuperclassPredictionResponse
    AppState ..> MILocalizationResponse
```

---

## 6. Paket Bağımlılıkları

```mermaid
graph TD
    BACKEND["src.backend"]
    PIPELINE["src.pipeline"]
    MODELS["src.models"]
    DATA["src.data"]
    XAI["src.xai"]
    CONTRACTS["src.contracts"]
    UTILS["src.utils"]
    
    BACKEND --> PIPELINE
    BACKEND --> CONTRACTS
    PIPELINE --> MODELS
    PIPELINE --> DATA
    PIPELINE --> XAI
    XAI --> MODELS
    CONTRACTS --> DATA
    MODELS --> UTILS
```

---

## 7. Planlanan Sınıflar (v2.0)

```mermaid
classDiagram
    direction TB
    
    class RAGRetriever {
        <<planned>>
        -VectorStore store
        -Embedder embedder
        +retrieve_context(query) List
        +embed_query(text) ndarray
    }
    
    class UncertaintyEstimator {
        <<planned>>
        -int n_iterations
        +mc_dropout_inference(model, x) Dict
        +compute_confidence_interval(samples) Tuple
    }
    
    class LLMReportGenerator {
        <<planned>>
        -str model_name
        -RAGRetriever retriever
        +generate_clinical_report(predictions, context) str
        +format_findings(predictions) Dict
    }
    
    class TransformerBackbone {
        <<planned>>
        -MultiHeadAttention attention
        -PositionalEncoding pos_enc
        +forward(x) Tensor
        +get_attention_weights() Tensor
    }
    
    RAGRetriever ..> LLMReportGenerator
    TransformerBackbone --|> ECGBackbone
```

---

## 8. UML Notasyonu

| Sembol | Anlam |
|--------|-------|
| `*--` | Kompozisyon (parça bütüne bağımlı) |
| `o--` | Agregasyon (parça bağımsız yaşayabilir) |
| `--|>` | Kalıtım (generalization) |
| `..>` | Bağımlılık (dependency) |
| `-->` | Birliktelik (association) |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |
| Kalite Güvence Mühendisi | | | |

---

**Doküman Sonu**
