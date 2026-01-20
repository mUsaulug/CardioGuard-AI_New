# CardioGuard-AI
# Sınıf Diyagramı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sınıf Diyagramı (Class Diagram)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## 1. Model Paketi (src.models)

```mermaid
classDiagram
    direction TB
    
    class ECGCNNConfig {
        +int in_channels
        +int num_filters
        +int kernel_size
        +float dropout
    }
    
    class ECGBackbone {
        -Sequential features
        +forward(x) Tensor
    }
    
    class BinaryHead {
        -Linear classifier
        +forward(x) Tensor
    }
    
    class MultiClassHead {
        -Linear classifier
        +forward(x) Tensor
    }
    
    class FiveClassHead {
        +forward(x) Tensor
    }
    
    class ECGCNN {
        +ECGBackbone backbone
        +Module head
        +forward(x) Tensor
    }
    
    ECGBackbone ..> ECGCNNConfig
    ECGCNN *-- ECGBackbone
    ECGCNN o-- BinaryHead
    ECGCNN o-- MultiClassHead
    FiveClassHead --|> MultiClassHead
```

**Notasyon Açıklaması:**
- `*--` Kompozisyon (Composition)
- `o--` Agregasyon (Aggregation)  
- `--|>` Kalıtım (Generalization)
- `..>` Bağımlılık (Dependency)

---

## 2. Veri Paketi (src.data)

```mermaid
classDiagram
    direction TB
    
    class SignalDataset {
        -DataFrame df
        -Path base_path
        +__len__() int
        +__getitem__(idx) Tuple
    }
    
    class CachedSignalDataset {
        -ndarray signals
        -Dict labels
        +__len__() int
        +__getitem__(idx) Tuple
    }
    
    class LabelProcessor {
        +add_binary_mi_labels()
        +add_superclass_labels()
    }
    
    class MILocalizationProcessor {
        +extract_mi_regions()
        +get_localization_mask()
    }
    
    class SignalLoader {
        +load_single_signal()
        +load_signals_batch()
    }
    
    class Normalizer {
        +min_max_normalize()
        +per_lead_normalize()
    }
    
    SignalDataset ..> SignalLoader
    SignalDataset ..> LabelProcessor
    MILocalizationProcessor --|> LabelProcessor
    CachedSignalDataset ..> Normalizer
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
        +float superclass_mi_prob
        +float binary_mi_prob
        +AgreementType agreement
        +str triage_level
    }
    
    class ConsistencyGuard {
        +check_consistency()
        +should_run_localization()
        +derive_norm()
    }
    
    ConsistencyResult *-- AgreementType
    ConsistencyGuard ..> ConsistencyResult
```

---

## 4. XAI Paketi (src.xai)

```mermaid
classDiagram
    direction TB
    
    class GradCAM {
        -Module model
        -Tensor gradients
        -Tensor activations
        +generate() ndarray
    }
    
    class SHAPExplainer {
        <<abstract>>
        +explain() ndarray
    }
    
    class SHAPXGBExplainer {
        -TreeExplainer explainer
        +explain_single()
        +get_feature_importance()
    }
    
    class XAIVisualizer {
        +plot_gradcam_overlay()
        +plot_shap_summary()
    }
    
    SHAPXGBExplainer --|> SHAPExplainer
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
        +bool is_loaded
        +load_models()
    }
    
    class PredictionResponse {
        +Dict probabilities
        +List labels
        +str triage
    }
    
    class HealthResponse {
        +str status
        +str timestamp
    }
    
    AppState ..> PredictionResponse
```

---

## 6. Kontrat Paketi (src.contracts)

```mermaid
classDiagram
    direction TB
    
    class AIResultMapper {
        +clamp()
        +compute_triage()
        +map_to_airesult()
    }
    
    class TriageLevel {
        <<enumeration>>
        HIGH
        MEDIUM
        LOW
        REVIEW
    }
    
    class ArtifactDiscovery {
        +discover_xai_artifacts()
        +validate_path()
    }
    
    AIResultMapper ..> TriageLevel
    AIResultMapper ..> ArtifactDiscovery
```

---

## 7. Paket Bağımlılıkları

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

## 8. Planlanan Sınıflar (v2.0)

```mermaid
classDiagram
    direction TB
    
    class RAGRetriever {
        <<planned>>
        +retrieve_context()
        +embed_query()
    }
    
    class UncertaintyEstimator {
        <<planned>>
        +mc_dropout_inference()
        +compute_confidence_interval()
    }
    
    class LLMReportGenerator {
        <<planned>>
        +generate_clinical_report()
        +format_findings()
    }
    
    class TransformerBackbone {
        <<planned>>
        +forward()
        +attention_weights()
    }
    
    RAGRetriever ..> LLMReportGenerator
    UncertaintyEstimator ..> ECGCNN
    TransformerBackbone --|> ECGBackbone
```

---

## 9. UML Notasyonu Referansı

| Sembol | Anlam | Açıklama |
|--------|-------|----------|
| `--|>` | Kalıtım | Alt sınıf üst sınıftan türer |
| `*--` | Kompozisyon | Parça bütüne bağımlı yaşar |
| `o--` | Agregasyon | Parça bütünden bağımsız yaşayabilir |
| `-->` | Birliktelik | Sınıflar arası referans |
| `..>` | Bağımlılık | Geçici kullanım ilişkisi |
| `..` | Gerçekleştirme | Arayüz implementasyonu |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |

---

**Doküman Sonu**
