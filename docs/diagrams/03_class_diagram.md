# CardioGuard-AI
# Sınıf Diyagramı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sınıf Diyagramı (Class Diagram)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## 1. Genel Bakış

CardioGuard-AI, modüler bir mimari üzerine inşa edilmiştir. Sistem, birbirine bağımlı ancak ayrı sorumluluklar taşıyan yedi ana paketten oluşmaktadır. Bu doküman, her paketin sınıf yapısını UML standartlarına uygun şekilde tanımlamaktadır.

### 1.1 Paket Bağımlılıkları

```mermaid
graph TD
    BACKEND["src.backend<br/>Web Servisi"]
    PIPELINE["src.pipeline<br/>İş Akışları"]
    MODELS["src.models<br/>Model Tanımları"]
    DATA["src.data<br/>Veri İşleme"]
    XAI["src.xai<br/>Açıklanabilirlik"]
    CONTRACTS["src.contracts<br/>API Kontratları"]
    UTILS["src.utils<br/>Yardımcılar"]
    
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

## 2. Model Paketi (src.models)

Model paketi, EKG sinyallerini analiz eden derin öğrenme modellerini içermektedir. Merkezi bileşen olan ECGBackbone, tüm modeller tarafından paylaşılan özellik çıkarıcı olarak görev yapmaktadır.

### 2.1 Sınıf Diyagramı

```mermaid
classDiagram
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
        +forward(x) Tensor
    }
    
    class MultiClassHead {
        <<nn.Module>>
        -Linear classifier
        +forward(x) Tensor
    }
    
    class FiveClassHead {
        <<nn.Module>>
        +forward(x) Tensor
    }
    
    class ECGCNN {
        <<nn.Module>>
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

### 2.2 Model Varyantları

| Model | Head Tipi | Çıktı | Kullanım Amacı |
|-------|-----------|-------|----------------|
| Binary MI | BinaryHead | 1 | MI vs Normal sınıflandırma |
| Superclass | MultiClassHead | 4 | MI, STTC, CD, HYP çoklu-etiket |
| Lokalizasyon | FiveClassHead | 5 | 5 anatomik bölge tespiti |

### 2.3 ECGBackbone Mimarisi

ECGBackbone, 12 derivasyonlu EKG sinyallerinden 64 boyutlu gömme vektörü çıkaran evrişimsel sinir ağıdır.

| Katman | Tip | Parametreler |
|--------|-----|--------------|
| 1 | Conv1d | 12→64, kernel=7 |
| 2 | BatchNorm1d | 64 |
| 3 | ReLU | - |
| 4 | Dropout | p=0.3 |
| 5 | Conv1d | 64→64, kernel=7 |
| 6 | BatchNorm1d | 64 |
| 7 | ReLU | - |
| 8 | Dropout | p=0.3 |
| 9 | AdaptiveAvgPool1d | output=1 |

---

## 3. Veri Paketi (src.data)

Veri paketi, PTB-XL veri setinin yüklenmesi, etiketlenmesi ve ön işlenmesinden sorumludur.

### 3.1 Sınıf Diyagramı

```mermaid
classDiagram
    class SignalDataset {
        <<Dataset>>
        -DataFrame df
        -Path base_path
        +__len__() int
        +__getitem__(idx) Tuple
    }
    
    class LabelProcessor {
        +add_binary_mi_labels()
        +add_superclass_labels()
    }
    
    class MILocalizationProcessor {
        +List MI_REGIONS
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
```

### 3.2 Etiketleme Stratejileri

| Strateji | Açıklama | Kullanım |
|----------|----------|----------|
| Binary | MI=1, NORM=0 | Binary MI modeli |
| Superclass | 4 sınıf multi-hot | Superclass modeli |
| Lokalizasyon | 5 bölge multi-hot | MI lokalizasyon modeli |

---

## 4. Pipeline Paketi (src.pipeline)

Pipeline paketi, model eğitimi ve çıkarım iş akışlarını yönetmektedir.

### 4.1 Tutarlılık Kontrolü Sınıfları

```mermaid
classDiagram
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
        +AgreementType agreement
        +str triage_level
        +to_dict() Dict
    }
    
    class ConsistencyGuard {
        +check_consistency()
        +should_run_localization()
        +derive_norm()
    }
    
    ConsistencyResult *-- AgreementType
    ConsistencyGuard ..> ConsistencyResult
```

### 4.2 Uyum Tipleri

| Tip | Super MI | Binary MI | Triaj | Aksiyon |
|-----|----------|-----------|-------|---------|
| AGREE_MI | + | + | Yüksek | Lokalizasyon çalıştır |
| AGREE_NO_MI | - | - | Düşük | Normal raporla |
| DISAGREE_TYPE_1 | + | - | İnceleme | Lokalizasyon çalıştır |
| DISAGREE_TYPE_2 | - | + | İnceleme | Manuel inceleme |

---

## 5. XAI Paketi (src.xai)

Açıklanabilir yapay zeka paketi, model kararlarının görselleştirilmesini sağlar.

### 5.1 Sınıf Diyagramı

```mermaid
classDiagram
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

### 5.2 Açıklama Yöntemleri

| Yöntem | Model | Çıktı | Görselleştirme |
|--------|-------|-------|----------------|
| Grad-CAM | CNN | Isı haritası | Sinyal üzeri overlay |
| SHAP | XGBoost | Özellik önemi | Çubuk grafik |

---

## 6. Backend Paketi (src.backend)

Backend paketi, FastAPI tabanlı REST API servisini içermektedir.

### 6.1 Sınıf Diyagramı

```mermaid
classDiagram
    class AppState {
        +Module superclass_model
        +Module binary_model
        +Module localization_model
        +Dict xgb_models
        +bool is_loaded
        +load_models()
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
        +str triage_level
    }
    
    SuperclassPredictionResponse *-- PredictionProbabilities
    AppState ..> SuperclassPredictionResponse
```

### 6.2 API Endpoints

| Endpoint | Metod | Response Sınıfı |
|----------|-------|-----------------|
| /predict/superclass | POST | SuperclassPredictionResponse |
| /predict/mi-localization | POST | MILocalizationResponse |
| /health | GET | HealthResponse |
| /ready | GET | ReadyResponse |

---

## 7. Planlanan Sınıflar (v2.0)

```mermaid
classDiagram
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
    }
    
    class TransformerBackbone {
        <<planned>>
        +forward()
        +get_attention_weights()
    }
    
    RAGRetriever ..> LLMReportGenerator
    TransformerBackbone --|> ECGBackbone
```

---

## 8. UML Notasyonu

| Sembol | Anlam | Açıklama |
|--------|-------|----------|
| `*--` | Kompozisyon | Parça bütüne bağımlı yaşar |
| `o--` | Agregasyon | Parça bağımsız yaşayabilir |
| `--|>` | Kalıtım | Alt sınıf üst sınıftan türer |
| `..>` | Bağımlılık | Geçici kullanım ilişkisi |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |

---

**Doküman Sonu**
