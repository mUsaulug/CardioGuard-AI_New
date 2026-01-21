# CardioGuard-AI
# Sınıf Diyagramı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sınıf Diyagramı (Class Diagram)  
**Versiyon:** 2.0.0  
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

Model paketi, yapay zeka modellerinin temel yapı taşlarını içerir. PyTorch `nn.Module` sınıfından türetilen model bileşenleri modüler ve yeniden kullanılabilir şekilde tasarlanmıştır.

### 2.1 Sınıf Diyagramı

```mermaid
classDiagram
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
        +forward(x: Tensor) Tensor
    }
    
    class LocalizationHead {
        <<nn.Module>>
        -Linear regressor
        +__init__(in_features: int, output_dim: int)
        +forward(x: Tensor) Tensor
    }
    
    class ECGCNN {
        <<nn.Module>>
        +ECGBackbone backbone
        +Module head
        +__init__(config: ECGCNNConfig, num_classes: int)
        +forward(x: Tensor) Tensor
    }
    
    class MultiTaskECGCNN {
        <<nn.Module>>
        +ECGBackbone backbone
        +Module head
        +LocalizationHead localization_head
        +__init__(config: ECGCNNConfig, num_classes: int)
        +forward(x: Tensor) Dict
    }
    
    ECGBackbone ..> ECGCNNConfig
    ECGCNN *-- ECGBackbone
    ECGCNN o-- BinaryHead
    ECGCNN o-- MultiClassHead
    MultiTaskECGCNN *-- ECGBackbone
    MultiTaskECGCNN *-- LocalizationHead
    MultiTaskECGCNN o-- BinaryHead
    MultiTaskECGCNN o-- MultiClassHead
    FiveClassHead --|> MultiClassHead
```

### 2.2 Model Varyantları

| Model | Head Tipi | Çıktı | Kullanım Amacı |
|-------|-----------|-------|----------------|
| Binary MI | BinaryHead | 1 | MI vs Normal sınıflandırma |
| Superclass | MultiClassHead | 4 | MI, STTC, CD, HYP çoklu-etiket |
| Lokalizasyon | FiveClassHead | 5 | AMI, ASMI, ALMI, IMI, LMI bölge tespiti |
| Multi-Task | LocalizationHead | 2 | Zamansal regresyon çıktısı |

### 2.3 ECGBackbone Mimarisi

ECGBackbone, 12 derivasyonlu EKG sinyallerinden (12 × 1000) 64 boyutlu gömme vektörü çıkaran evrişimsel sinir ağıdır.

| Katman | Tip | Parametreler |
|--------|-----|--------------|
| 1 | Conv1d | 12→64, kernel=7, padding=3 |
| 2 | BatchNorm1d | 64 |
| 3 | ReLU | inplace=False |
| 4 | Dropout | p=0.3 |
| 5 | Conv1d | 64→64, kernel=7, padding=3 |
| 6 | BatchNorm1d | 64 |
| 7 | ReLU | inplace=False |
| 8 | Dropout | p=0.3 |
| 9 | AdaptiveAvgPool1d | output=1 |

### 2.4 Yardımcı Fonksiyonlar

| Fonksiyon | Açıklama |
|-----------|----------|
| `build_classification_head()` | BinaryHead veya MultiClassHead oluşturur |
| `build_localization_head()` | LocalizationHead oluşturur |
| `build_sequential_cnn()` | Sequential(backbone, head) yapısı oluşturur |
| `build_multitask_cnn()` | MultiTaskECGCNN instance oluşturur |

---

## 3. Veri Paketi (src.data)

Veri paketi, EKG sinyallerinin yüklenmesi, işlenmesi ve etiketlenmesi süreçlerini yönetir. PyTorch Dataset arayüzüne uygun sınıflar içerir.

### 3.1 Sınıf Diyagramı

```mermaid
classDiagram
    class SignalDataset {
        <<Dataset>>
        -DataFrame df
        -Path base_path
        -str filename_column
        -str label_column
        -callable transform
        +__init__(df, base_path, ...)
        +__len__() int
        +__getitem__(idx: int) Tuple
        +__iter__()
    }
    
    class CachedSignalDataset {
        <<Dataset>>
        -ndarray signals
        -ndarray ecg_ids
        -Dict labels
        -callable transform
        +__init__(signals, ecg_ids, labels, transform)
        +__len__() int
        +__getitem__(idx: int) Tuple
    }
    
    class SignalLoader {
        <<module>>
        +load_single_signal(filename, base_path) ndarray
        +load_signals_batch(df, base_path) ndarray
        +build_npz_cache(df, base_path, cache_path)
        +load_npz_cache(cache_path) Tuple
    }
    
    class LabelProcessor {
        <<module>>
        +add_binary_mi_labels(df, scp_df) DataFrame
        +add_superclass_labels(df, scp_df) DataFrame
        +add_5class_labels(df, scp_df) DataFrame
        +extract_codes_above_threshold(scp_codes, min_likelihood) Set
    }
    
    class MILocalizationProcessor {
        <<module>>
        +List MI_LOCALIZATION_REGIONS
        +Dict MI_CODE_TO_REGIONS
        +List EXCLUDED_MI_CODES
        +extract_mi_regions(scp_codes) List
        +extract_mi_localization_labels(df) ndarray
        +add_mi_localization_labels(df) DataFrame
        +get_mi_localization_mask(y_multi4) ndarray
    }
    
    class Normalizer {
        <<module>>
        +min_max_normalize(signal) ndarray
        +z_score_normalize(signal) ndarray
        +per_lead_normalize(signal) ndarray
    }
    
    SignalDataset ..> SignalLoader
    SignalDataset ..> LabelProcessor
    CachedSignalDataset ..> SignalLoader
    MILocalizationProcessor ..> LabelProcessor
```

### 3.2 Etiketleme Stratejileri

| Strateji | Açıklama | Kullanım |
|----------|----------|----------|
| Binary | MI=1, NORM=0, excluded=-1 | Binary MI modeli |
| Superclass | 4 sınıf multi-hot [MI, STTC, CD, HYP] | Superclass modeli |
| Lokalizasyon | 5 bölge multi-hot [AMI, ASMI, ALMI, IMI, LMI] | MI lokalizasyon modeli |

### 3.3 MI Lokalizasyon Bölgeleri

| Bölge Kodu | Açıklama | İlişkili PTB-XL Kodları |
|------------|----------|-------------------------|
| AMI | Anterior MI | AMI |
| ASMI | Anteroseptal MI | ASMI, INJAS |
| ALMI | Anterolateral MI | ALMI, INJAL |
| IMI | Inferior MI | IMI, IPMI, INJIN |
| LMI | Lateral MI | LMI, INJLA |

---

## 4. Pipeline Paketi (src.pipeline)

Pipeline paketi, tahmin ve eğitim iş akışlarını yöneten sınıfları içerir. Consistency Guard mekanizması burada tanımlanmıştır.

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
        +bool superclass_mi_decision
        +bool binary_mi_decision
        +AgreementType agreement
        +str triage_level
        +List~str~ warnings
        +to_dict() Dict
    }
    
    class ConsistencyGuardModule {
        <<module>>
        +check_consistency(superclass_mi_prob, binary_mi_prob) ConsistencyResult
        +should_run_localization(consistency, gate_mode) bool
        +derive_norm_from_superclass(superclass_probs, threshold) Dict
    }
    
    ConsistencyResult *-- AgreementType
    ConsistencyGuardModule ..> ConsistencyResult
```

### 4.2 Uyum Tipleri

| Tip | Super MI | Binary MI | Triaj | Aksiyon |
|-----|----------|-----------|-------|---------|
| AGREE_MI | + | + | HIGH | Lokalizasyon çalıştır |
| AGREE_NO_MI | - | - | LOW | Normal raporla |
| DISAGREE_TYPE_1 | + | - | REVIEW | Lokalizasyon çalıştır, low confidence |
| DISAGREE_TYPE_2 | - | + | REVIEW | Manuel inceleme gerekli |

### 4.3 Gate Mode Stratejileri

| Mod | Açıklama |
|-----|----------|
| `strict` | Her iki model de MI tespit etmeli |
| `recall_first` | Sadece superclass MI yeterli |

---

## 5. XAI Paketi (src.xai)

Açıklanabilir yapay zeka paketi, model kararlarının görselleştirilmesini ve doğrulanmasını sağlar.

### 5.1 Sınıf Diyagramı

```mermaid
classDiagram
    class GradCAM {
        -Module model
        -Module target_layer
        -Tensor gradients
        -Tensor activations
        +__init__(model, target_layer)
        -_register_hooks()
        +generate(inputs, class_index) ndarray
    }
    
    class XAISanityChecker {
        -Module model
        -int window_ms
        -int sampling_rate
        +__init__(model, window_ms, sampling_rate)
        +run_checks(input_tensor, original_explanation, explanation_func) Dict
        -_check_model_randomization() Dict
        -_check_faithfulness() Dict
        -_check_input_perturbation() Dict
        -_compute_similarity(exp1, exp2) float
        -_create_baseline(input_np) ndarray
        -_compute_overall_assessment(results) Dict
    }
    
    class XAIReporter {
        -str run_id
        -Path output_dir
        -str task
        -List cards
        +__init__(run_id, output_dir, task, ...)
        +add_sample(sample_id, explanation, sanity, prediction)
        +save_visual_report(sample_id, figure_path)
        +finalize() Path
        -_create_card() Dict
        -_save_tensors()
        -_create_summary_tables()
        -_create_manifest()
    }
    
    class XAIVisualizer {
        <<module>>
        +plot_12lead_gradcam(signal, gradcam_maps, output_path)
        +plot_ecg_with_localization(signal, localization_probs, output_path)
        +plot_ecg_with_prediction(signal, prediction, output_path)
        +plot_gradcam_heatmap(signal, cam, save_path)
        +plot_lead_attention(attention_scores, output_path)
        +generate_xai_report_png(signal, combined_heatmap, ...)
    }
    
    class SHAPModule {
        <<module>>
        +explain_xgb(model, X) Dict
        +plot_shap_summary(shap_values, features) Figure
        +plot_shap_waterfall(shap_values, base_value, sample_idx) Figure
        +get_top_features(shap_values) List
    }
    
    XAIVisualizer ..> GradCAM
    XAIReporter ..> XAISanityChecker
    XAIReporter ..> XAIVisualizer
```

### 5.2 Sanity Check Testleri

| Test | Açıklama | Başarı Kriteri |
|------|----------|----------------|
| Model Randomization | Rastgele ağırlıklarla açıklama değişmeli | Spearman korelasyon < 0.5 |
| Faithfulness Deletion | Önemli bölgeler maskelendiğinde skor düşmeli | Score drop > 0.1 |
| Faithfulness Insertion | Önemli bölgeler eklendiğinde skor artmalı | Score rise > 0.1 |
| Input Perturbation | Küçük gürültüyle açıklama stabil kalmalı | Similarity > 0.7 |

### 5.3 Açıklama Yöntemleri

| Yöntem | Model | Çıktı | Görselleştirme |
|--------|-------|-------|----------------|
| Grad-CAM | CNN | Isı haritası (1000,) | Sinyal üzeri overlay |
| SmoothGrad-CAM | CNN | Ortalanmış ısı haritası | Daha stabil overlay |
| SHAP TreeExplainer | XGBoost | Özellik önemi | Çubuk grafik / Waterfall |

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
        +Dict thresholds
        +bool is_loaded
        +str model_hash
        +str threshold_hash
        +__init__()
        +load_models(superclass_checkpoint, binary_checkpoint, ...)
    }
    
    class PredictionProbabilities {
        <<Pydantic>>
        +float MI
        +float STTC
        +float CD
        +float HYP
        +float NORM
    }
    
    class PrimaryPrediction {
        <<Pydantic>>
        +str label
        +float confidence
        +str rule
    }
    
    class SourceProbabilities {
        <<Pydantic>>
        +Dict cnn
        +Dict xgb
        +Dict ensemble
    }
    
    class VersionInfo {
        <<Pydantic>>
        +str model_hash
        +str threshold_hash
        +str api_version
        +str timestamp
    }
    
    class SuperclassPredictionResponse {
        <<Pydantic>>
        +str mode
        +PredictionProbabilities probabilities
        +List predicted_labels
        +Dict thresholds
        +PrimaryPrediction primary
        +SourceProbabilities sources
        +VersionInfo versions
    }
    
    class MILocalizationResponse {
        <<Pydantic>>
        +bool mi_detected
        +Dict region_probabilities
        +List predicted_regions
        +str label_space
        +str mapping_fingerprint
        +str localization_head_type
    }
    
    class HealthResponse {
        <<Pydantic>>
        +str status
        +str timestamp
    }
    
    class ReadyResponse {
        <<Pydantic>>
        +bool ready
        +Dict models_loaded
        +str message
    }
    
    SuperclassPredictionResponse *-- PredictionProbabilities
    SuperclassPredictionResponse *-- PrimaryPrediction
    SuperclassPredictionResponse *-- SourceProbabilities
    SuperclassPredictionResponse *-- VersionInfo
    AppState ..> SuperclassPredictionResponse
    AppState ..> MILocalizationResponse
```

### 6.2 API Endpoints

| Endpoint | Metod | Response Sınıfı | Açıklama |
|----------|-------|-----------------|----------|
| /predict/superclass | POST | SuperclassPredictionResponse | Çoklu-etiket patoloji tahmini |
| /predict/mi-localization | POST | MILocalizationResponse | MI bölge lokalizasyonu |
| /health | GET | HealthResponse | Servis sağlık kontrolü |
| /ready | GET | ReadyResponse | Model yüklenme durumu |

---

## 7. Contracts Paketi (src.contracts)

API kontratları ve veri dönüşüm fonksiyonlarını içerir.

### 7.1 Sabitler

| Sabit | Değer | Açıklama |
|-------|-------|----------|
| AIRESULT_VERSION | "1.0" | Kontrat versiyonu |
| PATHOLOGY_CLASSES | ["MI", "STTC", "CD", "HYP"] | Patoloji sınıfları |
| MI_LOCALIZATION_LABELS | ["AMI", "ASMI", "ALMI", "IMI", "LMI"] | Lokalizasyon etiketleri |
| LEADS | ["I", "II", ..., "V6"] | 12 derivasyon isimleri |

### 7.2 Fonksiyonlar

| Fonksiyon | Açıklama |
|-----------|----------|
| `map_predict_output_to_airesult()` | Raw tahmin → AIResult v1.0 dönüşümü |
| `compute_triage()` | Triaj seviyesi hesaplama |
| `discover_xai_artifacts()` | XAI çıktı dosyalarını bulma |

---

## 8. Planlanan Sınıflar (v2.0)

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

## 9. UML Notasyonu

| Sembol | Anlam | Açıklama |
|--------|-------|----------|
| `*--` | Kompozisyon | Parça bütüne bağımlı yaşar |
| `o--` | Agregasyon | Parça bağımsız yaşayabilir |
| `--|>` | Kalıtım | Alt sınıf üst sınıftan türer |
| `..>` | Bağımlılık | Geçici kullanım ilişkisi |
| `<<stereotype>>` | Stereotip | Sınıf türü (dataclass, module, vb.) |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |

---

**Doküman Sonu**
