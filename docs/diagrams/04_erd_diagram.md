# CardioGuard-AI: ERD (Varlık-İlişki Diyagramı)
## (Entity-Relationship Diagram)

---

## 📋 Doküman Bilgileri

| Özellik | Değer |
|---------|-------|
| **Proje Adı** | CardioGuard-AI |
| **Doküman Tipi** | ERD (Varlık-İlişki Diyagramı) |
| **Versiyon** | 1.0.0 |
| **Tarih** | 2026-01-21 |

---

## 1. Genel Bakış

CardioGuard-AI sistemi, PTB-XL veri setini kullanarak EKG sinyallerini işler ve tahmin sonuçları üretir. Bu ERD, sistemdeki tüm veri varlıklarını ve ilişkilerini gösterir.

---

## 2. Tam ERD Diyagramı

```mermaid
erDiagram
    PATIENT ||--o{ ECG_RECORD : "has"
    ECG_RECORD ||--o{ SCP_CODE_ASSIGNMENT : "contains"
    SCP_STATEMENT ||--o{ SCP_CODE_ASSIGNMENT : "referenced by"
    ECG_RECORD ||--o| SUPERCLASS_LABEL : "has"
    ECG_RECORD ||--o| MI_LOCALIZATION_LABEL : "has (if MI)"
    ECG_RECORD ||--o{ PREDICTION_REQUEST : "analyzed by"
    PREDICTION_REQUEST ||--|| PREDICTION_RESULT : "produces"
    PREDICTION_RESULT ||--o{ MI_LOCALIZATION_RESULT : "includes (if MI)"
    PREDICTION_RESULT ||--o{ XAI_ARTIFACT : "generates"
    MODEL_CHECKPOINT ||--o{ PREDICTION_REQUEST : "used by"
    THRESHOLD_CONFIG ||--o{ PREDICTION_REQUEST : "applied to"
    
    PATIENT {
        int patient_id PK "Hasta benzersiz kimliği"
        float age "Yaş"
        string sex "Cinsiyet (M/F)"
        float height "Boy (cm)"
        float weight "Kilo (kg)"
    }
    
    ECG_RECORD {
        int ecg_id PK "EKG kayıt kimliği"
        int patient_id FK "Hasta referansı"
        string filename_lr "100Hz dosya yolu"
        string filename_hr "500Hz dosya yolu"
        datetime recording_date "Kayıt tarihi"
        int strat_fold "Çapraz doğrulama katmanı (1-10)"
        string device "Kayıt cihazı"
        float baseline_drift "Baseline kayması"
        float static_noise "Statik gürültü"
        float burst_noise "Ani gürültü"
        float electrodes_problems "Elektrot sorunları"
        string validated_by "Doğrulayan uzman"
    }
    
    SCP_STATEMENT {
        string scp_code PK "SCP kodu (ör: AMI, NORM)"
        string description "Açıklama"
        string diagnostic_class "Tanı sınıfı (ör: MI)"
        string diagnostic_subclass "Alt sınıf"
        string form_statement "Form ifadesi"
        string rhythm_statement "Ritim ifadesi"
        bool is_diagnostic "Tanısal mı?"
    }
    
    SCP_CODE_ASSIGNMENT {
        int assignment_id PK "Atama kimliği"
        int ecg_id FK "EKG referansı"
        string scp_code FK "SCP kodu referansı"
        float likelihood "Olasılık (0-100)"
    }
    
    SUPERCLASS_LABEL {
        int ecg_id PK,FK "EKG referansı"
        bool label_MI "MI etiketi"
        bool label_STTC "STTC etiketi"
        bool label_CD "CD etiketi"
        bool label_HYP "HYP etiketi"
        bool is_norm "Türetilmiş NORM"
        string primary_superclass "Birincil süpersınıf"
    }
    
    MI_LOCALIZATION_LABEL {
        int ecg_id PK,FK "EKG referansı"
        bool label_AMI "Anterior MI"
        bool label_ASMI "Anteroseptal MI"
        bool label_ALMI "Anterolateral MI"
        bool label_IMI "Inferior MI"
        bool label_LMI "Lateral MI"
        string primary_region "Birincil bölge"
    }
    
    PREDICTION_REQUEST {
        string request_id PK "İstek UUID"
        string case_id "Vaka kimliği"
        int ecg_id FK "İlişkili EKG (opsiyonel)"
        string model_version FK "Model versiyonu"
        datetime created_at "Oluşturulma zamanı"
        string input_format "Girdi formatı (npz/npy)"
        int sample_rate_hz "Örnekleme hızı"
        float duration_sec "Süre (saniye)"
        string signal_shape "Sinyal boyutu"
    }
    
    PREDICTION_RESULT {
        string result_id PK "Sonuç kimliği"
        string request_id FK "İstek referansı"
        float prob_MI "MI olasılığı"
        float prob_STTC "STTC olasılığı"
        float prob_CD "CD olasılığı"
        float prob_HYP "HYP olasılığı"
        float prob_NORM "Türetilmiş NORM"
        string predicted_labels "Tahmin edilen etiketler (JSON)"
        string primary_label "Birincil etiket"
        float primary_confidence "Birincil güven"
        string triage_level "Triaj seviyesi"
        string agreement_type "Model uyum tipi"
        float cnn_prob_MI "CNN MI olasılığı"
        float xgb_prob_MI "XGB MI olasılığı"
        float ensemble_weight "Ensemble ağırlığı (α)"
    }
    
    MI_LOCALIZATION_RESULT {
        string localization_id PK "Lokalizasyon kimliği"
        string result_id FK "Sonuç referansı"
        float prob_AMI "AMI olasılığı"
        float prob_ASMI "ASMI olasılığı"
        float prob_ALMI "ALMI olasılığı"
        float prob_IMI "IMI olasılığı"
        float prob_LMI "LMI olasılığı"
        string detected_regions "Tespit edilen bölgeler (JSON)"
        string label_space "Etiket uzayı ID"
        string mapping_fingerprint "Mapping parmak izi"
    }
    
    XAI_ARTIFACT {
        string artifact_id PK "Artifact kimliği"
        string result_id FK "Sonuç referansı"
        string artifact_type "Tip (gradcam/shap/narrative)"
        string file_path "Dosya yolu"
        string target_class "Hedef sınıf"
        datetime created_at "Oluşturulma zamanı"
        int file_size_bytes "Dosya boyutu"
    }
    
    MODEL_CHECKPOINT {
        string checkpoint_id PK "Checkpoint kimliği"
        string model_type "Model tipi (binary/superclass/localization)"
        string file_path "Dosya yolu"
        string model_hash "Model hash"
        int output_dimension "Çıktı boyutu"
        datetime trained_at "Eğitim tarihi"
        string training_config "Eğitim konfigürasyonu (JSON)"
        float validation_auroc "Doğrulama AUROC"
    }
    
    THRESHOLD_CONFIG {
        string config_id PK "Konfigürasyon kimliği"
        string config_hash "Konfigürasyon hash"
        float threshold_MI "MI eşiği"
        float threshold_STTC "STTC eşiği"
        float threshold_CD "CD eşiği"
        float threshold_HYP "HYP eşiği"
        float superclass_mi_threshold "Superclass MI eşiği"
        float binary_mi_threshold "Binary MI eşiği"
        datetime optimized_at "Optimizasyon tarihi"
    }
```

---

## 3. Varlık Detayları

### 3.1 PATIENT (Hasta)

| Alan | Tip | Açıklama | Kısıtlar |
|------|-----|----------|----------|
| `patient_id` | INT | Birincil anahtar | PK, NOT NULL, AUTO_INCREMENT |
| `age` | FLOAT | Yaş (yıl) | CHECK (age >= 0 AND age <= 120) |
| `sex` | VARCHAR(1) | Cinsiyet | CHECK (sex IN ('M', 'F')) |
| `height` | FLOAT | Boy (cm) | NULLABLE |
| `weight` | FLOAT | Kilo (kg) | NULLABLE |

**İstatistikler (PTB-XL):**
- Toplam: 18,885 benzersiz hasta
- Yaş aralığı: 18-89
- Cinsiyet dağılımı: ~52% Erkek, ~48% Kadın

---

### 3.2 ECG_RECORD (EKG Kaydı)

| Alan | Tip | Açıklama | Kısıtlar |
|------|-----|----------|----------|
| `ecg_id` | INT | Birincil anahtar | PK, NOT NULL |
| `patient_id` | INT | Hasta referansı | FK -> PATIENT |
| `filename_lr` | VARCHAR(255) | 100Hz dosya yolu | NOT NULL |
| `filename_hr` | VARCHAR(255) | 500Hz dosya yolu | NOT NULL |
| `strat_fold` | INT | Çapraz doğrulama katmanı | CHECK (strat_fold BETWEEN 1 AND 10) |

**Veri Bölümlemesi:**
```mermaid
pie title Strat Fold Dağılımı
    "Fold 1-8 (Train)" : 17469
    "Fold 9 (Val)" : 2189
    "Fold 10 (Test)" : 2179
```

---

### 3.3 SCP_STATEMENT (SCP İfadesi)

PTB-XL veri setindeki standart SCP kodları:

| Kategori | Kodlar | Açıklama |
|----------|--------|----------|
| **NORM** | NORM | Normal EKG |
| **MI** | AMI, IMI, ASMI, ALMI, LMI, ILMI, IPLMI, IPMI | Miyokard Enfarktüsü |
| **STTC** | NDT, NST_, ISCA, ISCI, ISC_, STD_, STE_ | ST/T Değişikliği |
| **CD** | CLBBB, CRBBB, IRBBB, 1AVB, 2AVB, 3AVB | İletim Bozukluğu |
| **HYP** | LVH, RVH, SEHYP, LAO/LAE, RAO/RAE | Hipertrofi |

---

### 3.4 SUPERCLASS_LABEL (Süpersınıf Etiketi)

```mermaid
graph LR
    subgraph "Çoklu-Etiket Yapısı"
        MI["MI: 0/1"]
        STTC["STTC: 0/1"]
        CD["CD: 0/1"]
        HYP["HYP: 0/1"]
    end
    
    subgraph "Türetilmiş"
        NORM["NORM = !any(MI, STTC, CD, HYP)"]
    end
    
    MI & STTC & CD & HYP --> NORM
```

**Etiket Dağılımı:**
| Sınıf | Sayı | Oran |
|-------|------|------|
| MI | 5,486 | 25.1% |
| STTC | 5,250 | 24.0% |
| CD | 4,907 | 22.5% |
| HYP | 2,655 | 12.2% |
| NORM | 9,528 | 43.6% |

> **Not:** Toplamlar %100'ü aşar çünkü çoklu-etiket yapısı kullanılmaktadır.

---

### 3.5 MI_LOCALIZATION_LABEL (MI Lokalizasyon Etiketi)

```mermaid
graph TB
    subgraph "SCP Kodları"
        AMI_CODE["AMI"]
        IMI_CODE["IMI"]
        ILMI_CODE["ILMI"]
        IPLMI_CODE["IPLMI"]
        INJXX["INJIN, INJAL, INJAS..."]
    end
    
    subgraph "Türetilmiş Bölgeler"
        AMI["AMI - Anterior"]
        ASMI["ASMI - Anteroseptal"]
        ALMI["ALMI - Anterolateral"]
        IMI["IMI - Inferior"]
        LMI["LMI - Lateral"]
    end
    
    AMI_CODE --> AMI
    IMI_CODE --> IMI
    ILMI_CODE --> IMI & LMI
    IPLMI_CODE --> IMI & LMI
    INJXX --> AMI & ASMI & ALMI & IMI & LMI
    
    style AMI fill:#ffcdd2
    style IMI fill:#c8e6c9
```

**Mapping Kuralları:**
```
MI_CODE_TO_REGIONS = {
    "AMI": ["AMI"],
    "ASMI": ["ASMI"],
    "ALMI": ["ALMI"],
    "IMI": ["IMI"],
    "LMI": ["LMI"],
    "ILMI": ["IMI", "LMI"],      # Inferolateral -> 2 bölge
    "IPLMI": ["IMI", "LMI"],     # Inferoposterolateral -> 2 bölge
    "IPMI": ["IMI"],             # Inferoposterior -> Inferior
}
```

---

### 3.6 PREDICTION_RESULT (Tahmin Sonucu)

```mermaid
graph TB
    subgraph Sources["Kaynak Olasılıklar"]
        CNN["CNN Probs"]
        XGB["XGB Probs"]
    end
    
    subgraph Ensemble["Ensemble"]
        ENS["P_final = α×P_cnn + (1-α)×P_xgb<br/>α = 0.15"]
    end
    
    subgraph Output["Çıktılar"]
        PROBS["Olasılıklar"]
        LABELS["Etiketler"]
        TRIAGE["Triaj"]
    end
    
    CNN & XGB --> ENS --> PROBS --> LABELS --> TRIAGE
```

---

## 4. İlişki Detayları

### 4.1 Kardinalite Tablosu

| İlişki | Tip | Açıklama |
|--------|-----|----------|
| PATIENT → ECG_RECORD | 1:N | Bir hasta birden fazla EKG kaydına sahip olabilir |
| ECG_RECORD → SCP_CODE_ASSIGNMENT | 1:N | Bir EKG birden fazla SCP koduna sahip olabilir |
| ECG_RECORD → SUPERCLASS_LABEL | 1:1 | Her EKG'nin bir süpersınıf etiketi var |
| ECG_RECORD → MI_LOCALIZATION_LABEL | 1:0..1 | MI tespit edilirse lokalizasyon etiketi var |
| ECG_RECORD → PREDICTION_REQUEST | 1:N | Bir EKG birden fazla kez analiz edilebilir |
| PREDICTION_REQUEST → PREDICTION_RESULT | 1:1 | Her istek bir sonuç üretir |
| PREDICTION_RESULT → MI_LOCALIZATION_RESULT | 1:0..1 | MI tespit edilirse lokalizasyon sonucu var |
| PREDICTION_RESULT → XAI_ARTIFACT | 1:N | Bir sonuç birden fazla XAI artifact'ı üretebilir |

---

## 5. İndeksler ve Performans

### 5.1 Önerilen İndeksler

```sql
-- Hasta aramaları için
CREATE INDEX idx_patient_age ON PATIENT(age);
CREATE INDEX idx_patient_sex ON PATIENT(sex);

-- EKG kayıt aramaları için
CREATE INDEX idx_ecg_patient ON ECG_RECORD(patient_id);
CREATE INDEX idx_ecg_fold ON ECG_RECORD(strat_fold);

-- SCP kod aramaları için
CREATE INDEX idx_scp_assignment_ecg ON SCP_CODE_ASSIGNMENT(ecg_id);
CREATE INDEX idx_scp_assignment_code ON SCP_CODE_ASSIGNMENT(scp_code);

-- Tahmin aramaları için
CREATE INDEX idx_prediction_request_created ON PREDICTION_REQUEST(created_at);
CREATE INDEX idx_prediction_result_triage ON PREDICTION_RESULT(triage_level);
CREATE INDEX idx_prediction_result_mi ON PREDICTION_RESULT(prob_MI);
```

### 5.2 Sorgu Optimizasyonu

```mermaid
graph LR
    subgraph "Sık Kullanılan Sorgular"
        Q1["MI pozitif EKG'leri bul"]
        Q2["Hasta bazlı sonuçlar"]
        Q3["Triaj bazlı filtreleme"]
        Q4["Zaman bazlı raporlama"]
    end
    
    subgraph "İndeksler"
        I1["idx_ecg_patient"]
        I2["idx_prediction_result_mi"]
        I3["idx_prediction_result_triage"]
        I4["idx_prediction_request_created"]
    end
    
    Q1 --> I2
    Q2 --> I1
    Q3 --> I3
    Q4 --> I4
```

---

## 6. Veri Bütünlüğü Kuralları

### 6.1 Foreign Key Kısıtları

```sql
-- EKG -> Hasta ilişkisi
ALTER TABLE ECG_RECORD
ADD CONSTRAINT fk_ecg_patient
FOREIGN KEY (patient_id) REFERENCES PATIENT(patient_id)
ON DELETE RESTRICT ON UPDATE CASCADE;

-- Tahmin Sonucu -> İstek ilişkisi
ALTER TABLE PREDICTION_RESULT
ADD CONSTRAINT fk_result_request
FOREIGN KEY (request_id) REFERENCES PREDICTION_REQUEST(request_id)
ON DELETE CASCADE ON UPDATE CASCADE;
```

### 6.2 Check Kısıtları

```sql
-- Olasılık değerleri 0-1 arasında olmalı
ALTER TABLE PREDICTION_RESULT
ADD CONSTRAINT chk_prob_range
CHECK (prob_MI BETWEEN 0 AND 1
   AND prob_STTC BETWEEN 0 AND 1
   AND prob_CD BETWEEN 0 AND 1
   AND prob_HYP BETWEEN 0 AND 1);

-- Triaj seviyeleri geçerli olmalı
ALTER TABLE PREDICTION_RESULT
ADD CONSTRAINT chk_triage_level
CHECK (triage_level IN ('HIGH', 'MEDIUM', 'LOW', 'REVIEW'));

-- Eşik değerleri 0-1 arasında olmalı
ALTER TABLE THRESHOLD_CONFIG
ADD CONSTRAINT chk_threshold_range
CHECK (threshold_MI BETWEEN 0 AND 1
   AND threshold_STTC BETWEEN 0 AND 1);
```

---

## 7. Veri Akışı

```mermaid
flowchart TB
    subgraph Input["Girdi Katmanı"]
        PTBXL["PTB-XL<br/>Ham Veriler"]
    end
    
    subgraph Processing["İşleme Katmanı"]
        PATIENT_TBL["PATIENT"]
        ECG_TBL["ECG_RECORD"]
        SCP_TBL["SCP_STATEMENT"]
        LABEL_TBL["SUPERCLASS_LABEL<br/>MI_LOCALIZATION_LABEL"]
    end
    
    subgraph Inference["Çıkarım Katmanı"]
        REQUEST["PREDICTION_REQUEST"]
        RESULT["PREDICTION_RESULT"]
        MILOC["MI_LOCALIZATION_RESULT"]
        XAI["XAI_ARTIFACT"]
    end
    
    subgraph Config["Konfigürasyon"]
        MODEL["MODEL_CHECKPOINT"]
        THRESH["THRESHOLD_CONFIG"]
    end
    
    PTBXL --> PATIENT_TBL & ECG_TBL & SCP_TBL
    ECG_TBL --> LABEL_TBL
    SCP_TBL --> LABEL_TBL
    
    ECG_TBL --> REQUEST
    MODEL & THRESH --> REQUEST
    REQUEST --> RESULT
    RESULT --> MILOC & XAI
    
    style PTBXL fill:#e3f2fd
    style RESULT fill:#e8f5e9
```

---

## 8. Örnek Veri

### 8.1 Örnek EKG Kaydı

```json
{
  "ecg_id": 1,
  "patient_id": 15709,
  "filename_lr": "records100/00000/00001_lr",
  "filename_hr": "records500/00000/00001_hr",
  "strat_fold": 3,
  "scp_codes": {
    "AMI": 80.0,
    "IMI": 100.0
  }
}
```

### 8.2 Örnek Tahmin Sonucu

```json
{
  "result_id": "res_abc123",
  "request_id": "req_xyz789",
  "prob_MI": 0.85,
  "prob_STTC": 0.12,
  "prob_CD": 0.08,
  "prob_HYP": 0.05,
  "prob_NORM": 0.15,
  "predicted_labels": ["MI"],
  "primary_label": "MI",
  "primary_confidence": 0.85,
  "triage_level": "HIGH",
  "agreement_type": "AGREE_MI"
}
```

---

> **Not:** Bu ERD, CardioGuard-AI v1.0.0 veri modelini temsil eder. Veritabanı şeması, file-based storage kullanıldığından kavramsal düzeydedir. Üretim ortamında PostgreSQL veya MongoDB kullanılması önerilir.
