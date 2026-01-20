# CardioGuard-AI
# Varlık-İlişki Diyagramı (ERD)

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Varlık-İlişki Diyagramı (Entity-Relationship Diagram)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## İçindekiler

1. [Genel Bakış](#1-genel-bakış)
2. [ERD Diyagramı](#2-erd-diyagramı)
3. [Varlık Detayları](#3-varlık-detayları)
4. [İlişki Detayları](#4-ilişki-detayları)
5. [Veri Bütünlüğü Kuralları](#5-veri-bütünlüğü-kuralları)
6. [Veri Akışı](#6-veri-akışı)
7. [Örnek Veri](#7-örnek-veri)

---

## 1. Genel Bakış

CardioGuard-AI sistemi, PTB-XL veri setini kullanarak EKG sinyallerini işler ve tahmin sonuçları üretir. Bu doküman, sistemdeki tüm veri varlıklarını ve ilişkilerini tanımlamaktadır.

### 1.1 Varlık Kategorileri

| Kategori | Varlıklar | Açıklama |
|----------|-----------|----------|
| Kaynak Veri | PATIENT, ECG_RECORD, SCP_STATEMENT | PTB-XL veritabanından gelen ham veriler |
| Etiketler | SUPERCLASS_LABEL, MI_LOCALIZATION_LABEL | İşlenmiş etiket verileri |
| Tahmin | PREDICTION_REQUEST, PREDICTION_RESULT | Tahmin işlem verileri |
| Çıktı | MI_LOCALIZATION_RESULT, XAI_ARTIFACT | Tahmin çıktıları |
| Konfigürasyon | MODEL_CHECKPOINT, THRESHOLD_CONFIG | Sistem yapılandırma verileri |

---

## 2. ERD Diyagramı

```mermaid
erDiagram
    PATIENT ||--o{ ECG_RECORD : "sahiptir"
    ECG_RECORD ||--o{ SCP_CODE_ASSIGNMENT : "içerir"
    SCP_STATEMENT ||--o{ SCP_CODE_ASSIGNMENT : "referans verilir"
    ECG_RECORD ||--o| SUPERCLASS_LABEL : "sahiptir"
    ECG_RECORD ||--o| MI_LOCALIZATION_LABEL : "sahiptir (MI durumunda)"
    ECG_RECORD ||--o{ PREDICTION_REQUEST : "analiz edilir"
    PREDICTION_REQUEST ||--|| PREDICTION_RESULT : "üretir"
    PREDICTION_RESULT ||--o{ MI_LOCALIZATION_RESULT : "içerir (MI durumunda)"
    PREDICTION_RESULT ||--o{ XAI_ARTIFACT : "oluşturur"
    MODEL_CHECKPOINT ||--o{ PREDICTION_REQUEST : "kullanılır"
    THRESHOLD_CONFIG ||--o{ PREDICTION_REQUEST : "uygulanır"
    
    PATIENT {
        int patient_id PK "Hasta benzersiz kimliği"
        float age "Yaş"
        string sex "Cinsiyet (E/K)"
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
        string validated_by "Doğrulayan uzman"
    }
    
    SCP_STATEMENT {
        string scp_code PK "SCP kodu"
        string description "Açıklama"
        string diagnostic_class "Tanı sınıfı"
        string diagnostic_subclass "Alt sınıf"
        bool is_diagnostic "Tanısal mı"
    }
    
    SCP_CODE_ASSIGNMENT {
        int assignment_id PK "Atama kimliği"
        int ecg_id FK "EKG referansı"
        string scp_code FK "SCP kodu referansı"
        float likelihood "Olasılık (0-100)"
    }
    
    SUPERCLASS_LABEL {
        int ecg_id PK "EKG referansı"
        bool label_MI "MI etiketi"
        bool label_STTC "STTC etiketi"
        bool label_CD "CD etiketi"
        bool label_HYP "HYP etiketi"
        bool is_norm "Türetilmiş NORM"
        string primary_superclass "Birincil süpersınıf"
    }
    
    MI_LOCALIZATION_LABEL {
        int ecg_id PK "EKG referansı"
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
        int ecg_id FK "İlişkili EKG"
        string model_version FK "Model versiyonu"
        datetime created_at "Oluşturulma zamanı"
        string input_format "Girdi formatı"
        int sample_rate_hz "Örnekleme hızı"
        float duration_sec "Süre (saniye)"
    }
    
    PREDICTION_RESULT {
        string result_id PK "Sonuç kimliği"
        string request_id FK "İstek referansı"
        float prob_MI "MI olasılığı"
        float prob_STTC "STTC olasılığı"
        float prob_CD "CD olasılığı"
        float prob_HYP "HYP olasılığı"
        float prob_NORM "Türetilmiş NORM"
        string predicted_labels "Tahmin edilen etiketler"
        string primary_label "Birincil etiket"
        float primary_confidence "Birincil güven"
        string triage_level "Triaj seviyesi"
        string agreement_type "Model uyum tipi"
    }
    
    MI_LOCALIZATION_RESULT {
        string localization_id PK "Lokalizasyon kimliği"
        string result_id FK "Sonuç referansı"
        float prob_AMI "AMI olasılığı"
        float prob_ASMI "ASMI olasılığı"
        float prob_ALMI "ALMI olasılığı"
        float prob_IMI "IMI olasılığı"
        float prob_LMI "LMI olasılığı"
        string detected_regions "Tespit edilen bölgeler"
        string label_space "Etiket uzayı kimliği"
        string mapping_fingerprint "Eşleme parmak izi"
    }
    
    XAI_ARTIFACT {
        string artifact_id PK "Yapıt kimliği"
        string result_id FK "Sonuç referansı"
        string artifact_type "Tip"
        string file_path "Dosya yolu"
        string target_class "Hedef sınıf"
        datetime created_at "Oluşturulma zamanı"
        int file_size_bytes "Dosya boyutu"
    }
    
    MODEL_CHECKPOINT {
        string checkpoint_id PK "Kontrol noktası kimliği"
        string model_type "Model tipi"
        string file_path "Dosya yolu"
        string model_hash "Model özeti"
        int output_dimension "Çıktı boyutu"
        datetime trained_at "Eğitim tarihi"
        float validation_auroc "Doğrulama AUROC"
    }
    
    THRESHOLD_CONFIG {
        string config_id PK "Yapılandırma kimliği"
        string config_hash "Yapılandırma özeti"
        float threshold_MI "MI eşiği"
        float threshold_STTC "STTC eşiği"
        float threshold_CD "CD eşiği"
        float threshold_HYP "HYP eşiği"
        datetime optimized_at "Optimizasyon tarihi"
    }
```

---

## 3. Varlık Detayları

### 3.1 PATIENT (Hasta)

| Alan | Tip | Açıklama | Kısıtlar |
|------|-----|----------|----------|
| patient_id | INT | Birincil anahtar | PK, NOT NULL, AUTO_INCREMENT |
| age | FLOAT | Yaş (yıl) | CHECK (age >= 0 AND age <= 120) |
| sex | VARCHAR(1) | Cinsiyet | CHECK (sex IN ('E', 'K')) |
| height | FLOAT | Boy (cm) | NULL değeri alabilir |
| weight | FLOAT | Kilo (kg) | NULL değeri alabilir |

**İstatistikler (PTB-XL Veritabanı):**

| Metrik | Değer |
|--------|-------|
| Toplam Hasta | 18,885 |
| Yaş Aralığı | 18-89 |
| Erkek Oranı | %52 |
| Kadın Oranı | %48 |

---

### 3.2 ECG_RECORD (EKG Kaydı)

| Alan | Tip | Açıklama | Kısıtlar |
|------|-----|----------|----------|
| ecg_id | INT | Birincil anahtar | PK, NOT NULL |
| patient_id | INT | Hasta referansı | FK, PATIENT tablosuna referans |
| filename_lr | VARCHAR(255) | 100Hz dosya yolu | NOT NULL |
| filename_hr | VARCHAR(255) | 500Hz dosya yolu | NOT NULL |
| strat_fold | INT | Çapraz doğrulama katmanı | CHECK (strat_fold BETWEEN 1 AND 10) |

**Veri Bölümlemesi:**

| Bölüm | Katmanlar | Kayıt Sayısı | Oran |
|-------|-----------|--------------|------|
| Eğitim | 1-8 | 17,469 | %80 |
| Doğrulama | 9 | 2,189 | %10 |
| Test | 10 | 2,179 | %10 |

---

### 3.3 SCP_STATEMENT (SCP İfadesi)

PTB-XL veri setindeki standart SCP kodları:

| Kategori | Kodlar | Açıklama |
|----------|--------|----------|
| NORM | NORM | Normal EKG |
| MI | AMI, IMI, ASMI, ALMI, LMI, ILMI, IPLMI, IPMI | Miyokard Enfarktüsü |
| STTC | NDT, NST_, ISCA, ISCI, ISC_, STD_, STE_ | ST/T Değişikliği |
| CD | CLBBB, CRBBB, IRBBB, 1AVB, 2AVB, 3AVB | İletim Bozukluğu |
| HYP | LVH, RVH, SEHYP, LAO/LAE, RAO/RAE | Hipertrofi |

---

### 3.4 SUPERCLASS_LABEL (Süpersınıf Etiketi)

**Çoklu Etiket Yapısı:**

```mermaid
graph LR
    subgraph Coklu_Etiket["Çoklu Etiket Yapısı"]
        MI["MI: 0/1"]
        STTC["STTC: 0/1"]
        CD["CD: 0/1"]
        HYP["HYP: 0/1"]
    end
    
    subgraph Turetilmis["Türetilmiş"]
        NORM["NORM = NOR(MI, STTC, CD, HYP)"]
    end
    
    MI --> NORM
    STTC --> NORM
    CD --> NORM
    HYP --> NORM
```

**Etiket Dağılımı:**

| Sınıf | Kayıt Sayısı | Oran |
|-------|--------------|------|
| MI | 5,486 | %25.1 |
| STTC | 5,250 | %24.0 |
| CD | 4,907 | %22.5 |
| HYP | 2,655 | %12.2 |
| NORM | 9,528 | %43.6 |

*Not: Toplamlar %100'ü aşar çünkü çoklu etiket yapısı kullanılmaktadır.*

---

### 3.5 MI_LOCALIZATION_LABEL (MI Lokalizasyon Etiketi)

**SCP Kodundan Bölge Eşlemesi:**

```mermaid
graph TB
    subgraph SCP_Kodlari["SCP Kodları"]
        AMI_CODE["AMI"]
        IMI_CODE["IMI"]
        ILMI_CODE["ILMI"]
        IPLMI_CODE["IPLMI"]
        INJXX["INJIN, INJAL, INJAS..."]
    end
    
    subgraph Turetilmis_Bolgeler["Türetilmiş Bölgeler"]
        AMI["AMI - Anterior"]
        ASMI["ASMI - Anteroseptal"]
        ALMI["ALMI - Anterolateral"]
        IMI["IMI - Inferior"]
        LMI["LMI - Lateral"]
    end
    
    AMI_CODE --> AMI
    IMI_CODE --> IMI
    ILMI_CODE --> IMI
    ILMI_CODE --> LMI
    IPLMI_CODE --> IMI
    IPLMI_CODE --> LMI
```

**Eşleme Kuralları:**

| Kaynak Kod | Hedef Bölgeler | Açıklama |
|------------|----------------|----------|
| AMI | AMI | Anterior miyokard enfarktüsü |
| ASMI | ASMI | Anteroseptal miyokard enfarktüsü |
| ALMI | ALMI | Anterolateral miyokard enfarktüsü |
| IMI | IMI | Inferior miyokard enfarktüsü |
| LMI | LMI | Lateral miyokard enfarktüsü |
| ILMI | IMI, LMI | Inferolateral - iki bölgeye eşlenir |
| IPLMI | IMI, LMI | Inferoposterolateral - iki bölgeye eşlenir |
| IPMI | IMI | Inferoposterior - inferior bölgeye eşlenir |

---

### 3.6 PREDICTION_RESULT (Tahmin Sonucu)

**Ensemble Kombinasyonu:**

```mermaid
graph TB
    subgraph Kaynaklar["Kaynak Olasılıklar"]
        CNN["CNN Olasılıkları"]
        XGB["XGBoost Olasılıkları"]
    end
    
    subgraph Ensemble
        ENS["P_final = alpha × P_cnn + (1-alpha) × P_xgb"]
    end
    
    subgraph Cikti["Çıktılar"]
        PROBS["Olasılıklar"]
        LABELS["Etiketler"]
        TRIAGE["Triaj"]
    end
    
    CNN --> ENS
    XGB --> ENS
    ENS --> PROBS --> LABELS --> TRIAGE
```

---

## 4. İlişki Detayları

### 4.1 Kardinalite Tablosu

| İlişki | Tip | Açıklama |
|--------|-----|----------|
| PATIENT — ECG_RECORD | 1:N | Bir hasta birden fazla EKG kaydına sahip olabilir |
| ECG_RECORD — SCP_CODE_ASSIGNMENT | 1:N | Bir EKG birden fazla SCP koduna sahip olabilir |
| ECG_RECORD — SUPERCLASS_LABEL | 1:1 | Her EKG için bir süpersınıf etiketi bulunur |
| ECG_RECORD — MI_LOCALIZATION_LABEL | 1:0..1 | MI tespit edilirse lokalizasyon etiketi bulunur |
| ECG_RECORD — PREDICTION_REQUEST | 1:N | Bir EKG birden fazla kez analiz edilebilir |
| PREDICTION_REQUEST — PREDICTION_RESULT | 1:1 | Her istek bir sonuç üretir |
| PREDICTION_RESULT — MI_LOCALIZATION_RESULT | 1:0..1 | MI tespit edilirse lokalizasyon sonucu bulunur |
| PREDICTION_RESULT — XAI_ARTIFACT | 1:N | Bir sonuç birden fazla XAI yapıtı üretebilir |

---

## 5. Veri Bütünlüğü Kuralları

### 5.1 Yabancı Anahtar Kısıtları

| Kaynak Tablo | Hedef Tablo | Silme Kuralı | Güncelleme Kuralı |
|--------------|-------------|--------------|-------------------|
| ECG_RECORD | PATIENT | RESTRICT | CASCADE |
| PREDICTION_RESULT | PREDICTION_REQUEST | CASCADE | CASCADE |
| XAI_ARTIFACT | PREDICTION_RESULT | CASCADE | CASCADE |

### 5.2 Kontrol Kısıtları

| Tablo | Kısıt | Açıklama |
|-------|-------|----------|
| PREDICTION_RESULT | prob_MI BETWEEN 0 AND 1 | Olasılık değer aralığı |
| PREDICTION_RESULT | triage_level IN ('HIGH', 'MEDIUM', 'LOW', 'REVIEW') | Geçerli triaj seviyeleri |
| THRESHOLD_CONFIG | threshold_MI BETWEEN 0 AND 1 | Eşik değer aralığı |

---

## 6. Veri Akışı

```mermaid
flowchart TB
    subgraph Girdi_Katmani["Girdi Katmanı"]
        PTBXL["PTB-XL Ham Veriler"]
    end
    
    subgraph Isleme_Katmani["İşleme Katmanı"]
        PATIENT_TBL["PATIENT"]
        ECG_TBL["ECG_RECORD"]
        SCP_TBL["SCP_STATEMENT"]
        LABEL_TBL["SUPERCLASS_LABEL ve MI_LOCALIZATION_LABEL"]
    end
    
    subgraph Cikarim_Katmani["Çıkarım Katmanı"]
        REQUEST["PREDICTION_REQUEST"]
        RESULT["PREDICTION_RESULT"]
        MILOC["MI_LOCALIZATION_RESULT"]
        XAI["XAI_ARTIFACT"]
    end
    
    subgraph Konfigurasyon["Konfigürasyon"]
        MODEL["MODEL_CHECKPOINT"]
        THRESH["THRESHOLD_CONFIG"]
    end
    
    PTBXL --> PATIENT_TBL
    PTBXL --> ECG_TBL
    PTBXL --> SCP_TBL
    ECG_TBL --> LABEL_TBL
    SCP_TBL --> LABEL_TBL
    
    ECG_TBL --> REQUEST
    MODEL --> REQUEST
    THRESH --> REQUEST
    REQUEST --> RESULT
    RESULT --> MILOC
    RESULT --> XAI
```

---

## 7. Örnek Veri

### 7.1 Örnek EKG Kaydı

| Alan | Değer |
|------|-------|
| ecg_id | 1 |
| patient_id | 15709 |
| filename_lr | records100/00000/00001_lr |
| filename_hr | records500/00000/00001_hr |
| strat_fold | 3 |
| scp_codes | AMI: 80.0, IMI: 100.0 |

### 7.2 Örnek Tahmin Sonucu

| Alan | Değer |
|------|-------|
| result_id | res_abc123 |
| request_id | req_xyz789 |
| prob_MI | 0.85 |
| prob_STTC | 0.12 |
| prob_CD | 0.08 |
| prob_HYP | 0.05 |
| prob_NORM | 0.15 |
| predicted_labels | [MI] |
| primary_label | MI |
| primary_confidence | 0.85 |
| triage_level | HIGH |
| agreement_type | AGREE_MI |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Veritabanı Mimarı | | | |
| Teknik Lider | | | |
| Kalite Güvence Mühendisi | | | |

---

**Doküman Sonu**

*Bu ERD, CardioGuard-AI v1.0.0 veri modelini temsil eder. Veritabanı şeması, dosya tabanlı depolama kullanıldığından kavramsal düzeydedir. Üretim ortamında PostgreSQL veya MongoDB kullanılması önerilir.*
