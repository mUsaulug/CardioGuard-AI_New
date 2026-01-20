# CardioGuard-AI: Use Case Diyagramı
## (Kullanım Senaryoları)

---

## 📋 Doküman Bilgileri

| Özellik | Değer |
|---------|-------|
| **Proje Adı** | CardioGuard-AI |
| **Doküman Tipi** | Use Case Diyagramı |
| **Versiyon** | 1.0.0 |
| **Tarih** | 2026-01-21 |

---

## 1. Aktörler

### 1.1 Birincil Aktörler

| Aktör | Açıklama | Rol |
|-------|----------|-----|
| 👨‍⚕️ **Klinisyen** | Hastane doktoru veya kardiyolog | EKG analizi talep eder, sonuçları yorumlar |
| 🧑‍💻 **ML Mühendisi** | Makine öğrenimi uzmanı | Model eğitimi ve optimizasyonu yapar |
| 🖥️ **Sistem Yöneticisi** | IT operasyon personeli | Sistem bakımı ve izleme yapar |

### 1.2 İkincil Aktörler

| Aktör | Açıklama | Rol |
|-------|----------|-----|
| ⏰ **Zamanlayıcı** | Otomatik görev planlayıcı | Periyodik görevleri tetikler |
| 📊 **Harici Sistemler** | Hastane bilgi sistemi (HIS) | Veri entegrasyonu sağlar |

---

## 2. Use Case Diyagramı

```mermaid
graph TB
    subgraph Aktörler
        Clinician["👨‍⚕️ Klinisyen"]
        MLEngineer["🧑‍💻 ML Mühendisi"]
        SysAdmin["🖥️ Sistem Yöneticisi"]
        Scheduler["⏰ Zamanlayıcı"]
    end
    
    subgraph CardioGuard-AI["🏥 CardioGuard-AI Sistemi"]
        subgraph Klinik["Klinik Use Case'ler"]
            UC1["UC-01: EKG Sinyali Yükleme"]
            UC2["UC-02: Patoloji Tespiti"]
            UC3["UC-03: MI Lokalizasyonu"]
            UC4["UC-04: Sonuç Raporlama"]
            UC5["UC-05: XAI Açıklamaları Görüntüleme"]
            UC9["UC-09: Triaj Belirleme"]
        end
        
        subgraph Geliştirme["Geliştirme Use Case'ler"]
            UC6["UC-06: Model Eğitimi"]
            UC10["UC-10: Model Değerlendirme"]
            UC11["UC-11: Hyperparametre Optimizasyonu"]
        end
        
        subgraph Operasyon["Operasyon Use Case'ler"]
            UC7["UC-07: Checkpoint Yönetimi"]
            UC8["UC-08: Health Check"]
            UC12["UC-12: Log İzleme"]
        end
    end
    
    %% Klinisyen bağlantıları
    Clinician --> UC1
    Clinician --> UC4
    Clinician --> UC5
    
    %% ML Mühendisi bağlantıları
    MLEngineer --> UC6
    MLEngineer --> UC7
    MLEngineer --> UC10
    MLEngineer --> UC11
    
    %% Sistem Yöneticisi bağlantıları
    SysAdmin --> UC7
    SysAdmin --> UC8
    SysAdmin --> UC12
    
    %% Zamanlayıcı bağlantıları
    Scheduler --> UC8
    
    %% Include ilişkileri
    UC1 -.->|includes| UC2
    UC2 -.->|includes| UC9
    UC2 -.->|extends| UC3
    UC2 -.->|includes| UC4
    UC4 -.->|extends| UC5
    
    %% Styling
    style UC1 fill:#e3f2fd,stroke:#1976d2
    style UC2 fill:#fff3e0,stroke:#f57c00
    style UC3 fill:#ffebee,stroke:#c62828
    style UC4 fill:#e8f5e9,stroke:#388e3c
    style UC5 fill:#f3e5f5,stroke:#7b1fa2
    style UC6 fill:#fce4ec,stroke:#c2185b
    style UC7 fill:#e0f2f1,stroke:#00796b
    style UC8 fill:#f5f5f5,stroke:#616161
    style UC9 fill:#fff8e1,stroke:#ffa000
    style UC10 fill:#e8eaf6,stroke:#3f51b5
    style UC11 fill:#fbe9e7,stroke:#e64a19
    style UC12 fill:#eceff1,stroke:#455a64
```

---

## 3. Use Case Detayları

### UC-01: EKG Sinyali Yükleme

```mermaid
sequenceDiagram
    actor Clinician as 👨‍⚕️ Klinisyen
    participant API as FastAPI
    participant Parser as ECG Parser
    participant Validator as Sinyal Doğrulayıcı
    
    Clinician->>API: POST /predict/superclass (file)
    API->>Parser: parse_ecg_file()
    Parser->>Parser: Format kontrolü (.npz/.npy)
    Parser->>Validator: Boyut doğrulama (12×1000)
    alt Geçerli Sinyal
        Validator-->>API: ✅ Sinyal hazır
    else Geçersiz Format
        Validator-->>API: ❌ HTTP 400 Error
    end
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-01 |
| **İsim** | EKG Sinyali Yükleme |
| **Aktör** | Klinisyen |
| **Ön Koşul** | Kullanıcı sisteme bağlı |
| **Son Koşul** | Sinyal işlenmeye hazır |
| **Temel Akış** | 1. Kullanıcı EKG dosyasını seçer<br/>2. Sistem formatı doğrular<br/>3. Sistem boyutu kontrol eder<br/>4. Sinyal normalize edilir |
| **Alternatif Akış** | - Geçersiz format: Hata mesajı göster<br/>- Yanlış boyut: Hata mesajı göster |

---

### UC-02: Patoloji Tespiti

```mermaid
graph TB
    subgraph Girdi
        ECG["12-Lead EKG<br/>(12×1000)"]
    end
    
    subgraph İşlem
        SUPER["Superclass CNN<br/>(4-sınıf)"]
        BINARY["Binary MI CNN"]
        XGB["XGBoost Ensemble"]
        GUARD["Consistency Guard"]
    end
    
    subgraph Çıktı
        PROBS["Olasılıklar<br/>MI: 0.85<br/>STTC: 0.12<br/>CD: 0.08<br/>HYP: 0.05"]
        LABELS["Etiketler<br/>[MI]"]
    end
    
    ECG --> SUPER & BINARY
    SUPER --> XGB
    SUPER --> GUARD
    BINARY --> GUARD
    XGB --> PROBS
    GUARD --> LABELS
    
    style GUARD fill:#ffebee,stroke:#c62828
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-02 |
| **İsim** | Patoloji Tespiti |
| **Aktör** | Sistem (UC-01 tarafından tetiklenir) |
| **Ön Koşul** | EKG sinyali yüklenmiş ve normalize edilmiş |
| **Son Koşul** | Patoloji olasılıkları ve etiketler üretilmiş |
| **Temel Akış** | 1. CNN modeli ileri geçiş<br/>2. XGBoost tahmin<br/>3. Ensemble kombinasyonu<br/>4. Tutarlılık kontrolü<br/>5. Etiket belirleme |

---

### UC-03: MI Lokalizasyonu

```mermaid
graph LR
    subgraph Gate["Lokalizasyon Kapısı"]
        CHECK{MI Tespit<br/>Edildi mi?}
    end
    
    subgraph Model
        LOC["Localization CNN<br/>(5-bölge)"]
    end
    
    subgraph Çıktı
        REGIONS["Anatomik Bölgeler<br/>AMI: ✅ 0.85<br/>ASMI: ❌ 0.12<br/>ALMI: ❌ 0.08<br/>IMI: ✅ 0.72<br/>LMI: ❌ 0.15"]
    end
    
    CHECK -->|Evet| LOC
    CHECK -->|Hayır| SKIP["Atla"]
    LOC --> REGIONS
    
    style CHECK fill:#fff3e0
    style LOC fill:#e3f2fd
    style REGIONS fill:#e8f5e9
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-03 |
| **İsim** | MI Lokalizasyonu |
| **Aktör** | Sistem (UC-02 sonrası koşullu) |
| **Ön Koşul** | MI tespit edilmiş (AGREE_MI veya DISAGREE_TYPE_1) |
| **Son Koşul** | Anatomik MI bölgeleri belirlenmiş |
| **Temel Akış** | 1. MI gate kontrolü<br/>2. Lokalizasyon CNN çalıştır<br/>3. 5 bölge için olasılık üret<br/>4. Eşik üzeri bölgeleri işaretle |
| **Bölgeler** | AMI (Anterior), ASMI (Anteroseptal), ALMI (Anterolateral), IMI (Inferior), LMI (Lateral) |

---

### UC-04: Sonuç Raporlama

```mermaid
graph TB
    subgraph Girdiler
        PREDS["Tahminler"]
        TRIAGE["Triaj Seviyesi"]
        XAI_OUT["XAI Artifacts"]
    end
    
    subgraph Mapper["AIResult Mapper v1.0"]
        MAP["JSON Dönüşümü"]
    end
    
    subgraph Çıktı
        JSON["```json
{
  identity: {...},
  predictions: {...},
  localization: {...},
  triage: {...},
  explanations: {...}
}
```"]
    end
    
    PREDS & TRIAGE & XAI_OUT --> MAP --> JSON
    
    style MAP fill:#f3e5f5
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-04 |
| **İsim** | Sonuç Raporlama |
| **Aktör** | Klinisyen |
| **Ön Koşul** | Tüm tahminler tamamlanmış |
| **Son Koşul** | AIResult v1.0 formatında rapor üretilmiş |
| **Çıktı Alanları** | identity, mode, input, predictions, localization, triage, sources, explanations, versions |

---

### UC-05: XAI Açıklamaları Görüntüleme

```mermaid
graph LR
    subgraph Açıklama Tipleri
        GCAM["Grad-CAM<br/>Temporal Heatmap"]
        SHAP["SHAP<br/>Feature Importance"]
    end
    
    subgraph Görselleştirme
        PLOT["Matplotlib Plot"]
        OVERLAY["Sinyal Üzeri<br/>Overlay"]
    end
    
    subgraph Çıktı
        PNG["PNG Dosyası"]
        NARRATIVE["Text Narrative"]
    end
    
    GCAM --> OVERLAY --> PNG
    SHAP --> PLOT --> PNG
    PNG --> NARRATIVE
    
    style GCAM fill:#ffecb3
    style SHAP fill:#ffecb3
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-05 |
| **İsim** | XAI Açıklamaları Görüntüleme |
| **Aktör** | Klinisyen |
| **Ön Koşul** | Tahmin tamamlanmış, XAI aktif |
| **Son Koşul** | Görsel açıklamalar üretilmiş |
| **Artifact Tipleri** | gradcam_MI.png, shap_summary.png, narrative.md |

---

### UC-06: Model Eğitimi

```mermaid
graph TB
    subgraph Eğitim Görevleri
        T1["Binary MI CNN<br/>train_binary_cnn.py"]
        T2["Superclass CNN<br/>train_superclass_cnn.py"]
        T3["MI Localization<br/>train_mi_localization.py"]
        T4["XGBoost<br/>train_superclass_xgb_ovr.py"]
    end
    
    subgraph Çıktılar
        C1["ecgcnn.pt"]
        C2["ecgcnn_superclass.pt"]
        C3["ecgcnn_localization.pt"]
        C4["xgb_models/*.json"]
    end
    
    T1 --> C1
    T2 --> C2
    T3 --> C3
    T4 --> C4
    
    style T1 fill:#e3f2fd
    style T2 fill:#e3f2fd
    style T3 fill:#e3f2fd
    style T4 fill:#fff3e0
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-06 |
| **İsim** | Model Eğitimi |
| **Aktör** | ML Mühendisi |
| **Ön Koşul** | PTB-XL veri seti mevcut |
| **Son Koşul** | Checkpoint dosyaları üretilmiş |
| **Eğitim Parametreleri** | epochs=50, lr=1e-3, batch_size=64 |

---

### UC-07: Checkpoint Yönetimi

| Özellik | Değer |
|---------|-------|
| **ID** | UC-07 |
| **İsim** | Checkpoint Yönetimi |
| **Aktörler** | ML Mühendisi, Sistem Yöneticisi |
| **Ön Koşul** | Checkpoint dosyaları mevcut |
| **Son Koşul** | Checkpoint'ler doğrulanmış ve yüklenmiş |
| **Alt Use Case'ler** | - Checkpoint doğrulama<br/>- Checkpoint güncelleme<br/>- Rollback |

---

### UC-08: Health Check

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant State as AppState
    
    Client->>API: GET /health
    API-->>Client: {"status": "ok", "timestamp": "..."}
    
    Client->>API: GET /ready
    API->>State: models_loaded?
    State-->>API: {superclass: true, binary: true, ...}
    API-->>Client: {"ready": true, "models_loaded": {...}}
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-08 |
| **İsim** | Health Check |
| **Aktörler** | Sistem Yöneticisi, Zamanlayıcı |
| **Endpoint'ler** | `/health`, `/ready` |
| **Amaç** | Sistem ve model durumunu izleme |

---

### UC-09: Triaj Belirleme

```mermaid
graph TD
    subgraph Karar Ağacı
        START{Tahmin<br/>Sonucu}
        MI_CHECK{MI Tespit?}
        OTHER_CHECK{Diğer<br/>Patoloji?}
        AGREE_CHECK{Model<br/>Uyumu?}
    end
    
    subgraph Triaj Seviyeleri
        HIGH["🔴 HIGH<br/>MI Tespit"]
        MEDIUM["🟡 MEDIUM<br/>Diğer Patoloji"]
        LOW["🟢 LOW<br/>NORM"]
        REVIEW["🟠 REVIEW<br/>Model Uyumsuzluğu"]
    end
    
    START --> MI_CHECK
    MI_CHECK -->|Evet| AGREE_CHECK
    MI_CHECK -->|Hayır| OTHER_CHECK
    
    AGREE_CHECK -->|AGREE_MI| HIGH
    AGREE_CHECK -->|DISAGREE| REVIEW
    
    OTHER_CHECK -->|Evet| MEDIUM
    OTHER_CHECK -->|Hayır| LOW
    
    style HIGH fill:#ffcdd2
    style MEDIUM fill:#fff9c4
    style LOW fill:#c8e6c9
    style REVIEW fill:#ffe0b2
```

| Özellik | Değer |
|---------|-------|
| **ID** | UC-09 |
| **İsim** | Triaj Belirleme |
| **Aktör** | Sistem (otomatik) |
| **Seviyeleri** | HIGH, MEDIUM, LOW, REVIEW |
| **Kural** | MI → HIGH, Diğer Patoloji → MEDIUM, NORM → LOW, Uyumsuzluk → REVIEW |

---

## 4. Use Case İlişkileri

```mermaid
graph TB
    UC1[UC-01: EKG Yükleme] -->|<<includes>>| UC2[UC-02: Patoloji Tespiti]
    UC2 -->|<<includes>>| UC9[UC-09: Triaj]
    UC2 -->|<<extends>>| UC3[UC-03: MI Lokalizasyonu]
    UC2 -->|<<includes>>| UC4[UC-04: Raporlama]
    UC4 -->|<<extends>>| UC5[UC-05: XAI]
    
    UC6[UC-06: Eğitim] -->|<<includes>>| UC7[UC-07: Checkpoint]
    UC7 -->|<<includes>>| UC8[UC-08: Health Check]
    
    style UC1 fill:#e3f2fd
    style UC2 fill:#fff3e0
    style UC3 fill:#ffebee
    style UC4 fill:#e8f5e9
    style UC5 fill:#f3e5f5
    style UC6 fill:#fce4ec
    style UC7 fill:#e0f2f1
    style UC8 fill:#f5f5f5
    style UC9 fill:#fffde7
```

---

## 5. Öncelik Matrisi

| Use Case | Öncelik | Zorunluluk | Versiyon |
|----------|---------|------------|----------|
| UC-01 | Yüksek | Zorunlu | v1.0 |
| UC-02 | Yüksek | Zorunlu | v1.0 |
| UC-03 | Yüksek | Zorunlu | v1.0 |
| UC-04 | Yüksek | Zorunlu | v1.0 |
| UC-05 | Orta | Önerilen | v1.0 |
| UC-06 | Yüksek | Zorunlu | v1.0 |
| UC-07 | Yüksek | Zorunlu | v1.0 |
| UC-08 | Orta | Önerilen | v1.0 |
| UC-09 | Yüksek | Zorunlu | v1.0 |
| UC-10 | Orta | Önerilen | v1.1 |
| UC-11 | Düşük | Opsiyonel | v2.0 |
| UC-12 | Düşük | Opsiyonel | v1.1 |

---

> **Not:** Gelecek versiyonlarda RAG entegrasyonu (UC-13), Monte Carlo Dropout ile belirsizlik tahmini (UC-14) ve Canlı EKG Streaming (UC-15) use case'leri eklenecektir.
