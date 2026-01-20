# CardioGuard-AI
# Kullanım Senaryoları Diyagramı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Kullanım Senaryoları (Use Case) Dokümanı  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## İçindekiler

1. [Aktörler](#1-aktörler)
2. [Kullanım Senaryoları Diyagramı](#2-kullanım-senaryoları-diyagramı)
3. [Kullanım Senaryosu Detayları](#3-kullanım-senaryosu-detayları)
4. [Kullanım Senaryosu İlişkileri](#4-kullanım-senaryosu-ilişkileri)
5. [Öncelik Matrisi](#5-öncelik-matrisi)

---

## 1. Aktörler

### 1.1 Birincil Aktörler

| Aktör | Açıklama | Rol |
|-------|----------|-----|
| Klinisyen | Hastane doktoru veya kardiyolog | EKG analizi talep eder, sonuçları yorumlar |
| Makine Öğrenimi Mühendisi | Yapay zeka uzmanı | Model eğitimi ve optimizasyonu yapar |
| Sistem Yöneticisi | Bilgi teknolojileri operasyon personeli | Sistem bakımı ve izleme yapar |

### 1.2 İkincil Aktörler

| Aktör | Açıklama | Rol |
|-------|----------|-----|
| Zamanlayıcı Servisi | Otomatik görev planlayıcı | Periyodik görevleri tetikler |
| Harici Sistemler | Hastane bilgi sistemi (HIS) | Veri entegrasyonu sağlar |

---

## 2. Kullanım Senaryoları Diyagramı

```mermaid
graph TB
    subgraph Aktörler
        Clinician["Klinisyen"]
        MLEngineer["ML Mühendisi"]
        SysAdmin["Sistem Yöneticisi"]
        Scheduler["Zamanlayıcı"]
    end
    
    subgraph CardioGuard_AI["CardioGuard-AI Sistemi"]
        subgraph Klinik["Klinik Kullanım Senaryoları"]
            UC1["US-01: EKG Sinyali Yükleme"]
            UC2["US-02: Patoloji Tespiti"]
            UC3["US-03: MI Lokalizasyonu"]
            UC4["US-04: Sonuç Raporlama"]
            UC5["US-05: XAI Açıklamaları Görüntüleme"]
            UC9["US-09: Triaj Belirleme"]
        end
        
        subgraph Gelistirme["Geliştirme Kullanım Senaryoları"]
            UC6["US-06: Model Eğitimi"]
            UC10["US-10: Model Değerlendirme"]
            UC11["US-11: Hiperparametre Optimizasyonu"]
        end
        
        subgraph Operasyon["Operasyon Kullanım Senaryoları"]
            UC7["US-07: Kontrol Noktası Yönetimi"]
            UC8["US-08: Sağlık Kontrolü"]
            UC12["US-12: Günlük İzleme"]
        end
    end
    
    Clinician --> UC1
    Clinician --> UC4
    Clinician --> UC5
    
    MLEngineer --> UC6
    MLEngineer --> UC7
    MLEngineer --> UC10
    MLEngineer --> UC11
    
    SysAdmin --> UC7
    SysAdmin --> UC8
    SysAdmin --> UC12
    
    Scheduler --> UC8
    
    UC1 -.->|içerir| UC2
    UC2 -.->|içerir| UC9
    UC2 -.->|genişletir| UC3
    UC2 -.->|içerir| UC4
    UC4 -.->|genişletir| UC5
```

---

## 3. Kullanım Senaryosu Detayları

### 3.1 US-01: EKG Sinyali Yükleme

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-01 |
| Ad | EKG Sinyali Yükleme |
| Aktör | Klinisyen |
| Ön Koşul | Kullanıcı sisteme bağlı olmalıdır |
| Son Koşul | Sinyal işlenmeye hazır duruma getirilir |
| Temel Akış | 1. Kullanıcı EKG dosyasını seçer. 2. Sistem dosya formatını doğrular. 3. Sistem sinyal boyutunu kontrol eder. 4. Sinyal normalize edilir. |
| Alternatif Akış | Geçersiz format durumunda hata mesajı gösterilir. Yanlış boyut durumunda hata mesajı gösterilir. |

**Akış Diyagramı:**

```mermaid
sequenceDiagram
    participant Klinisyen
    participant API as API Servisi
    participant Parser as EKG Ayrıştırıcı
    participant Dogrulayici as Sinyal Doğrulayıcı
    
    Klinisyen->>API: POST /predict/superclass (dosya)
    API->>Parser: parse_ecg_file()
    Parser->>Parser: Format kontrolü (.npz/.npy)
    Parser->>Dogrulayici: Boyut doğrulama (12×1000)
    alt Geçerli Sinyal
        Dogrulayici-->>API: Sinyal hazır
    else Geçersiz Format
        Dogrulayici-->>API: HTTP 400 Hata
    end
```

---

### 3.2 US-02: Patoloji Tespiti

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-02 |
| Ad | Patoloji Tespiti |
| Aktör | Sistem (US-01 tarafından tetiklenir) |
| Ön Koşul | EKG sinyali yüklenmiş ve normalize edilmiş olmalıdır |
| Son Koşul | Patoloji olasılıkları ve etiketler üretilir |
| Temel Akış | 1. CNN modeli ileri geçiş yapar. 2. XGBoost tahmin üretir. 3. Ensemble kombinasyonu uygulanır. 4. Tutarlılık kontrolü yapılır. 5. Etiketler belirlenir. |

**İşlem Akışı:**

```mermaid
graph TB
    subgraph Girdi
        ECG["12 Derivasyonlu EKG (12×1000)"]
    end
    
    subgraph Islem["İşlem"]
        SUPER["Superclass CNN (4 sınıf)"]
        BINARY["Binary MI CNN"]
        XGB["XGBoost Ensemble"]
        GUARD["Tutarlılık Kontrolü"]
    end
    
    subgraph Cikti["Çıktı"]
        PROBS["Olasılıklar: MI, STTC, CD, HYP"]
        LABELS["Etiketler"]
    end
    
    ECG --> SUPER
    ECG --> BINARY
    SUPER --> XGB
    SUPER --> GUARD
    BINARY --> GUARD
    XGB --> PROBS
    GUARD --> LABELS
```

---

### 3.3 US-03: MI Lokalizasyonu

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-03 |
| Ad | MI Lokalizasyonu |
| Aktör | Sistem (US-02 sonrası koşullu olarak çalışır) |
| Ön Koşul | MI tespit edilmiş olmalıdır (AGREE_MI veya DISAGREE_TYPE_1) |
| Son Koşul | Anatomik MI bölgeleri belirlenir |
| Temel Akış | 1. MI kapısı kontrol edilir. 2. Lokalizasyon CNN çalıştırılır. 3. Beş bölge için olasılık üretilir. 4. Eşik üzeri bölgeler işaretlenir. |
| Bölgeler | AMI (Anterior), ASMI (Anteroseptal), ALMI (Anterolateral), IMI (Inferior), LMI (Lateral) |

**Karar Akışı:**

```mermaid
graph LR
    subgraph Kapi["Lokalizasyon Kapısı"]
        CHECK{"MI Tespit Edildi mi?"}
    end
    
    subgraph Model
        LOC["Lokalizasyon CNN (5 bölge)"]
    end
    
    subgraph Cikti["Çıktı"]
        REGIONS["Anatomik Bölgeler ve Olasılıkları"]
    end
    
    CHECK -->|Evet| LOC
    CHECK -->|Hayır| SKIP["Atla"]
    LOC --> REGIONS
```

---

### 3.4 US-04: Sonuç Raporlama

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-04 |
| Ad | Sonuç Raporlama |
| Aktör | Klinisyen |
| Ön Koşul | Tüm tahminler tamamlanmış olmalıdır |
| Son Koşul | AIResult v1.0 formatında rapor üretilir |
| Çıktı Alanları | Kimlik bilgileri, mod, girdi, tahminler, lokalizasyon, triaj, kaynaklar, açıklamalar, versiyonlar |

---

### 3.5 US-05: XAI Açıklamaları Görüntüleme

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-05 |
| Ad | Açıklanabilir Yapay Zeka Görüntüleme |
| Aktör | Klinisyen |
| Ön Koşul | Tahmin tamamlanmış ve XAI etkinleştirilmiş olmalıdır |
| Son Koşul | Görsel açıklamalar üretilir |
| Çıktı Türleri | Grad-CAM ısı haritası (PNG), SHAP özet grafiği (PNG), Metin açıklaması (MD) |

---

### 3.6 US-06: Model Eğitimi

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-06 |
| Ad | Model Eğitimi |
| Aktör | Makine Öğrenimi Mühendisi |
| Ön Koşul | PTB-XL veri seti mevcut olmalıdır |
| Son Koşul | Kontrol noktası dosyaları üretilir |
| Eğitim Parametreleri | Epok sayısı: 50, öğrenme oranı: 0.001, yığın boyutu: 64 |

**Eğitim Akışı:**

```mermaid
graph TB
    subgraph Egitim_Gorevileri["Eğitim Görevleri"]
        T1["Binary MI CNN"]
        T2["Superclass CNN"]
        T3["MI Lokalizasyon"]
        T4["XGBoost"]
    end
    
    subgraph Ciktilar["Çıktılar"]
        C1["ecgcnn.pt"]
        C2["ecgcnn_superclass.pt"]
        C3["ecgcnn_localization.pt"]
        C4["xgb_models/*.json"]
    end
    
    T1 --> C1
    T2 --> C2
    T3 --> C3
    T4 --> C4
```

---

### 3.7 US-07: Kontrol Noktası Yönetimi

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-07 |
| Ad | Kontrol Noktası Yönetimi |
| Aktörler | Makine Öğrenimi Mühendisi, Sistem Yöneticisi |
| Ön Koşul | Kontrol noktası dosyaları mevcut olmalıdır |
| Son Koşul | Kontrol noktaları doğrulanmış ve yüklenmiş olmalıdır |
| Alt Senaryolar | Kontrol noktası doğrulama, kontrol noktası güncelleme, geri alma |

---

### 3.8 US-08: Sağlık Kontrolü

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-08 |
| Ad | Sağlık Kontrolü |
| Aktörler | Sistem Yöneticisi, Zamanlayıcı Servisi |
| Uç Noktalar | /health, /ready |
| Amaç | Sistem ve model durumunu izleme |

**Kontrol Akışı:**

```mermaid
sequenceDiagram
    participant Istemci as İstemci
    participant API as API Servisi
    participant Durum as Uygulama Durumu
    
    Istemci->>API: GET /health
    API-->>Istemci: {"status": "ok", "timestamp": "..."}
    
    Istemci->>API: GET /ready
    API->>Durum: Model yükleme durumu kontrolü
    Durum-->>API: {superclass: true, binary: true, ...}
    API-->>Istemci: {"ready": true, "models_loaded": {...}}
```

---

### 3.9 US-09: Triaj Belirleme

| Özellik | Değer |
|---------|-------|
| Tanımlayıcı | US-09 |
| Ad | Triaj Belirleme |
| Aktör | Sistem (otomatik) |
| Seviyeler | Yüksek (HIGH), Orta (MEDIUM), Düşük (LOW), İnceleme (REVIEW) |
| Kural | MI tespiti durumunda Yüksek, diğer patolojilerde Orta, normal durumda Düşük, model uyumsuzluğunda İnceleme |

**Karar Ağacı:**

```mermaid
graph TD
    subgraph Karar_Agaci["Karar Ağacı"]
        START{"Tahmin Sonucu"}
        MI_CHECK{"MI Tespit Edildi mi?"}
        OTHER_CHECK{"Diğer Patoloji Var mı?"}
        AGREE_CHECK{"Model Uyumu Var mı?"}
    end
    
    subgraph Triaj_Seviyeleri["Triaj Seviyeleri"]
        HIGH["YÜKSEK: MI Tespit Edildi"]
        MEDIUM["ORTA: Diğer Patoloji"]
        LOW["DÜŞÜK: Normal"]
        REVIEW["İNCELEME: Model Uyumsuzluğu"]
    end
    
    START --> MI_CHECK
    MI_CHECK -->|Evet| AGREE_CHECK
    MI_CHECK -->|Hayır| OTHER_CHECK
    
    AGREE_CHECK -->|AGREE_MI| HIGH
    AGREE_CHECK -->|DISAGREE| REVIEW
    
    OTHER_CHECK -->|Evet| MEDIUM
    OTHER_CHECK -->|Hayır| LOW
```

---

## 4. Kullanım Senaryosu İlişkileri

### 4.1 İlişki Diyagramı

```mermaid
graph TB
    UC1["US-01: EKG Yükleme"] -->|içerir| UC2["US-02: Patoloji Tespiti"]
    UC2 -->|içerir| UC9["US-09: Triaj"]
    UC2 -->|genişletir| UC3["US-03: MI Lokalizasyonu"]
    UC2 -->|içerir| UC4["US-04: Raporlama"]
    UC4 -->|genişletir| UC5["US-05: XAI"]
    
    UC6["US-06: Eğitim"] -->|içerir| UC7["US-07: Kontrol Noktası"]
    UC7 -->|içerir| UC8["US-08: Sağlık Kontrolü"]
```

### 4.2 İlişki Türleri

| İlişki | Kaynak | Hedef | Açıklama |
|--------|--------|-------|----------|
| İçerir (include) | US-01 | US-02 | EKG yükleme, patoloji tespitini tetikler |
| İçerir (include) | US-02 | US-09 | Patoloji tespiti, triaj belirlemeyi tetikler |
| Genişletir (extend) | US-02 | US-03 | MI tespit edilirse lokalizasyon çalışır |
| İçerir (include) | US-02 | US-04 | Patoloji tespiti sonrası raporlama yapılır |
| Genişletir (extend) | US-04 | US-05 | XAI etkinse açıklamalar gösterilir |

---

## 5. Öncelik Matrisi

| Kullanım Senaryosu | Öncelik | Zorunluluk | Hedef Versiyon |
|-------------------|---------|------------|----------------|
| US-01: EKG Yükleme | Yüksek | Zorunlu | v1.0 |
| US-02: Patoloji Tespiti | Yüksek | Zorunlu | v1.0 |
| US-03: MI Lokalizasyonu | Yüksek | Zorunlu | v1.0 |
| US-04: Sonuç Raporlama | Yüksek | Zorunlu | v1.0 |
| US-05: XAI Görüntüleme | Orta | Önerilen | v1.0 |
| US-06: Model Eğitimi | Yüksek | Zorunlu | v1.0 |
| US-07: Kontrol Noktası Yönetimi | Yüksek | Zorunlu | v1.0 |
| US-08: Sağlık Kontrolü | Orta | Önerilen | v1.0 |
| US-09: Triaj Belirleme | Yüksek | Zorunlu | v1.0 |
| US-10: Model Değerlendirme | Orta | Önerilen | v1.1 |
| US-11: Hiperparametre Optimizasyonu | Düşük | Opsiyonel | v2.0 |
| US-12: Günlük İzleme | Düşük | Opsiyonel | v1.1 |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Proje Yöneticisi | | | |
| Teknik Lider | | | |
| Kalite Güvence Mühendisi | | | |

---

**Doküman Sonu**

*Gelecek versiyonlarda RAG entegrasyonu (US-13), Monte Carlo Dropout ile belirsizlik tahmini (US-14) ve Canlı EKG Akışı (US-15) kullanım senaryoları eklenecektir.*
