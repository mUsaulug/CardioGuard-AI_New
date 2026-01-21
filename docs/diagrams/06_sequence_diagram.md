# CardioGuard-AI
# Sıralı Akış Diyagramları

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sıralı Akış Diyagramları  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## 1. Ana Tahmin Akışı

Bu diyagram, EKG sinyalinin yüklenmesinden sonuç üretilmesine kadar olan tam akışı göstermektedir.

### 1.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant K as Klinisyen
    participant API
    participant CNN
    participant XGB as XGBoost
    participant Guard
    participant Loc as Lokalizasyon
    participant XAI
    
    K->>API: POST /predict (EKG)
    API->>API: Normalize Et
    
    par Paralel Tahmin
        API->>CNN: Superclass
        CNN-->>API: P_cnn
    and
        API->>CNN: Binary
        CNN-->>API: P_binary
    end
    
    API->>XGB: Gömme Tahmin
    XGB-->>API: P_xgb
    
    API->>API: Ensemble
    API->>Guard: Tutarlılık
    Guard-->>API: Triaj
    
    alt MI Tespit
        API->>Loc: Bölge Tespiti
        Loc-->>API: 5 Bölge
    end
    
    API->>XAI: Grad-CAM
    XAI-->>API: Isı Haritası
    
    API-->>K: AIResult v1.0
```

### 1.2 Akış Açıklaması

| Faz | Bileşenler | Açıklama |
|-----|------------|----------|
| 1. Girdi | API, Normalizer | Sinyal yükleme ve normalizasyon |
| 2. Tahmin | CNN, XGBoost | Paralel model çıkarımı |
| 3. Birleştirme | Ensemble | CNN ve XGB olasılıklarını birleştirme |
| 4. Kontrol | ConsistencyGuard | Tutarlılık ve triaj belirleme |
| 5. Lokalizasyon | Lokalizasyon CNN | MI bölge tespiti (koşullu) |
| 6. Açıklama | GradCAM | XAI üretimi |
| 7. Yanıt | AIResultMapper | JSON yanıt oluşturma |

---

## 2. Sistem Başlatma Akışı

### 2.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant Main
    participant Val as Doğrulayıcı
    participant Loader
    participant State
    
    Main->>Val: Checkpoint Doğrula
    
    loop Her Model
        Val->>Val: Boyut Kontrolü
        alt Uyuşmazlık
            Val-->>Main: Hata
            Main->>Main: sys.exit(1)
        end
    end
    
    Val->>Val: Fingerprint Kontrolü
    Val-->>Main: Doğrulama OK
    
    Main->>Loader: Modelleri Yükle
    Loader->>State: Binary CNN
    Loader->>State: Superclass CNN
    Loader->>State: Lokalizasyon CNN
    Loader->>State: XGBoost
    
    State-->>Main: Hazır
```

### 2.2 Fail-Fast Mekanizması

Sistem başlangıcında uygulanan doğrulama kontrolleri:

| Kontrol | Açıklama | Başarısızlık Durumu |
|---------|----------|---------------------|
| Dosya Varlığı | Checkpoint dosyaları mevcut mu? | FileNotFoundError |
| Boyut Uyumu | Model çıktı boyutu beklenenle eşleşiyor mu? | CheckpointMismatchError |
| Fingerprint | MI eşleme parmak izi değişmiş mi? | MappingDriftError |

---

## 3. Tutarlılık Kontrolü

### 3.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant API
    participant Guard
    
    API->>Guard: check(P_super, P_binary)
    
    Guard->>Guard: super_karar = P >= 0.01
    Guard->>Guard: binary_karar = P >= 0.5
    
    alt Her İkisi Pozitif
        Guard-->>API: AGREE_MI, YÜKSEK
    else Her İkisi Negatif
        Guard-->>API: AGREE_NO_MI, DÜŞÜK
    else Uyumsuz
        Guard-->>API: DISAGREE, İNCELEME
    end
```

### 3.2 Karar Matrisi

| Superclass MI | Binary MI | Sonuç | Triaj |
|---------------|-----------|-------|-------|
| ≥ 0.01 | ≥ 0.5 | AGREE_MI | Yüksek |
| < 0.01 | < 0.5 | AGREE_NO_MI | Düşük |
| ≥ 0.01 | < 0.5 | DISAGREE_TYPE_1 | İnceleme |
| < 0.01 | ≥ 0.5 | DISAGREE_TYPE_2 | İnceleme |

---

## 4. XGBoost Hibrit Pipeline

### 4.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant API
    participant Backbone
    participant Scaler
    participant XGB
    participant Calib
    
    API->>Backbone: Gömme Çıkar
    Backbone-->>API: 64-dim vektör
    
    API->>Scaler: Ölçekle
    Scaler-->>API: Normalize
    
    loop Her Sınıf
        API->>XGB: predict_proba
        XGB-->>API: P_raw
        API->>Calib: Kalibre Et
        Calib-->>API: P_calibrated
    end
    
    API->>API: Ensemble
```

### 4.2 Ensemble Formülü

```
P_final = α × P_cnn + (1 - α) × P_xgb
```

Varsayılan: α = 0.15 (XGBoost ağırlıklı)

---

## 5. Grad-CAM Üretimi

### 5.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant API
    participant GC as GradCAM
    participant Model
    
    API->>GC: Başlat(model, katman)
    GC->>Model: Hook Kaydet
    
    API->>GC: generate(sinyal, sınıf)
    GC->>Model: İleri Geçiş
    Model-->>GC: Aktivasyonlar
    
    GC->>Model: Geri Geçiş
    Model-->>GC: Gradyanlar
    
    GC->>GC: Ağırlıklı Toplam
    GC->>GC: ReLU + Normalize
    GC-->>API: Isı Haritası
```

### 5.2 Grad-CAM Hesaplama Adımları

| Adım | İşlem | Çıktı |
|------|-------|-------|
| 1 | İleri geçiş | Aktivasyonlar (A) |
| 2 | Geri geçiş | Gradyanlar (∂y/∂A) |
| 3 | Global ortalama | Ağırlıklar (α) |
| 4 | Ağırlıklı toplam | Σ(α × A) |
| 5 | ReLU | max(0, cam) |
| 6 | Normalize | [0, 1] aralığı |

---

## 6. Sağlık Kontrolü

### 6.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant LB as Yük Dengeleyici
    participant API
    participant State
    
    loop Her 30 Saniye
        LB->>API: GET /health
        API-->>LB: status: ok
    end
    
    loop Her 10 Saniye
        LB->>API: GET /ready
        API->>State: Model Durumu
        State-->>API: Yükleme Bilgisi
        API-->>LB: ready: true/false
    end
```

### 6.2 Health Check Endpoints

| Endpoint | Amaç | Yanıt |
|----------|------|-------|
| /health | Canlılık kontrolü | {"status": "ok"} |
| /ready | Hazırlık kontrolü | {"ready": true/false} |

---

## 7. Hata Senaryoları

### 7.1 Akış Diyagramı

```mermaid
sequenceDiagram
    autonumber
    participant K as İstemci
    participant API
    
    K->>API: Geçersiz Format
    API-->>K: 400 Format Hatası
    
    K->>API: Yanlış Boyut
    API-->>K: 400 Boyut Hatası
    
    K->>API: Model Yüklenmemiş
    API-->>K: 503 Servis Hazır Değil
```

### 7.2 Hata Kodları

| Kod | Açıklama | Çözüm |
|-----|----------|-------|
| 400 | Geçersiz girdi | Dosya formatını kontrol edin |
| 503 | Servis hazır değil | Modellerin yüklenmesini bekleyin |
| 500 | İç hata | Günlükleri inceleyin |

---

## 8. Planlanan Akışlar (v2.0)

### 8.1 RAG Entegrasyonu

```mermaid
sequenceDiagram
    participant API
    participant RAG
    participant LLM
    
    API->>RAG: Tahmin + Sorgu
    RAG-->>API: Klinik Kılavuzlar
    
    API->>LLM: Tahmin + Bağlam
    LLM-->>API: Klinik Rapor
```

### 8.2 Belirsizlik Tahmini

```mermaid
sequenceDiagram
    participant API
    participant Model
    
    loop N=30
        API->>Model: Dropout Aktif
        Model-->>API: Tahmin_i
    end
    
    API->>API: Ortalama + Varyans
```

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |

---

**Doküman Sonu**
