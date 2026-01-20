# CardioGuard-AI
# Sıralı Akış Diyagramları

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sıralı Akış Diyagramları (Sequence Diagrams)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## 1. Ana Tahmin Akışı

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
    
    K->>API: EKG Yükle
    API->>API: Normalize Et
    
    par Paralel Tahmin
        API->>CNN: Superclass Çıkarımı
        CNN-->>API: P_cnn (4 sınıf)
    and
        API->>CNN: Binary Çıkarımı
        CNN-->>API: P_binary
    end
    
    API->>XGB: Gömme + Tahmin
    XGB-->>API: P_xgb (4 sınıf)
    
    API->>API: Ensemble Birleştirme
    
    API->>Guard: Tutarlılık Kontrolü
    Guard-->>API: Triaj Seviyesi
    
    alt MI Tespit Edildi
        API->>Loc: Bölge Tespiti
        Loc-->>API: 5 Bölge Olasılığı
    end
    
    API->>XAI: Grad-CAM Üret
    XAI-->>API: Isı Haritası
    
    API-->>K: AIResult v1.0
```

---

## 2. Sistem Başlatma Akışı

```mermaid
sequenceDiagram
    autonumber
    participant Main
    participant Val as Doğrulayıcı
    participant Loader
    participant State
    
    Main->>Val: Checkpoint Doğrula
    
    loop Her Model İçin
        Val->>Val: Boyut Kontrolü
        alt Uyuşmazlık
            Val-->>Main: Hata - Çıkış
        end
    end
    
    Val->>Val: Fingerprint Kontrolü
    Val-->>Main: Doğrulama Tamam
    
    Main->>Loader: Modelleri Yükle
    Loader->>State: Binary CNN
    Loader->>State: Superclass CNN
    Loader->>State: Lokalizasyon CNN
    Loader->>State: XGBoost Modelleri
    
    State-->>Main: Hazır
```

---

## 3. Tutarlılık Kontrolü

```mermaid
sequenceDiagram
    autonumber
    participant API
    participant Guard
    
    API->>Guard: check(P_super, P_binary)
    
    Guard->>Guard: super_karar = P_super >= 0.01
    Guard->>Guard: binary_karar = P_binary >= 0.5
    
    alt Her İkisi Pozitif
        Guard-->>API: AGREE_MI, YÜKSEK
    else Her İkisi Negatif
        Guard-->>API: AGREE_NO_MI, DÜŞÜK
    else Uyumsuz
        Guard-->>API: DISAGREE, İNCELEME
    end
```

---

## 4. XGBoost Hibrit Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant API
    participant Backbone
    participant Scaler
    participant XGB
    participant Calib as Kalibratör
    
    API->>Backbone: Gömme Çıkar
    Backbone-->>API: 64-dim vektör
    
    API->>Scaler: Ölçekle
    Scaler-->>API: Normalize vektör
    
    loop MI, STTC, CD, HYP
        API->>XGB: predict_proba
        XGB-->>API: P_raw
        API->>Calib: Kalibre Et
        Calib-->>API: P_calibrated
    end
    
    API->>API: Ensemble = 0.15×CNN + 0.85×XGB
```

---

## 5. Grad-CAM Üretimi

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

---

## 6. Sağlık Kontrolü

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

---

## 7. Hata Senaryoları

```mermaid
sequenceDiagram
    autonumber
    participant K as İstemci
    participant API
    
    K->>API: Geçersiz Format
    API-->>K: 400 - Format Hatası
    
    K->>API: Yanlış Boyut
    API-->>K: 400 - Boyut Hatası
    
    K->>API: Model Yüklenmemiş
    API-->>K: 503 - Servis Hazır Değil
```

---

## 8. Özet Akış

```mermaid
graph TD
    A[EKG Yükleme] --> B[Normalizasyon]
    B --> C[CNN Tahminleri]
    B --> D[XGBoost Tahminleri]
    C --> E[Ensemble]
    D --> E
    E --> F[Tutarlılık Kontrolü]
    F --> G{MI?}
    G -->|Evet| H[Lokalizasyon]
    G -->|Hayır| I[XAI]
    H --> I
    I --> J[AIResult]
```

---

## 9. Planlanan Akışlar (v2.0)

### 9.1 RAG Entegrasyonu

```mermaid
sequenceDiagram
    participant API
    participant RAG
    participant LLM
    
    API->>RAG: Tahmin + Sorgu
    RAG->>RAG: Bağlam Ara
    RAG-->>API: İlgili Kılavuzlar
    
    API->>LLM: Tahmin + Bağlam
    LLM-->>API: Klinik Rapor
```

### 9.2 Belirsizlik Tahmini

```mermaid
sequenceDiagram
    participant API
    participant Model
    
    loop N Kez (N=30)
        API->>Model: Dropout Aktif Çıkarım
        Model-->>API: Tahmin_i
    end
    
    API->>API: Ortalama ve Varyans Hesapla
    API->>API: Güven Aralığı Belirle
```

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |

---

**Doküman Sonu**
