# CardioGuard-AI
# Sıralı Akış Diyagramları

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Sıralı Akış Diyagramları (Sequence Diagrams)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## İçindekiler

1. [Ana Tahmin Akışı](#1-ana-tahmin-akışı)
2. [Sistem Başlatma Akışı](#2-sistem-başlatma-akışı)
3. [Tutarlılık Kontrolü](#3-tutarlılık-kontrolü)
4. [XGBoost Hibrit Pipeline](#4-xgboost-hibrit-pipeline)
5. [Grad-CAM Üretimi](#5-grad-cam-üretimi)
6. [Sağlık Kontrolü](#6-sağlık-kontrolü)
7. [Hata Senaryoları](#7-hata-senaryoları)
8. [Planlanan Akışlar](#8-planlanan-akışlar)

---

## 1. Ana Tahmin Akışı

```mermaid
sequenceDiagram
    autonumber
    participant Klinisyen
    participant API as API Servisi
    participant Norm as Normalleştirici
    participant SuperCNN as Superclass CNN
    participant BinaryCNN as Binary CNN
    participant XGB as XGBoost
    participant Guard as Tutarlılık Denetimi
    participant LocCNN as Lokalizasyon CNN
    participant XAI as Grad-CAM
    participant Mapper as AIResult Eşleyici
    
    Klinisyen->>+API: POST /predict/superclass (EKG dosyası)
    
    rect rgb(240, 248, 255)
        Note over API, Norm: Faz 1: Girdi İşleme
        API->>+Norm: normalize(sinyal)
        Norm->>Norm: MinMax normalizasyonu (0-1)
        Norm-->>-API: X_norm (12×1000)
    end
    
    rect rgb(255, 248, 240)
        Note over API, XGB: Faz 2: Model Tahminleri
        par Paralel CNN Çıkarımı
            API->>+SuperCNN: forward(X_tensor)
            SuperCNN-->>-API: P_cnn: MI, STTC, CD, HYP
        and
            API->>+BinaryCNN: forward(X_tensor)
            BinaryCNN-->>-API: P_binary_MI
        end
        
        API->>+XGB: predict_proba(gömme)
        XGB-->>-API: P_xgb: MI, STTC, CD, HYP
        
        API->>API: Ensemble: P = 0.15×P_cnn + 0.85×P_xgb
    end
    
    rect rgb(255, 240, 245)
        Note over API, Guard: Faz 3: Tutarlılık Kontrolü
        API->>+Guard: check_consistency(P_super_MI, P_binary_MI)
        Guard->>Guard: Karar karşılaştırması
        Guard-->>-API: ConsistencyResult, Triaj
    end
    
    rect rgb(240, 255, 240)
        Note over API, LocCNN: Faz 4: MI Lokalizasyonu (Koşullu)
        alt MI Tespit Edildi
            API->>+LocCNN: forward(X_tensor)
            LocCNN-->>-API: P_loc: AMI, ASMI, ALMI, IMI, LMI
        end
    end
    
    rect rgb(255, 255, 240)
        Note over API, XAI: Faz 5: XAI Üretimi
        API->>+XAI: generate(X_tensor, class_index)
        XAI->>XAI: İleri + Geri geçiş
        XAI-->>-API: Isı haritası
    end
    
    rect rgb(245, 240, 255)
        Note over API, Mapper: Faz 6: Yanıt Oluşturma
        API->>+Mapper: map_to_airesult(tahminler)
        Mapper->>Mapper: NORM türetme, triaj
        Mapper-->>-API: AIResult v1.0
    end
    
    API-->>-Klinisyen: SuperclassPredictionResponse
```

---

## 2. Sistem Başlatma Akışı

```mermaid
sequenceDiagram
    autonumber
    participant Main as Ana Modül
    participant App as FastAPI
    participant Validator as Checkpoint Doğrulayıcı
    participant Loader as Model Yükleyici
    participant State as Uygulama Durumu
    
    Main->>+App: FastAPI() oluştur
    App->>App: CORS ve yönlendirme yapılandırması
    App-->>-Main: app örneği
    
    Main->>+App: @on_event("startup")
    
    rect rgb(255, 240, 240)
        Note over App, Validator: Hızlı Başarısızlık (Fail-Fast) Doğrulaması
        App->>+Validator: validate_all_checkpoints(strict=True)
        
        loop Her Checkpoint (binary, superclass, localization)
            Validator->>Validator: Dosya yükle
            Validator->>Validator: Çıktı boyutu kontrolü
            alt Boyut Uyuşmazlığı
                Validator-->>App: CheckpointMismatchError
                App->>App: sys.exit(1)
            end
        end
        
        Validator->>Validator: Mapping fingerprint kontrolü
        alt Fingerprint Değişmiş
            Validator-->>App: MappingDriftError
            App->>App: sys.exit(1)
        end
        
        Validator-->>-App: Tüm doğrulamalar başarılı
    end
    
    rect rgb(240, 255, 240)
        Note over App, State: Model Yükleme
        App->>+State: load_models(yollar)
        
        State->>+Loader: load_model_safe(binary_path)
        Loader-->>-State: binary_model
        
        State->>+Loader: load_model_safe(superclass_path)
        Loader-->>-State: superclass_model
        
        State->>+Loader: load_model_safe(localization_path)
        Loader-->>-State: localization_model
        
        State->>State: XGBoost modelleri yükle
        State->>State: Kalibratörler ve eşikler yükle
        
        State->>State: is_loaded = True
        State-->>-App: Modeller hazır
    end
    
    App-->>-Main: Başlatma tamamlandı
    
    Note over Main, State: API :8000 portunda hazır
```

---

## 3. Tutarlılık Kontrolü

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Çağıran Bileşen
    participant Guard as Tutarlılık Denetimi
    participant Result as Sonuç
    
    Caller->>+Guard: check_consistency(superclass_mi_prob, binary_mi_prob)
    
    Guard->>Guard: superclass_esik = 0.01 (yüksek duyarlılık)
    Guard->>Guard: binary_esik = 0.5 (standart)
    
    Guard->>Guard: superclass_karar = (P_super >= 0.01)
    Guard->>Guard: binary_karar = (P_binary >= 0.5)
    
    alt Her İkisi Pozitif
        Guard->>+Result: AGREE_MI
        Note right of Result: Triaj: YÜKSEK<br/>MI Onaylandı
    else Her İkisi Negatif
        Guard->>+Result: AGREE_NO_MI
        Note right of Result: Triaj: DÜŞÜK<br/>Normal
    else Superclass+, Binary-
        Guard->>+Result: DISAGREE_TYPE_1
        Note right of Result: Triaj: İNCELEME<br/>Düşük Güvenli MI
    else Superclass-, Binary+
        Guard->>+Result: DISAGREE_TYPE_2
        Note right of Result: Triaj: İNCELEME<br/>Olası Kaçırılmış MI
    end
    
    Result-->>-Guard: ConsistencyResult
    Guard-->>-Caller: sonuç, triaj, uyarılar
    
    Caller->>+Guard: should_run_localization(consistency)
    alt AGREE_MI veya DISAGREE_TYPE_1
        Guard-->>Caller: True (Lokalizasyon çalıştır)
    else Diğer
        Guard-->>Caller: False (Atla)
    end
    Guard-->>-Caller: karar
```

---

## 4. XGBoost Hibrit Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant API as API Servisi
    participant Backbone as CNN Omurgası
    participant Scaler as StandardScaler
    participant XGB as XGBoost
    participant Calib as Kalibratör
    participant Ensemble as Ensemble
    
    API->>+Backbone: backbone.forward(X_tensor)
    Note right of Backbone: Conv1d × 2<br/>BatchNorm<br/>ReLU, Dropout<br/>AdaptiveAvgPool
    Backbone-->>-API: gömme: 64-boyutlu vektör
    
    API->>+Scaler: transform(gömme)
    Note right of Scaler: Z-skoru normalizasyonu<br/>(x - μ) / σ
    Scaler-->>-API: ölçekli_gömme
    
    rect rgb(240, 255, 240)
        Note over XGB, Calib: Sınıf Başına Tahmin
        loop MI, STTC, CD, HYP
            API->>+XGB: predict_proba(ölçekli_gömme)
            XGB->>XGB: Karar ağacı ensemble
            XGB-->>-API: P_raw
            
            API->>+Calib: transform(P_raw)
            Note right of Calib: İzotonik Regresyon
            Calib-->>-API: P_calibrated
        end
    end
    
    API->>+Ensemble: combine(P_cnn, P_xgb, alpha=0.15)
    Note right of Ensemble: P_final = α×P_cnn + (1-α)×P_xgb<br/>α = 0.15 (XGBoost ağırlıklı)
    Ensemble-->>-API: P_ensemble
```

---

## 5. Grad-CAM Üretimi

```mermaid
sequenceDiagram
    autonumber
    participant API as API Servisi
    participant GC as GradCAM
    participant Model as CNN Modeli
    participant Viz as Görselleştirici
    
    API->>+GC: GradCAM(model, target_layer)
    GC->>GC: gradients = None
    GC->>GC: activations = None
    GC->>Model: register_forward_hook()
    GC->>Model: register_backward_hook()
    GC-->>-API: gradcam örneği
    
    API->>+GC: generate(X_tensor, class_index)
    
    rect rgb(240, 248, 255)
        Note over GC, Model: İleri Geçiş
        GC->>+Model: forward(X_tensor)
        Note right of Model: Aktivasyonlar yakalandı
        Model-->>-GC: logits
    end
    
    rect rgb(255, 248, 240)
        Note over GC, Model: Geri Geçiş
        GC->>GC: score = logits[:, class_index].sum()
        GC->>+Model: score.backward()
        Note right of Model: Gradyanlar yakalandı
        Model-->>-GC: gradyanlar hesaplandı
    end
    
    rect rgb(240, 255, 240)
        Note over GC: Isı Haritası Hesaplama
        GC->>GC: weights = mean(gradients, dim=temporal)
        GC->>GC: cam = sum(weights × activations)
        GC->>GC: cam = ReLU(cam)
        GC->>GC: cam = normalize(cam, [0, 1])
    end
    
    GC-->>-API: cam_heatmap (1000 nokta)
    
    API->>+Viz: plot_gradcam_overlay(sinyal, cam, sınıf)
    Viz->>Viz: 12 derivasyon grafiği oluştur
    Viz->>Viz: Isı haritası bindirme
    Viz-->>-API: figure
    
    API->>API: save("gradcam_MI.png")
```

---

## 6. Sağlık Kontrolü

```mermaid
sequenceDiagram
    autonumber
    participant LB as Yük Dengeleyici
    participant API as FastAPI
    participant State as Uygulama Durumu
    participant Models as Modeller
    
    rect rgb(240, 255, 240)
        Note over LB, API: Canlılık Kontrolü (Liveness)
        loop Her 30 saniyede
            LB->>+API: GET /health
            API->>API: timestamp al
            API-->>-LB: {"status": "ok", "timestamp": "..."}
            
            alt Yanıt Başarılı
                Note right of LB: Servis çalışıyor
            else Zaman Aşımı veya Hata
                Note right of LB: Konteyneri yeniden başlat
            end
        end
    end
    
    rect rgb(240, 248, 255)
        Note over LB, Models: Hazırlık Kontrolü (Readiness)
        loop Her 10 saniyede
            LB->>+API: GET /ready
            
            API->>+State: check_models_loaded()
            State->>+Models: superclass_model?
            Models-->>-State: True
            State->>+Models: binary_model?
            Models-->>-State: True
            State->>+Models: localization_model?
            Models-->>-State: True
            State->>+Models: xgb_models?
            Models-->>-State: True
            State-->>-API: models_loaded: tümü True
            
            API-->>-LB: {"ready": true, "models_loaded": {...}}
            
            alt Hazır
                Note right of LB: Trafiği yönlendir
            else Hazır Değil
                Note right of LB: Yönlendirmeyi atla
            end
        end
    end
```

---

## 7. Hata Senaryoları

```mermaid
sequenceDiagram
    autonumber
    participant Istemci as İstemci
    participant API as FastAPI
    participant Parser as Ayrıştırıcı
    participant Model as Model
    
    rect rgb(255, 240, 240)
        Note over Istemci, Parser: Senaryo 1: Geçersiz Dosya Formatı
        Istemci->>+API: POST /predict (file.txt)
        API->>+Parser: parse_ecg_file()
        Parser->>Parser: Uzantı kontrolü
        Parser-->>-API: ValueError
        API-->>-Istemci: HTTP 400: "Desteklenmeyen format"
    end
    
    rect rgb(255, 248, 240)
        Note over Istemci, Parser: Senaryo 2: Yanlış Sinyal Boyutu
        Istemci->>+API: POST /predict (wrong.npz)
        API->>+Parser: parse_ecg_file()
        Parser->>Parser: Boyut kontrolü: (8, 500)
        Parser-->>-API: ValueError
        API-->>-Istemci: HTTP 400: "Beklenen (12, 1000)"
    end
    
    rect rgb(240, 240, 255)
        Note over Istemci, Model: Senaryo 3: Model Yüklenmemiş
        Istemci->>+API: POST /predict (valid.npz)
        API->>API: State.is_loaded kontrolü
        API-->>-Istemci: HTTP 503: "Modeller yüklenmedi"
    end
    
    rect rgb(255, 240, 255)
        Note over Istemci, Model: Senaryo 4: İç Hata
        Istemci->>+API: POST /predict (valid.npz)
        API->>+Model: forward(X_tensor)
        Model->>Model: RuntimeError
        Model-->>-API: Exception
        API->>API: Hata günlükle
        API-->>-Istemci: HTTP 500: "Tahmin başarısız"
    end
```

---

## 8. Planlanan Akışlar (v2.0)

### 8.1 RAG Entegrasyonu

```mermaid
sequenceDiagram
    autonumber
    participant API as API Servisi
    participant RAG as RAG Retriever
    participant VectorDB as Vektör Veritabanı
    participant LLM as Büyük Dil Modeli
    
    API->>+RAG: retrieve_context(tahmin_sonucu)
    RAG->>RAG: Sorgu gömme vektörü oluştur
    RAG->>+VectorDB: similarity_search(query_embedding)
    VectorDB-->>-RAG: İlgili klinik kılavuzlar
    RAG-->>-API: bağlam_belgeleri
    
    API->>+LLM: generate_report(tahmin, bağlam)
    LLM->>LLM: Bulguları formatla
    LLM->>LLM: Klinik öneriler oluştur
    LLM-->>-API: klinik_rapor
```

### 8.2 Belirsizlik Tahmini

```mermaid
sequenceDiagram
    autonumber
    participant API as API Servisi
    participant Model as CNN Model
    participant Stats as İstatistik Modülü
    
    API->>Model: model.train() (dropout aktif)
    
    loop N iterasyon (N=30)
        API->>+Model: forward(X_tensor)
        Note right of Model: Dropout rasgele<br/>farklı sonuçlar
        Model-->>-API: tahmin_i
    end
    
    API->>+Stats: compute_statistics(tahminler)
    Stats->>Stats: ortalama = mean(tahminler)
    Stats->>Stats: varyans = var(tahminler)
    Stats->>Stats: güven_aralığı = 1.96 × std
    Stats-->>-API: {ortalama, varyans, güven_aralığı}
```

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Yazılım Mimarı | | | |
| Teknik Lider | | | |
| Kalite Güvence Mühendisi | | | |

---

**Doküman Sonu**
