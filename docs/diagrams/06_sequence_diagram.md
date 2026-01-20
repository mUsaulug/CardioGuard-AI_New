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
2. [Model Yükleme ve Başlatma](#2-model-yükleme-ve-başlatma)
3. [Tutarlılık Kontrolü Detayı](#3-tutarlılık-kontrolü-detayı)
4. [XGBoost Hibrit Pipeline](#4-xgboost-hibrit-pipeline)
5. [Grad-CAM Açıklama Üretimi](#5-grad-cam-açıklama-üretimi)
6. [Model Eğitim Akışı](#6-model-eğitim-akışı)
7. [Sağlık Kontrolü ve Hazırlık](#7-sağlık-kontrolü-ve-hazırlık)
8. [Hata Senaryoları](#8-hata-senaryoları)
9. [Toplu Tahmin Akışı](#9-toplu-tahmin-akışı)
10. [Özet Akış Şeması](#10-özet-akış-şeması)

---

## 1. Ana Tahmin Akışı

Bu diyagram, bir EKG sinyalinin yüklenip analiz edilmesinden sonuç üretilmesine kadar olan tüm akışı göstermektedir.

```mermaid
sequenceDiagram
    autonumber
    
    participant Klinisyen
    participant API as API Servisi
    participant Parser as EKG Ayrıştırıcı
    participant Norm as Normalleştirici
    participant State as Uygulama Durumu
    participant Super as Superclass CNN
    participant Binary as Binary CNN
    participant XGB as XGBoost
    participant Guard as Tutarlılık Denetimi
    participant Loc as Lokalizasyon CNN
    participant XAI as GradCAM
    participant Mapper as AIResult Eşleyici
    
    Note over Klinisyen, Mapper: Faz 1: Girdi İşleme
    
    Klinisyen->>+API: POST /predict/superclass (EKG dosyası)
    API->>+Parser: parse_ecg_file(içerik, dosya_adı)
    Parser->>Parser: Format doğrulama (.npz/.npy)
    Parser->>Parser: Boyut kontrolü (12×1000)
    Parser-->>-API: sinyal: ndarray[12, 1000]
    
    API->>+Norm: normalize(sinyal)
    Norm->>Norm: Her derivasyon için MinMax normalizasyonu
    Norm-->>-API: X_norm: ndarray[12, 1000]
    
    API->>API: torch.tensor(X_norm), boyut: (1, 12, 1000)
    
    Note over API, Binary: Faz 2: CNN Tahminleri
    
    API->>+State: get_models()
    State-->>-API: superclass_model, binary_model
    
    API->>+Super: forward(X_tensor)
    Super->>Super: backbone - gömme (64 boyut)
    Super->>Super: head - logits (4 boyut)
    Super->>Super: sigmoid - olasılıklar
    Super-->>-API: P_cnn: MI, STTC, CD, HYP
    
    API->>+Binary: forward(X_tensor)
    Binary->>Binary: backbone - gömme
    Binary->>Binary: head - logit (1 boyut)
    Binary->>Binary: sigmoid - olasılık
    Binary-->>-API: P_binary_MI: 0.78
    
    Note over API, XGB: Faz 3: XGBoost Takviyesi
    
    API->>+Super: backbone.forward(X_tensor)
    Super-->>-API: gömme: (1, 64)
    
    API->>+State: get_xgb_models()
    State-->>-API: xgb_models, scaler, ayarlayıcılar
    
    API->>API: scaler.transform(gömme)
    
    loop Her sınıf için [MI, STTC, CD, HYP]
        API->>+XGB: predict_proba(ölçekli_gömme)
        XGB-->>-API: P_raw[sınıf]
        API->>API: calibrator.transform(P_raw)
    end
    
    API->>API: P_xgb: MI, STTC, CD, HYP
    
    Note over API: Ensemble: P = alpha × P_cnn + (1-alpha) × P_xgb
    API->>API: P_ensemble hesaplama (alpha = 0.15)
    
    Note over API, Guard: Faz 4: Tutarlılık Kontrolü
    
    API->>+Guard: check_consistency(P_super_MI, P_binary_MI)
    Guard->>Guard: superclass_karar = (0.87 >= 0.01) = True
    Guard->>Guard: binary_karar = (0.78 >= 0.5) = True
    Guard->>Guard: uyum = AGREE_MI
    Guard->>Guard: triaj = YÜKSEK
    Guard-->>-API: ConsistencyResult(AGREE_MI, YÜKSEK)
    
    Note over API, Loc: Faz 5: MI Lokalizasyonu (Koşullu)
    
    API->>+Guard: should_run_localization(consistency)
    Guard-->>-API: True (AGREE_MI)
    
    API->>+State: get_localization_model()
    State-->>-API: localization_model
    
    API->>+Loc: forward(X_tensor)
    Loc->>Loc: backbone - gömme
    Loc->>Loc: head - logits (5 boyut)
    Loc->>Loc: sigmoid - olasılıklar
    Loc-->>-API: P_loc: AMI, ASMI, ALMI, IMI, LMI
    
    API->>API: detected_regions = [AMI, IMI] (eşik: 0.5)
    
    Note over API, XAI: Faz 6: XAI Üretimi
    
    API->>+XAI: GradCAM(model, hedef_katman)
    XAI->>XAI: register_hooks()
    API->>+XAI: generate(X_tensor, class_index=0)
    XAI->>XAI: ileri geçiş
    XAI->>XAI: geri geçiş
    XAI->>XAI: ağırlıklı aktivasyonlar hesaplama
    XAI->>XAI: ReLU + normalize
    XAI-->>-API: cam_heatmap: ndarray[1000]
    XAI-->>-API: tamamlandı
    
    API->>API: save_artifact(cam_heatmap, "gradcam_MI.png")
    
    Note over API, Mapper: Faz 7: Yanıt Oluşturma
    
    API->>+Mapper: map_predict_output_to_airesult(...)
    Mapper->>Mapper: olasılıkları doğrula
    Mapper->>Mapper: compute_triage()
    Mapper->>Mapper: NORM türet = 1 - max(olasılıklar)
    Mapper->>Mapper: XAI yapıtları bul
    Mapper->>Mapper: AIResult v1.0 oluştur
    Mapper-->>-API: airesult: Dict
    
    Note over Klinisyen, Mapper: Yanıt Dönüşü
    
    API-->>-Klinisyen: SuperclassPredictionResponse
```

---

## 2. Model Yükleme ve Başlatma

```mermaid
sequenceDiagram
    autonumber
    
    participant Main as Ana Modül
    participant App as FastAPI Uygulaması
    participant Validator as Kontrol Noktası Doğrulayıcı
    participant Loader as Güvenli Model Yükleyici
    participant State as Uygulama Durumu
    
    Note over Main, State: Uygulama Başlatma
    
    Main->>+App: FastAPI()
    App->>App: CORS ve yönlendirme yapılandırması
    App-->>-Main: app örneği
    
    Main->>+App: on_event("startup")
    
    Note over App, Validator: Hızlı Başarısızlık Doğrulaması
    
    App->>+Validator: validate_all_checkpoints(strict=True)
    
    loop Her kontrol noktası için [binary, superclass, localization]
        Validator->>Validator: load_checkpoint(yol)
        Validator->>Validator: extract_output_dimension(state_dict)
        
        alt Boyut Uyuşmazlığı
            Validator-->>App: CheckpointMismatchError
            App->>App: sys.exit(1)
        end
    end
    
    Validator->>Validator: compute_mapping_fingerprint()
    alt Parmak İzi Değişikliği
        Validator-->>App: MappingDriftError
        App->>App: sys.exit(1)
    end
    
    Validator-->>-App: Tüm kontrol noktaları geçerli
    
    Note over App, State: Model Yükleme
    
    App->>+State: load_models(yollar)
    
    State->>+Loader: load_model_safe(binary_path, ECGCNN, config)
    Loader->>Loader: torch.load(path)
    Loader->>Loader: normalize_state_dict()
    Loader->>Loader: model.load_state_dict()
    Loader-->>-State: binary_model
    
    State->>+Loader: load_model_safe(superclass_path, ECGCNN, config)
    Loader-->>-State: superclass_model
    
    State->>+Loader: load_model_safe(localization_path, ECGCNN, config)
    Loader-->>-State: localization_model
    
    State->>State: XGBoost modellerini yükle
    State->>State: Ölçekleyici ve ayarlayıcıları yükle
    State->>State: Eşikleri yükle
    
    State->>State: is_loaded = True
    State-->>-App: Modeller yüklendi
    
    App-->>-Main: Başlatma tamamlandı
    
    Note over Main, State: API :8000 portunda hazır
```

---

## 3. Tutarlılık Kontrolü Detayı

```mermaid
sequenceDiagram
    autonumber
    
    participant Cagiran as Çağıran Bileşen
    participant Guard as Tutarlılık Denetimi
    participant Result as Tutarlılık Sonucu
    
    Note over Cagiran, Result: Tutarlılık Kontrolü
    
    Cagiran->>+Guard: check_consistency(superclass_mi_prob=0.87, binary_mi_prob=0.78, esikler)
    
    Guard->>Guard: superclass_esik = 0.01
    Guard->>Guard: binary_esik = 0.5
    
    Guard->>Guard: superclass_karar = (0.87 >= 0.01) = True
    Guard->>Guard: binary_karar = (0.78 >= 0.5) = True
    
    alt Her İkisi de True (AGREE_MI)
        Guard->>+Result: ConsistencyResult(uyum=AGREE_MI, triaj="YÜKSEK")
        Note right of Result: Yüksek Öncelik - MI Onaylandı
    else Her İkisi de False (AGREE_NO_MI)
        Guard->>+Result: ConsistencyResult(uyum=AGREE_NO_MI, triaj="DÜŞÜK")
        Note right of Result: Düşük Öncelik - Normal
    else Super+ Binary- (DISAGREE_TYPE_1)
        Guard->>+Result: ConsistencyResult(uyum=DISAGREE_TYPE_1, triaj="İNCELEME")
        Note right of Result: İnceleme Gerekli - Düşük Güvenli MI
    else Super- Binary+ (DISAGREE_TYPE_2)
        Guard->>+Result: ConsistencyResult(uyum=DISAGREE_TYPE_2, triaj="İNCELEME")
        Note right of Result: İnceleme Gerekli - Superclass Kaçırmış
    end
    
    Result-->>-Guard: result
    Guard-->>-Cagiran: ConsistencyResult
    
    Note over Cagiran, Result: Lokalizasyon Kapısı
    
    Cagiran->>+Guard: should_run_localization(result)
    
    alt AGREE_MI veya DISAGREE_TYPE_1
        Guard-->>Cagiran: True (Lokalizasyon Çalıştır)
    else AGREE_NO_MI veya DISAGREE_TYPE_2
        Guard-->>Cagiran: False (Lokalizasyonu Atla)
    end
    Guard-->>-Cagiran: karar
```

---

## 4. XGBoost Hibrit Pipeline

```mermaid
sequenceDiagram
    autonumber
    
    participant API as API Servisi
    participant CNN as CNN Omurgası
    participant Scaler as StandardScaler
    participant XGB as XGBoost
    participant Calib as Ayarlayıcı
    participant Ensemble as Ensemble Birleştirici
    
    Note over API, Ensemble: Hibrit Pipeline
    
    API->>+CNN: backbone.forward(X_tensor)
    Note right of CNN: Conv1d - BN - ReLU - Dropout × 2 ve AdaptiveAvgPool1d
    CNN-->>-API: gömme: (1, 64)
    
    API->>+Scaler: transform(gömme)
    Note right of Scaler: Z-skoru normalizasyonu: (x - ortalama) / standart_sapma
    Scaler-->>-API: ölçekli_gömme
    
    Note over XGB, Calib: Sınıf Başına Tahmin
    
    loop Her sınıf için [MI, STTC, CD, HYP]
        API->>+XGB: predict_proba(ölçekli_gömme)
        XGB->>XGB: karar ağacı ensemble çıkarımı
        XGB-->>-API: P_raw: 0.85
        
        API->>+Calib: transform(P_raw)
        Note right of Calib: İzotonik Regresyon
        Calib-->>-API: P_calibrated: 0.88
    end
    
    Note over API: P_xgb = {MI: 0.88, STTC: 0.15, ...}
    
    Note over Ensemble: Ensemble Kombinasyonu
    
    API->>+Ensemble: combine(P_cnn, P_xgb, alpha=0.15)
    
    Ensemble->>Ensemble: P_MI = 0.15 × 0.82 + 0.85 × 0.88
    Ensemble->>Ensemble: P_STTC = 0.15 × 0.12 + 0.85 × 0.15
    Ensemble->>Ensemble: Diğer sınıflar...
    
    Note right of Ensemble: P_final = alpha × P_cnn + (1-alpha) × P_xgb
    Ensemble-->>-API: P_ensemble: {MI: 0.87, ...}
```

---

## 5. Grad-CAM Açıklama Üretimi

```mermaid
sequenceDiagram
    autonumber
    
    participant API as API Servisi
    participant GC as GradCAM
    participant Model as CNN Modeli
    participant Target as Hedef Katman
    participant Viz as Görselleştirici
    
    Note over API, Viz: Grad-CAM Üretimi
    
    API->>+GC: GradCAM(model, hedef_katman)
    GC->>GC: self.gradients = None
    GC->>GC: self.activations = None
    
    GC->>+Target: register_forward_hook()
    Note right of Target: Aktivasyonları yakalar
    Target-->>-GC: hook_handle
    
    GC->>+Target: register_backward_hook()
    Note right of Target: Gradyanları yakalar
    Target-->>-GC: hook_handle
    
    GC-->>-API: gradcam_instance
    
    Note over API, Model: İleri ve Geri Geçiş
    
    API->>+GC: generate(X_tensor, class_index=0)
    
    GC->>+Model: forward(X_tensor)
    Note right of Target: Hook yakalıyor: self.activations
    Model-->>-GC: logits: (1, 4)
    
    GC->>GC: score = logits[:, 0].sum()
    GC->>+Model: score.backward()
    Note right of Target: Hook yakalıyor: self.gradients
    Model-->>-GC: gradyanlar hesaplandı
    
    GC->>GC: weights = mean(gradients, dim=2)
    Note right of GC: Zamansal boyut üzerinden Global Ortalama Havuzlama
    
    GC->>GC: cam = sum(weights × activations, dim=1)
    GC->>GC: cam = ReLU(cam)
    GC->>GC: cam = (cam - min) / (max - min)
    Note right of GC: [0, 1] aralığına normalize et
    
    GC-->>-API: cam_heatmap: (1, 1000)
    
    Note over API, Viz: Görselleştirme
    
    API->>+Viz: plot_gradcam_overlay(sinyal, cam, "MI")
    Viz->>Viz: Figür oluştur (12 alt grafik)
    Viz->>Viz: Her derivasyon için EKG sinyali çiz
    Viz->>Viz: Renk haritası ile ısı haritası bindirme
    Viz->>Viz: Renk çubuğu ve etiketler ekle
    Viz-->>-API: figure
    
    API->>API: figure.savefig("gradcam_MI.png")
```

---

## 6. Model Eğitim Akışı

```mermaid
sequenceDiagram
    autonumber
    
    participant Kullanici as ML Mühendisi
    participant Script as Eğitim Betiği
    participant Data as Veri Yükleyici
    participant Model as ECGCNN
    participant Optim as Optimize Edici
    participant Sched as Zamanlayıcı
    participant Logger as Günlükleyici
    
    Note over Kullanici, Logger: Model Eğitimi
    
    Kullanici->>+Script: python -m train --epochs 50
    
    Note over Script, Data: Başlatma
    
    Script->>+Data: create_dataloaders(batch_size=64)
    Data->>Data: PTB-XL metadata yükle
    Data->>Data: Superclass etiketleri ekle
    Data->>Data: Böl: eğitim/doğrulama/test
    Data->>Data: SignalDataset oluştur
    Data-->>-Script: train_loader, val_loader
    
    Script->>+Model: ECGCNN(config, num_classes=4)
    Model-->>-Script: model
    
    Script->>Optim: Adam(lr=1e-3)
    Script->>Sched: ReduceLROnPlateau
    
    Note over Script, Model: Eğitim Döngüsü
    
    loop Her epok için (50 epok)
        Script->>+Model: model.train()
        
        loop train_loader içindeki her yığın için
            Data-->>Script: X_batch, y_batch
            
            Script->>+Model: forward(X_batch)
            Model-->>-Script: logits
            
            Script->>Script: loss = BCEWithLogitsLoss(logits, y_batch)
            
            Script->>Optim: zero_grad()
            Script->>Script: loss.backward()
            Script->>Optim: step()
        end
        
        Model-->>-Script: epok tamamlandı
        
        Script->>+Model: model.eval()
        Script->>Script: val_loader üzerinde doğrulama
        Script->>Script: AUROC, AUPRC hesapla
        Model-->>-Script: val_metrics
        
        Script->>Sched: step(val_loss)
        Script->>Logger: log_metrics(epok, train_loss, val_metrics)
        
        alt En iyi model ise
            Script->>Script: save_checkpoint("best_model.pt")
        end
    end
    
    Note over Script, Logger: Sonlandırma
    
    Script->>Script: En iyi kontrol noktasını yükle
    Script->>Script: Test seti üzerinde değerlendirme
    Script->>Logger: log_final_metrics()
    Script->>Script: Son modeli kaydet
    
    Script-->>-Kullanici: Eğitim tamamlandı, checkpoints/ecgcnn_superclass.pt
```

---

## 7. Sağlık Kontrolü ve Hazırlık

```mermaid
sequenceDiagram
    autonumber
    
    participant LB as Yük Dengeleyici
    participant API as FastAPI
    participant State as Uygulama Durumu
    participant Models as Modeller
    
    Note over LB, Models: Sağlık ve Hazırlık Kontrolleri
    
    Note over LB, API: Canlılık Kontrolü
    
    loop Her 30 saniyede
        LB->>+API: GET /health
        API->>API: Mevcut zaman damgasını al
        API-->>-LB: {"status": "ok", "timestamp": "..."}
        
        alt Yanıt Başarılı
            Note right of LB: Servis çalışıyor
        else Zaman Aşımı veya Hata
            Note right of LB: Konteyneri yeniden başlat
            LB->>LB: Yeniden başlatma tetikle
        end
    end
    
    Note over LB, Models: Hazırlık Kontrolü
    
    loop Her 10 saniyede
        LB->>+API: GET /ready
        
        API->>+State: check_models_loaded()
        State->>+Models: superclass_model yüklü mü?
        Models-->>-State: True
        State->>+Models: binary_model yüklü mü?
        Models-->>-State: True
        State->>+Models: localization_model yüklü mü?
        Models-->>-State: True
        State->>+Models: xgb_models yüklü mü?
        Models-->>-State: True
        State-->>-API: models_loaded: tümü True
        
        API-->>-LB: {"ready": true, "models_loaded": {...}}
        
        alt Hazır
            Note right of LB: Trafiği yönlendir
        else Hazır Değil
            Note right of LB: Yönlendirmeyi atla
        end
    end
```

---

## 8. Hata Senaryoları

```mermaid
sequenceDiagram
    autonumber
    
    participant Istemci as İstemci
    participant API as FastAPI
    participant Parser as Ayrıştırıcı
    participant Model as Model
    
    Note over Istemci, Model: Hata Senaryoları
    
    Note over Istemci, Parser: Senaryo 1: Geçersiz Dosya Formatı
    
    Istemci->>+API: POST /predict (file.txt)
    API->>+Parser: parse_ecg_file()
    Parser->>Parser: Uzantıyı kontrol et
    Parser-->>-API: ValueError
    API-->>-Istemci: HTTP 400 Hatalı İstek: "Desteklenmeyen format. .npz veya .npy kullanın"
    
    Note over Istemci, Parser: Senaryo 2: Yanlış Sinyal Boyutu
    
    Istemci->>+API: POST /predict (wrong_shape.npz)
    API->>+Parser: parse_ecg_file()
    Parser->>Parser: Sinyali yükle
    Parser->>Parser: Boyut kontrolü (8, 500) ≠ (12, 1000)
    Parser-->>-API: ValueError
    API-->>-Istemci: HTTP 400 Hatalı İstek: "Beklenen boyut (12, 1000), alınan (8, 500)"
    
    Note over Istemci, Model: Senaryo 3: Model Yüklenmemiş
    
    Istemci->>+API: POST /predict (valid.npz)
    API->>API: State.is_loaded kontrolü
    API-->>-Istemci: HTTP 503 Servis Kullanılamıyor: "Modeller yüklenmedi. Lütfen bekleyin."
    
    Note over Istemci, Model: Senaryo 4: İç Hata
    
    Istemci->>+API: POST /predict (valid.npz)
    API->>+Model: forward(X_tensor)
    Model->>Model: RuntimeError (CUDA bellek yetersiz)
    Model-->>-API: Exception
    API->>API: Hata detaylarını günlükle
    API-->>-Istemci: HTTP 500 İç Sunucu Hatası: "Tahmin başarısız. Günlüklere bakın."
```

---

## 9. Toplu Tahmin Akışı

```mermaid
sequenceDiagram
    autonumber
    
    participant Script as Toplu Tahmin Betiği
    participant Loader as Veri Yükleyici
    participant Model as Modeller
    participant Writer as CSV Yazıcı
    
    Note over Script, Writer: Toplu Tahmin
    
    Script->>+Loader: load_test_data()
    Loader-->>-Script: test_loader (2179 örnek)
    
    Script->>Script: results = [] başlat
    
    loop test_loader içindeki her yığın için (batch_size=32)
        Loader-->>Script: X_batch: (32, 12, 1000)
        
        Script->>+Model: superclass_model(X_batch)
        Model-->>-Script: P_cnn: (32, 4)
        
        Script->>+Model: binary_model(X_batch)
        Model-->>-Script: P_binary: (32,)
        
        Script->>Script: Her örnek için tutarlılık kontrolü
        
        Script->>+Model: localization_model(X_batch[mi_mask])
        Model-->>-Script: P_loc: (n_mi, 5)
        
        Script->>Script: Tahminleri results listesine ekle
    end
    
    Script->>+Writer: write_csv("predictions.csv")
    Writer->>Writer: DataFrame oluştur
    Writer->>Writer: Diske kaydet
    Writer-->>-Script: Kaydedildi
    
    Note over Script, Writer: 2179 tahmin kaydedildi
```

---

## 10. Özet Akış Şeması

```mermaid
graph TB
    subgraph Girdi["1. Girdi"]
        UPLOAD["EKG Yükleme"]
        PARSE["Format Kontrolü"]
        NORM["Normalizasyon"]
    end
    
    subgraph Tahmin["2. Tahmin"]
        CNN["CNN Tahminleri"]
        XGB["XGBoost Tahminleri"]
        ENS["Ensemble"]
    end
    
    subgraph Dogrulama["3. Doğrulama"]
        GUARD["Tutarlılık Kontrolü"]
        TRIAGE["Triaj Belirleme"]
    end
    
    subgraph Lokalizasyon["4. Lokalizasyon"]
        CHECK{"MI?"}
        LOC["Bölge Tespiti"]
    end
    
    subgraph XAI["5. Açıklanabilirlik"]
        GCAM["Grad-CAM"]
        VIZ["Görselleştirme"]
    end
    
    subgraph Cikti["6. Çıktı"]
        MAPPER["AIResult Eşleyici"]
        RESPONSE["JSON Yanıt"]
    end
    
    UPLOAD --> PARSE --> NORM --> CNN
    NORM --> XGB
    CNN --> ENS
    XGB --> ENS
    ENS --> GUARD --> TRIAGE
    GUARD --> CHECK
    CHECK -->|Evet| LOC --> GCAM
    CHECK -->|Hayır| GCAM
    GCAM --> VIZ --> MAPPER --> RESPONSE
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

*Bu sıralı akış diyagramları CardioGuard-AI v1.0.0 akışlarını göstermektedir. Tüm diyagramlar Mermaid formatındadır ve GitHub, GitLab veya uyumlu Markdown görüntüleyicilerde işlenebilir.*
