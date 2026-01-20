# CardioGuard-AI
# Gereksinim Spesifikasyonu

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Yazılım Gereksinimleri Spesifikasyonu (SRS)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## İçindekiler

1. [Giriş](#1-giriş)
2. [Fonksiyonel Gereksinimler](#2-fonksiyonel-gereksinimler)
3. [Fonksiyonel Olmayan Gereksinimler](#3-fonksiyonel-olmayan-gereksinimler)
4. [Gereksinim İzlenebilirlik Matrisi](#4-gereksinim-izlenebilirlik-matrisi)
5. [Kaynak Dosyalar](#5-kaynak-dosyalar)

---

## 1. Giriş

### 1.1 Amaç

Bu doküman, CardioGuard-AI sisteminin tüm fonksiyonel ve fonksiyonel olmayan gereksinimlerini tanımlar. Sistem, 12 derivasyonlu EKG sinyallerinden kardiyak patolojileri tespit eden hibrit bir yapay zeka platformudur.

### 1.2 Kapsam

- EKG sinyal işleme ve analizi
- Çoklu etiketli patoloji sınıflandırması
- Miyokard enfarktüsü anatomik lokalizasyonu
- Açıklanabilir yapay zeka (XAI) desteği
- REST API servisi

### 1.3 Tanımlar ve Kısaltmalar

| Kısaltma | Açıklama |
|----------|----------|
| MI | Myocardial Infarction (Miyokard Enfarktüsü) |
| STTC | ST/T Change (ST/T Değişikliği) |
| CD | Conduction Disturbance (İletim Bozukluğu) |
| HYP | Hypertrophy (Hipertrofi) |
| NORM | Normal EKG |
| XAI | Explainable AI (Açıklanabilir Yapay Zeka) |
| CNN | Convolutional Neural Network (Evrişimsel Sinir Ağı) |
| XGBoost | Extreme Gradient Boosting |
| PTB-XL | PhysioNet PTB-XL Veritabanı |

---

## 2. Fonksiyonel Gereksinimler

### 2.1 Veri Yükleme ve İşleme Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Durum |
|----|------------|----------|---------|-------|
| FR-01 | PTB-XL Veri Yükleme | Sistem, PTB-XL veri setinden EKG sinyallerini yükleyebilmelidir. WFDB formatı desteklenmelidir. | Yüksek | Tamamlandı |
| FR-02 | 12 Derivasyon Sinyal İşleme | Sistem, 12 derivasyonlu EKG sinyallerini işleyebilmelidir. Sinyal boyutu: 1000 örnek × 12 kanal, örnekleme frekansı: 100 Hz. | Yüksek | Tamamlandı |
| FR-03 | Sinyal Normalizasyonu | Sistem, MinMax normalizasyonu ile her derivasyon için 0-1 aralığında ölçekleme yapabilmelidir. | Yüksek | Tamamlandı |

### 2.2 Sınıflandırma ve Tahmin Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Durum |
|----|------------|----------|---------|-------|
| FR-04 | Patoloji Sınıflandırması | Sistem, dört patoloji sınıfını (MI, STTC, CD, HYP) çoklu etiket olarak tespit edebilmelidir. Her sınıf için olasılık skoru ve karar çıktısı üretilmelidir. | Yüksek | Tamamlandı |
| FR-05 | MI Lokalizasyonu | Sistem, MI tespit edildiğinde beş anatomik bölgeyi (AMI, ASMI, ALMI, IMI, LMI) lokalize edebilmelidir. Çoklu bölge tespiti desteklenmelidir. | Yüksek | Tamamlandı |
| FR-06 | Tutarlılık Kontrolü | Sistem, Binary MI ve Superclass MI modelleri arasında tutarlılık kontrolü yapabilmelidir. Uyumsuzluk durumunda inceleme triajı atanmalıdır. | Yüksek | Tamamlandı |
| FR-10 | Triaj Seviyeleri | Sistem, tahmin sonuçlarına göre triaj seviyesi (Yüksek, Orta, Düşük, İnceleme) belirleyebilmelidir. | Yüksek | Tamamlandı |

### 2.3 Hibrit Model Pipeline Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Durum |
|----|------------|----------|---------|-------|
| FR-11 | NORM Türetme | Sistem, NORM skorunu "1 - max(P_MI, P_STTC, P_CD, P_HYP)" formülü ile türetebilmelidir. NORM bir sınıflandırıcı çıktısı değildir, diğer olasılıklardan türetilir. | Orta | Tamamlandı |
| FR-12 | Ensemble Kombinasyonu | Sistem, CNN ve XGBoost olasılıklarını ağırlıklı ortalama ile birleştirebilmelidir. Varsayılan ağırlık değeri: α = 0.15. | Orta | Tamamlandı |

### 2.4 Açıklanabilirlik (XAI) Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Durum |
|----|------------|----------|---------|-------|
| FR-07 | Grad-CAM Açıklamaları | Sistem, CNN modeli için Grad-CAM ile zamansal açıklama haritaları üretebilmelidir. SmoothGrad-CAM desteği sağlanmalıdır. | Orta | Tamamlandı |
| FR-08 | SHAP Açıklamaları | Sistem, XGBoost modelleri için SHAP TreeExplainer ile özellik önem skorları üretebilmelidir. | Orta | Tamamlandı |

### 2.5 API Servisi Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Durum |
|----|------------|----------|---------|-------|
| FR-09 | REST API | Sistem, FastAPI framework üzerinden tahmin servisi sunabilmelidir. Desteklenen uç noktalar: /predict/superclass, /predict/mi-localization, /health | Yüksek | Tamamlandı |

---

## 3. Fonksiyonel Olmayan Gereksinimler

### 3.1 Güvenilirlik ve Güvenlik Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Metrik |
|----|------------|----------|---------|--------|
| NFR-01 | Hızlı Başarısızlık (Fail-Fast) | Kontrol noktası doğrulaması başlangıçta yapılmalı, hatalı kontrol noktası varsa sistem başlamamalıdır. | Yüksek | Başlangıç süresi < 5 saniye |
| NFR-05 | Kontrat Uyumu | Tüm API çıktıları AIResult v1.0 kontratına uygun olmalıdır. | Yüksek | %100 uyum |
| NFR-06 | Eşleme Parmak İzi | MI lokalizasyon eşleme parmak izi değişikliğinde sistem uyarı vermeli ve başlamamalıdır. | Yüksek | Parmak izi: 8ab274e06afa1be8 |

### 3.2 Performans Gereksinimleri

| ID | Gereksinim | Açıklama | Öncelik | Metrik |
|----|------------|----------|---------|--------|
| NFR-02 | API Yanıt Süresi | Tek tahmin için API yanıt süresi kabul edilebilir olmalıdır. | Orta | < 500 ms (CPU) |
| NFR-03 | Model Boyutu | Toplam kontrol noktası boyutu optimize edilmelidir. | Düşük | < 2 MB |
| NFR-04 | CPU Desteği | Sistem, GPU olmadan sadece CPU üzerinde çalışabilmelidir. | Yüksek | PyTorch CPU backend |

---

## 4. Gereksinim İzlenebilirlik Matrisi

### 4.1 Gereksinim Bağımlılıkları

| Kaynak Gereksinim | Hedef Gereksinim | İlişki Tipi |
|-------------------|------------------|-------------|
| FR-01 | FR-04 | Önkoşul |
| FR-02 | FR-04 | Önkoşul |
| FR-03 | FR-04 | Önkoşul |
| FR-04 | FR-05 | Tetikler |
| FR-04 | FR-06 | Tetikler |
| FR-06 | FR-10 | Tetikler |
| FR-04 | FR-07, FR-08 | Tetikler |
| FR-04 | FR-09 | Kullanılır |

### 4.2 Gereksinim Kategorileri

| Kategori | Gereksinim Sayısı | Tamamlanma Durumu |
|----------|-------------------|-------------------|
| Veri İşleme | 3 | 3/3 (%100) |
| Sınıflandırma | 6 | 6/6 (%100) |
| Açıklanabilirlik | 2 | 2/2 (%100) |
| API Servisi | 1 | 1/1 (%100) |
| Fonksiyonel Olmayan | 6 | 6/6 (%100) |

---

## 5. Kaynak Dosyalar

| Gereksinim | İlgili Kaynak Dosyaları |
|------------|-------------------------|
| FR-01, FR-02 | src/data/signals.py, src/data/loader.py |
| FR-03 | src/data/signals.py (normalizasyon fonksiyonları) |
| FR-04 | src/models/cnn.py, src/pipeline/training/train_superclass_cnn.py |
| FR-05 | src/data/mi_localization.py, src/pipeline/training/train_mi_localization.py |
| FR-06 | src/pipeline/inference/consistency_guard.py |
| FR-07 | src/xai/gradcam.py |
| FR-08 | src/xai/shap_xgb.py, src/xai/shap_ovr.py |
| FR-09 | src/backend/main.py |
| FR-10 | src/contracts/airesult_mapper.py |
| FR-11 | src/pipeline/inference/consistency_guard.py |
| FR-12 | src/models/xgb.py |
| NFR-01, NFR-06 | src/utils/checkpoint_validation.py |
| NFR-05 | src/contracts/airesult_mapper.py |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Proje Yöneticisi | | | |
| Teknik Lider | | | |
| Kalite Güvence Mühendisi | | | |

---

**Doküman Sonu**

*Bu doküman CardioGuard-AI v1.0.0 için geçerlidir. Gelecek versiyonlarda PMI (Posterior MI) desteği, Monte Carlo Dropout ile belirsizlik tahmini ve RAG entegrasyonu planlanmaktadır.*
