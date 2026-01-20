# CardioGuard-AI
# Dağıtım Diyagramı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Dağıtım Diyagramı (Deployment Diagram)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026  
**Hazırlayan:** CardioGuard-AI Geliştirme Ekibi

---

## İçindekiler

1. [Genel Mimari](#1-genel-mimari)
2. [Bileşen Detayları](#2-bileşen-detayları)
3. [Dosya Sistemi Yapısı](#3-dosya-sistemi-yapısı)
4. [Ağ Mimarisi](#4-ağ-mimarisi)
5. [Docker Dağıtımı](#5-docker-dağıtımı)
6. [Geliştirme Ortamı](#6-geliştirme-ortamı)
7. [Üretim Ortamı](#7-üretim-ortamı)
8. [CI/CD Pipeline](#8-cicd-pipeline)
9. [Güvenlik Mimarisi](#9-güvenlik-mimarisi)
10. [İzleme ve Günlükleme](#10-izleme-ve-günlükleme)

---

## 1. Genel Mimari

```mermaid
graph TB
    subgraph Internet["İnternet"]
        CLIENT["İstemci (Web Tarayıcı / API İstemcisi)"]
    end
    
    subgraph Yuk_Dengeleyici["Yük Dengeleyici"]
        NGINX["NGINX Ters Proxy (:80/:443)"]
    end
    
    subgraph Uygulama_Sunucusu["Uygulama Sunucusu"]
        subgraph Container1["Docker Konteyner 1"]
            FASTAPI1["FastAPI (:8000)"]
            MODELS1["PyTorch Modelleri"]
            XGB1["XGBoost"]
        end
        
        subgraph Container2["Docker Konteyner 2"]
            FASTAPI2["FastAPI (:8001)"]
            MODELS2["PyTorch Modelleri"]
            XGB2["XGBoost"]
        end
    end
    
    subgraph Depolama["Depolama"]
        CHECKPOINTS["Kontrol Noktaları"]
        ARTIFACTS["Yapıtlar"]
        LOGS["Günlükler"]
    end
    
    subgraph Veri_Katmani["Veri Katmanı"]
        PTBXL["PTB-XL Veri Seti (Salt Okunur)"]
    end
    
    CLIENT --> NGINX
    NGINX --> FASTAPI1
    NGINX --> FASTAPI2
    FASTAPI1 --> CHECKPOINTS
    FASTAPI1 --> ARTIFACTS
    FASTAPI1 --> LOGS
    FASTAPI2 --> CHECKPOINTS
    FASTAPI2 --> ARTIFACTS
    FASTAPI2 --> LOGS
    FASTAPI1 --> PTBXL
    FASTAPI2 --> PTBXL
```

---

## 2. Bileşen Detayları

### 2.1 Uygulama Sunucusu Bileşenleri

```mermaid
graph TB
    subgraph Uygulama_Dugumu["Uygulama Düğümü"]
        subgraph Docker_Engine["Docker Motoru"]
            subgraph FastAPI_Container["FastAPI Konteyneri"]
                UVICORN["Uvicorn ASGI (4 İşçi)"]
                FASTAPI["FastAPI Uygulaması"]
                
                subgraph Model_Katmani["Model Katmanı"]
                    BINARY["Binary CNN (ecgcnn.pt, 145KB)"]
                    SUPER["Superclass CNN (ecgcnn_superclass.pt, 433KB)"]
                    LOC["Lokalizasyon CNN (ecgcnn_localization.pt, 433KB)"]
                end
                
                subgraph XGBoost_Katmani["XGBoost Katmanı"]
                    XGB_MI["XGBoost MI (model.json)"]
                    XGB_STTC["XGBoost STTC (model.json)"]
                    XGB_CD["XGBoost CD (model.json)"]
                    XGB_HYP["XGBoost HYP (model.json)"]
                    SCALER["StandardScaler (scaler.joblib)"]
                    CALIB["Kalibratörler (*.joblib)"]
                end
                
                subgraph XAI_Katmani["XAI Katmanı"]
                    GRADCAM["Grad-CAM"]
                    SHAP["SHAP TreeExplainer"]
                end
            end
        end
        
        UVICORN --> FASTAPI
        FASTAPI --> Model_Katmani
        FASTAPI --> XGBoost_Katmani
        FASTAPI --> XAI_Katmani
    end
```

### 2.2 Sunucu Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| İşlemci (CPU) | 4 çekirdek | 8 çekirdek |
| Bellek (RAM) | 8 GB | 16 GB |
| Disk | 50 GB SSD | 100 GB NVMe |
| Grafik İşlemci (GPU) | Gerekli değil | NVIDIA T4 (opsiyonel) |
| Ağ | 100 Mbps | 1 Gbps |

---

## 3. Dosya Sistemi Yapısı

```mermaid
graph TB
    subgraph Host["Ana Makine Dosya Sistemi"]
        ROOT["/opt/cardioguard-ai"]
        
        subgraph App["app/"]
            SRC["src/"]
            TESTS["tests/"]
            REQUIREMENTS["requirements.txt"]
        end
        
        subgraph Data["data/"]
            PTBXL_DIR["ptbxl/ (salt okunur)"]
            FEATURES["features_out/"]
        end
        
        subgraph Models["checkpoints/"]
            ECGCNN["ecgcnn.pt"]
            ECGCNN_SUPER["ecgcnn_superclass.pt"]
            ECGCNN_LOC["ecgcnn_localization.pt"]
        end
        
        subgraph XGB_DIR["logs/xgb_superclass/"]
            MI_DIR["MI/"]
            STTC_DIR["STTC/"]
            CD_DIR["CD/"]
            HYP_DIR["HYP/"]
            SCALER_FILE["scaler.joblib"]
            SCHEMA["feature_schema.json"]
        end
        
        subgraph Outputs["outputs/"]
            ARTIFACTS_DIR["artifacts/"]
            REPORTS_DIR["reports/"]
            LOGS_DIR["logs/"]
        end
    end
    
    ROOT --> App
    ROOT --> Data
    ROOT --> Models
    ROOT --> XGB_DIR
    ROOT --> Outputs
```

### 3.1 Birim Eşlemesi (Docker)

| Ana Makine Yolu | Konteyner Yolu | Erişim Modu |
|-----------------|----------------|-------------|
| ./checkpoints | /app/checkpoints | Salt okunur |
| ./logs/xgb_superclass | /app/logs/xgb_superclass | Salt okunur |
| ./artifacts | /app/artifacts | Okuma/Yazma |
| ./data/ptbxl | /app/data/ptbxl | Salt okunur |
| ./outputs/reports | /app/reports | Okuma/Yazma |
| ./outputs/logs | /app/logs | Okuma/Yazma |

---

## 4. Ağ Mimarisi

```mermaid
graph LR
    subgraph Harici_Ag["Harici Ağ"]
        CLIENT["İstemci (Dinamik IP)"]
    end
    
    subgraph DMZ["Silahsızlandırılmış Bölge"]
        NGINX["NGINX (:80/:443, IP: 10.0.1.10)"]
    end
    
    subgraph Ic_Ag["İç Ağ"]
        API1["FastAPI 1 (:8000, IP: 10.0.2.10)"]
        API2["FastAPI 2 (:8001, IP: 10.0.2.11)"]
    end
    
    subgraph Depolama_Agi["Depolama Ağı"]
        NFS["NFS Paylaşımı (IP: 10.0.3.10)"]
    end
    
    CLIENT -->|HTTPS| NGINX
    NGINX -->|HTTP| API1
    NGINX -->|HTTP| API2
    API1 -->|NFS| NFS
    API2 -->|NFS| NFS
```

### 4.1 Port Tahsisi

| Servis | Port | Protokol | Erişim |
|--------|------|----------|--------|
| NGINX (HTTP) | 80 | TCP | Genel |
| NGINX (HTTPS) | 443 | TCP | Genel |
| FastAPI (Birincil) | 8000 | TCP | İç |
| FastAPI (İkincil) | 8001 | TCP | İç |
| Prometheus Metrikleri | 9090 | TCP | İç |
| Sağlık Kontrolü | 8000/health | HTTP | İç |

---

## 5. Docker Dağıtımı

### 5.1 Dockerfile

```dockerfile
# Temel imaj
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimler
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodu
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY logs/xgb_superclass/ ./logs/xgb_superclass/
COPY artifacts/ ./artifacts/

# Ortam değişkenleri
ENV PYTHONPATH=/app
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Port
EXPOSE 8000

# Başlatma komutu
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 5.2 Docker Compose Yapılandırması

```yaml
version: '3.8'

services:
  cardioguard-api:
    build: .
    image: cardioguard-ai:latest
    container_name: cardioguard-api
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./logs/xgb_superclass:/app/logs/xgb_superclass:ro
      - ./artifacts:/app/artifacts:rw
      - ./reports:/app/reports:rw
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  nginx:
    image: nginx:alpine
    container_name: cardioguard-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - cardioguard-api
    restart: unless-stopped
```

---

## 6. Geliştirme Ortamı

```mermaid
graph TB
    subgraph Gelistirici_Makinesi["Geliştirici Makinesi"]
        subgraph IDE["Entegre Geliştirme Ortamı"]
            CODE["Kaynak Kod"]
            DEBUG["Hata Ayıklayıcı"]
        end
        
        subgraph Yerel_Python["Python 3.10 Sanal Ortam"]
            DEPS["Bağımlılıklar"]
            PYTEST["pytest"]
        end
        
        subgraph Yerel_Veri["Yerel Veri"]
            MINI_PTBXL["Mini PTB-XL (100 örnek)"]
            TEST_CHECKPOINTS["Test Kontrol Noktaları"]
        end
    end
    
    subgraph Yerel_Sunucu["Yerel Sunucu"]
        DEV_API["FastAPI (:8000, reload=True)"]
    end
    
    CODE --> Yerel_Python --> DEV_API
    MINI_PTBXL --> DEV_API
    TEST_CHECKPOINTS --> DEV_API
```

### 6.1 Geliştirme Komutları

| Komut | Açıklama |
|-------|----------|
| `python -m venv .venv` | Sanal ortam oluşturma |
| `.\.venv\Scripts\Activate.ps1` | Sanal ortamı etkinleştirme (Windows) |
| `pip install -r requirements.txt` | Bağımlılıkları yükleme |
| `python -m uvicorn src.backend.main:app --reload --port 8000` | Geliştirme sunucusu başlatma |
| `python -m pytest tests/ -v` | Testleri çalıştırma |
| `python -m flake8 src/` | Kod kalitesi kontrolü |

---

## 7. Üretim Ortamı

```mermaid
graph TB
    subgraph Bulut_Saglayici["Bulut Sağlayıcı (AWS/GCP/Azure)"]
        subgraph Bolge["Bölge: eu-west-1"]
            subgraph VPC["Sanal Özel Bulut"]
                subgraph Genel_Alt_Ag["Genel Alt Ağ"]
                    ALB["Uygulama Yük Dengeleyicisi"]
                end
                
                subgraph Ozel_Alt_Ag["Özel Alt Ağ"]
                    subgraph ECS["ECS Kümesi"]
                        TASK1["Görev 1: FastAPI"]
                        TASK2["Görev 2: FastAPI"]
                    end
                end
                
                subgraph Depolama_Alt_Ag["Depolama Alt Ağı"]
                    EFS["EFS (Kontrol Noktaları)"]
                    S3["S3 (Yapıtlar)"]
                end
            end
        end
        
        subgraph Izleme["İzleme"]
            CW["CloudWatch"]
            XRAY["X-Ray"]
        end
    end
    
    ALB --> TASK1
    ALB --> TASK2
    TASK1 --> EFS
    TASK1 --> S3
    TASK2 --> EFS
    TASK2 --> S3
    TASK1 -.-> CW
    TASK1 -.-> XRAY
    TASK2 -.-> CW
    TASK2 -.-> XRAY
```

### 7.1 Üretim Yapılandırması

| Bileşen | Servis | Boyut |
|---------|--------|-------|
| Hesaplama | ECS Fargate | 4 vCPU, 8GB RAM |
| Yük Dengeleyici | ALB | Otomatik ölçeklendirme |
| Dosya Depolama | EFS | 50 GB |
| Nesne Depolama | S3 | 100 GB |
| Günlükleme | CloudWatch | 30 gün saklama |
| İzleme | X-Ray | Dağıtık izleme |

---

## 8. CI/CD Pipeline

```mermaid
graph LR
    subgraph Kaynak["Kaynak Kontrol"]
        GIT["GitHub Deposu"]
    end
    
    subgraph CI["Sürekli Entegrasyon"]
        LINT["Lint ve Format"]
        TEST["Birim Testleri"]
        BUILD["Docker Derleme"]
        SCAN["Güvenlik Taraması"]
    end
    
    subgraph Registry["Konteyner Kayıt Defteri"]
        ECR["ECR / Docker Hub"]
    end
    
    subgraph CD["Sürekli Dağıtım"]
        STAGING["Hazırlık Dağıtımı"]
        PROD["Üretim Dağıtımı"]
    end
    
    GIT -->|push| LINT --> TEST --> BUILD --> SCAN --> ECR
    ECR -->|otomatik| STAGING
    STAGING -->|manuel onay| PROD
```

### 8.1 Pipeline Aşamaları

| Aşama | Açıklama | Tetikleyici |
|-------|----------|-------------|
| Lint | Kod formatı ve stil kontrolü | Her push |
| Test | Birim ve entegrasyon testleri | Her push |
| Build | Docker imajı derleme | Test başarılı |
| Scan | Güvenlik açığı taraması | Build başarılı |
| Staging | Hazırlık ortamına dağıtım | Tarama başarılı |
| Production | Üretim ortamına dağıtım | Manuel onay |

---

## 9. Güvenlik Mimarisi

```mermaid
graph TB
    subgraph Guvenlik["Güvenlik Katmanları"]
        subgraph Ag["Ağ Güvenliği"]
            WAF["Web Uygulama Güvenlik Duvarı"]
            SG["Güvenlik Grupları"]
            NACL["Ağ Erişim Kontrol Listeleri"]
        end
        
        subgraph Uygulama["Uygulama Güvenliği"]
            TLS["TLS 1.3"]
            CORS["CORS Politikası"]
            RATE["Hız Sınırlama"]
        end
        
        subgraph Veri["Veri Güvenliği"]
            ENCRYPT["Durağan Şifreleme"]
            IAM["Kimlik ve Erişim Yönetimi"]
            SECRETS["Gizli Anahtar Yöneticisi"]
        end
    end
    
    WAF --> SG --> NACL
    TLS --> CORS --> RATE
    ENCRYPT --> IAM --> SECRETS
```

### 9.1 Güvenlik Kontrolleri

| Katman | Kontrol | Uygulama |
|--------|---------|----------|
| Ağ | Güvenlik Duvarı | WAF + Güvenlik Grupları |
| İletim | Şifreleme | TLS 1.3 |
| Uygulama | Kimlik Doğrulama | API Anahtarı / JWT (planlanan) |
| Veri | Şifreleme | AES-256 (durağan) |
| Erişim | Yetkilendirme | IAM Rolleri |

---

## 10. İzleme ve Günlükleme

```mermaid
graph TB
    subgraph Uygulama["Uygulama"]
        API["FastAPI"]
    end
    
    subgraph Metrikler
        PROM["Prometheus"]
        GRAFANA["Grafana Kontrol Paneli"]
    end
    
    subgraph Gunlukleme["Günlükleme"]
        FLUENT["Fluentd"]
        ES["Elasticsearch"]
        KIBANA["Kibana"]
    end
    
    subgraph Uyarilar["Uyarılar"]
        ALERT["AlertManager"]
        SLACK["Slack"]
        EMAIL["E-posta"]
    end
    
    API -->|metrikler| PROM --> GRAFANA
    API -->|günlükler| FLUENT --> ES --> KIBANA
    PROM --> ALERT --> SLACK
    ALERT --> EMAIL
```

### 10.1 Önemli Metrikler

| Metrik | Açıklama | Eşik |
|--------|----------|------|
| request_latency_seconds | API yanıt süresi | < 500 ms |
| request_count | Toplam istek sayısı | İzleme |
| error_rate | Hata oranı | < %1 |
| model_load_time_seconds | Model yükleme süresi | < 5 s |
| prediction_confidence | Tahmin güven skoru | İzleme |
| memory_usage_bytes | Bellek kullanımı | < %80 |
| cpu_usage_percent | İşlemci kullanımı | < %70 |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Sistem Mimarı | | | |
| DevOps Mühendisi | | | |
| Güvenlik Mühendisi | | | |

---

**Doküman Sonu**

*Bu dağıtım diyagramı hem yerel geliştirme hem de bulut tabanlı üretim ortamlarını kapsamaktadır. Gerçek dağıtım, organizasyonun güvenlik gereksinimleri ve bütçesine göre özelleştirilmelidir.*
