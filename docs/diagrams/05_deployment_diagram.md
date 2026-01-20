# CardioGuard-AI: Deployment Diyagramı
## (Dağıtım Mimarisi)

---

## 📋 Doküman Bilgileri

| Özellik | Değer |
|---------|-------|
| **Proje Adı** | CardioGuard-AI |
| **Doküman Tipi** | Deployment Diyagramı |
| **Versiyon** | 1.0.0 |
| **Tarih** | 2026-01-21 |

---

## 1. Genel Mimari

```mermaid
graph TB
    subgraph Internet["🌐 Internet"]
        CLIENT["İstemci<br/>Web Browser / API Client"]
    end
    
    subgraph LoadBalancer["⚖️ Yük Dengeleyici"]
        NGINX["NGINX<br/>Reverse Proxy<br/>:80/:443"]
    end
    
    subgraph ApplicationServer["🖥️ Uygulama Sunucusu"]
        subgraph Container1["Docker Container 1"]
            FASTAPI1["FastAPI<br/>:8000"]
            MODELS1["PyTorch Models"]
            XGB1["XGBoost"]
        end
        
        subgraph Container2["Docker Container 2"]
            FASTAPI2["FastAPI<br/>:8001"]
            MODELS2["PyTorch Models"]
            XGB2["XGBoost"]
        end
    end
    
    subgraph Storage["💾 Depolama"]
        CHECKPOINTS["Checkpoints<br/>Volume"]
        ARTIFACTS["Artifacts<br/>Volume"]
        LOGS["Logs<br/>Volume"]
    end
    
    subgraph DataLayer["📊 Veri Katmanı"]
        PTBXL["PTB-XL Dataset<br/>Read-Only Volume"]
    end
    
    CLIENT --> NGINX
    NGINX --> FASTAPI1 & FASTAPI2
    FASTAPI1 & FASTAPI2 --> CHECKPOINTS & ARTIFACTS & LOGS
    FASTAPI1 & FASTAPI2 --> PTBXL
    
    MODELS1 -.-> Container1
    MODELS2 -.-> Container2
    XGB1 -.-> Container1
    XGB2 -.-> Container2
    
    style NGINX fill:#e8f5e9
    style FASTAPI1 fill:#e3f2fd
    style FASTAPI2 fill:#e3f2fd
    style CHECKPOINTS fill:#fff3e0
    style PTBXL fill:#f3e5f5
```

---

## 2. Bileşen Detayları

### 2.1 Uygulama Sunucusu

```mermaid
graph TB
    subgraph ApplicationNode["🖥️ Uygulama Sunucusu"]
        subgraph DockerEngine["Docker Engine"]
            subgraph FastAPIContainer["FastAPI Container"]
                UVICORN["Uvicorn ASGI<br/>Workers: 4"]
                FASTAPI["FastAPI App"]
                
                subgraph ModelLayer["Model Katmanı"]
                    BINARY["Binary CNN<br/>ecgcnn.pt<br/>145KB"]
                    SUPER["Superclass CNN<br/>ecgcnn_superclass.pt<br/>433KB"]
                    LOC["Localization CNN<br/>ecgcnn_localization.pt<br/>433KB"]
                end
                
                subgraph XGBLayer["XGBoost Katmanı"]
                    XGB_MI["XGB MI<br/>model.json"]
                    XGB_STTC["XGB STTC<br/>model.json"]
                    XGB_CD["XGB CD<br/>model.json"]
                    XGB_HYP["XGB HYP<br/>model.json"]
                    SCALER["StandardScaler<br/>scaler.joblib"]
                    CALIB["Calibrators<br/>*.joblib"]
                end
                
                subgraph XAILayer["XAI Katmanı"]
                    GRADCAM["Grad-CAM"]
                    SHAP["SHAP TreeExplainer"]
                end
            end
        end
        
        UVICORN --> FASTAPI
        FASTAPI --> ModelLayer & XGBLayer & XAILayer
    end
    
    style BINARY fill:#bbdefb
    style SUPER fill:#bbdefb
    style LOC fill:#bbdefb
    style XGB_MI fill:#c8e6c9
    style GRADCAM fill:#fff9c4
```

### 2.2 Sunucu Gereksinimleri

| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| **CPU** | 4 core | 8 core |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 50 GB SSD | 100 GB NVMe |
| **GPU** | - (CPU-only) | NVIDIA T4 (opsiyonel) |
| **Ağ** | 100 Mbps | 1 Gbps |

---

## 3. Dosya Sistemi Yapısı

```mermaid
graph TB
    subgraph Host["🖥️ Host Dosya Sistemi"]
        ROOT["/opt/cardioguard-ai"]
        
        subgraph App["app/"]
            SRC["src/"]
            TESTS["tests/"]
            REQUIREMENTS["requirements.txt"]
        end
        
        subgraph Data["data/"]
            PTBXL_DIR["ptbxl/<br/>(read-only)"]
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
    
    ROOT --> App & Data & Models & XGB_DIR & Outputs
    
    style ROOT fill:#e3f2fd
    style PTBXL_DIR fill:#f3e5f5
    style ECGCNN fill:#fff3e0
    style ECGCNN_SUPER fill:#fff3e0
    style ECGCNN_LOC fill:#fff3e0
```

### 3.1 Volume Mapping (Docker)

```yaml
volumes:
  - ./checkpoints:/app/checkpoints:ro
  - ./logs/xgb_superclass:/app/logs/xgb_superclass:ro
  - ./artifacts:/app/artifacts:rw
  - ./data/ptbxl:/app/data/ptbxl:ro
  - ./outputs/reports:/app/reports:rw
  - ./outputs/logs:/app/logs:rw
```

---

## 4. Ağ Mimarisi

```mermaid
graph LR
    subgraph External["Harici Ağ"]
        CLIENT["İstemci<br/>IP: Dinamik"]
    end
    
    subgraph DMZ["DMZ"]
        NGINX["NGINX<br/>:80/:443<br/>IP: 10.0.1.10"]
    end
    
    subgraph Internal["İç Ağ"]
        API1["FastAPI 1<br/>:8000<br/>IP: 10.0.2.10"]
        API2["FastAPI 2<br/>:8001<br/>IP: 10.0.2.11"]
    end
    
    subgraph Storage["Depolama Ağı"]
        NFS["NFS Share<br/>IP: 10.0.3.10"]
    end
    
    CLIENT -->|HTTPS| NGINX
    NGINX -->|HTTP| API1 & API2
    API1 & API2 -->|NFS| NFS
    
    style NGINX fill:#c8e6c9
    style API1 fill:#bbdefb
    style API2 fill:#bbdefb
```

### 4.1 Port Allocation

| Servis | Port | Protokol | Erişim |
|--------|------|----------|--------|
| NGINX (HTTP) | 80 | TCP | Public |
| NGINX (HTTPS) | 443 | TCP | Public |
| FastAPI (Primary) | 8000 | TCP | Internal |
| FastAPI (Secondary) | 8001 | TCP | Internal |
| Prometheus Metrics | 9090 | TCP | Internal |
| Health Check | 8000/health | HTTP | Internal |

---

## 5. Docker Deployment

### 5.1 Dockerfile

```dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY logs/xgb_superclass/ ./logs/xgb_superclass/
COPY artifacts/ ./artifacts/

# Environment variables
ENV PYTHONPATH=/app
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 5.2 Docker Compose

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

volumes:
  checkpoints:
  artifacts:
  reports:
```

---

## 6. Geliştirme Ortamı

```mermaid
graph TB
    subgraph DevMachine["💻 Geliştirici Makinesi"]
        subgraph IDE["VS Code / PyCharm"]
            CODE["Kaynak Kod"]
            DEBUG["Debugger"]
        end
        
        subgraph LocalPython["Python 3.10 venv"]
            DEPS["Dependencies"]
            PYTEST["pytest"]
        end
        
        subgraph LocalData["Yerel Veri"]
            MINI_PTBXL["Mini PTB-XL<br/>(100 örnek)"]
            TEST_CHECKPOINTS["Test Checkpoints"]
        end
    end
    
    subgraph LocalServer["Yerel Sunucu"]
        DEV_API["FastAPI<br/>:8000<br/>(reload=True)"]
    end
    
    CODE --> LocalPython --> DEV_API
    MINI_PTBXL & TEST_CHECKPOINTS --> DEV_API
    
    style DevMachine fill:#e8f5e9
    style DEV_API fill:#e3f2fd
```

### 6.1 Geliştirme Komutları

```powershell
# Sanal ortam oluşturma
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Bağımlılıkları yükleme
pip install -r requirements.txt

# Geliştirme sunucusu başlatma
python -m uvicorn src.backend.main:app --reload --port 8000

# Testleri çalıştırma
python -m pytest tests/ -v

# Kod kalitesi kontrolü
python -m flake8 src/
python -m mypy src/
```

---

## 7. Üretim Ortamı

```mermaid
graph TB
    subgraph CloudProvider["☁️ Cloud Provider (AWS/GCP/Azure)"]
        subgraph Region["Region: eu-west-1"]
            subgraph VPC["VPC"]
                subgraph PublicSubnet["Public Subnet"]
                    ALB["Application<br/>Load Balancer"]
                end
                
                subgraph PrivateSubnet["Private Subnet"]
                    subgraph ECS["ECS Cluster"]
                        TASK1["Task 1<br/>FastAPI"]
                        TASK2["Task 2<br/>FastAPI"]
                    end
                end
                
                subgraph StorageSubnet["Storage Subnet"]
                    EFS["EFS<br/>Checkpoints"]
                    S3["S3<br/>Artifacts"]
                end
            end
        end
        
        subgraph Monitoring["Monitoring"]
            CW["CloudWatch"]
            XRAY["X-Ray"]
        end
    end
    
    ALB --> TASK1 & TASK2
    TASK1 & TASK2 --> EFS & S3
    TASK1 & TASK2 -.-> CW & XRAY
    
    style ALB fill:#ff9800
    style TASK1 fill:#2196f3
    style TASK2 fill:#2196f3
    style EFS fill:#8bc34a
    style S3 fill:#8bc34a
```

### 7.1 Üretim Konfigürasyonu

| Bileşen | Servis | Boyut |
|---------|--------|-------|
| **Compute** | ECS Fargate | 4 vCPU, 8GB RAM |
| **Load Balancer** | ALB | - |
| **Storage** | EFS | 50 GB |
| **Object Storage** | S3 | 100 GB |
| **Logging** | CloudWatch | 30 gün retention |
| **Monitoring** | X-Ray | - |

---

## 8. CI/CD Pipeline

```mermaid
graph LR
    subgraph Source["Kaynak Kontrol"]
        GIT["GitHub Repository"]
    end
    
    subgraph CI["Continuous Integration"]
        LINT["Lint & Format"]
        TEST["Unit Tests"]
        BUILD["Docker Build"]
        SCAN["Security Scan"]
    end
    
    subgraph Registry["Container Registry"]
        ECR["ECR / Docker Hub"]
    end
    
    subgraph CD["Continuous Deployment"]
        STAGING["Staging Deploy"]
        PROD["Production Deploy"]
    end
    
    GIT -->|push| LINT --> TEST --> BUILD --> SCAN --> ECR
    ECR -->|auto| STAGING
    STAGING -->|manual approval| PROD
    
    style GIT fill:#24292e,color:#fff
    style TEST fill:#4caf50
    style BUILD fill:#2196f3
    style STAGING fill:#ff9800
    style PROD fill:#f44336
```

### 8.1 GitHub Actions Workflow

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ -v --cov=src

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t cardioguard-ai:${{ github.sha }} .
      - name: Push to ECR
        run: |
          aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REGISTRY
          docker push $ECR_REGISTRY/cardioguard-ai:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to ECS Staging
        run: aws ecs update-service --cluster staging --service cardioguard --force-new-deployment

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to ECS Production
        run: aws ecs update-service --cluster production --service cardioguard --force-new-deployment
```

---

## 9. Güvenlik Mimarisi

```mermaid
graph TB
    subgraph Security["🔒 Güvenlik Katmanları"]
        subgraph Network["Ağ Güvenliği"]
            WAF["WAF<br/>Web Application Firewall"]
            SG["Security Groups"]
            NACL["Network ACLs"]
        end
        
        subgraph Application["Uygulama Güvenliği"]
            TLS["TLS 1.3"]
            CORS["CORS Policy"]
            RATE["Rate Limiting"]
        end
        
        subgraph Data["Veri Güvenliği"]
            ENCRYPT["Encryption at Rest"]
            IAM["IAM Roles"]
            SECRETS["Secrets Manager"]
        end
    end
    
    WAF --> SG --> NACL
    TLS --> CORS --> RATE
    ENCRYPT --> IAM --> SECRETS
    
    style WAF fill:#ffcdd2
    style TLS fill:#c8e6c9
    style ENCRYPT fill:#bbdefb
```

### 9.1 Güvenlik Kontrolleri

| Katman | Kontrol | Uygulama |
|--------|---------|----------|
| **Ağ** | Firewall | WAF + Security Groups |
| **Transport** | Şifreleme | TLS 1.3 |
| **Uygulama** | Kimlik Doğrulama | API Key / JWT (gelecek) |
| **Veri** | Şifreleme | AES-256 at rest |
| **Erişim** | Yetkilendirme | IAM Roles |

---

## 10. İzleme ve Loglama

```mermaid
graph TB
    subgraph Application["Uygulama"]
        API["FastAPI"]
    end
    
    subgraph Metrics["Metrikler"]
        PROM["Prometheus"]
        GRAFANA["Grafana<br/>Dashboard"]
    end
    
    subgraph Logging["Loglama"]
        FLUENT["Fluentd"]
        ES["Elasticsearch"]
        KIBANA["Kibana"]
    end
    
    subgraph Alerting["Uyarılar"]
        ALERT["AlertManager"]
        SLACK["Slack"]
        EMAIL["Email"]
    end
    
    API -->|metrics| PROM --> GRAFANA
    API -->|logs| FLUENT --> ES --> KIBANA
    PROM --> ALERT --> SLACK & EMAIL
    
    style GRAFANA fill:#ff9800
    style KIBANA fill:#4caf50
```

### 10.1 Önemli Metrikler

| Metrik | Açıklama | Eşik |
|--------|----------|------|
| `request_latency_seconds` | API yanıt süresi | < 500ms |
| `request_count` | Toplam istek sayısı | - |
| `error_rate` | Hata oranı | < 1% |
| `model_load_time_seconds` | Model yükleme süresi | < 5s |
| `prediction_confidence` | Tahmin güveni | - |
| `memory_usage_bytes` | Bellek kullanımı | < 80% |
| `cpu_usage_percent` | CPU kullanımı | < 70% |

---

## 11. Yedekleme ve Kurtarma

```mermaid
graph LR
    subgraph Primary["Birincil Sistem"]
        PROD["Production"]
    end
    
    subgraph Backup["Yedekleme"]
        S3["S3 Backup<br/>Daily"]
        GLACIER["Glacier<br/>Monthly"]
    end
    
    subgraph DR["Disaster Recovery"]
        DR_REGION["DR Region<br/>eu-central-1"]
    end
    
    PROD -->|daily| S3
    S3 -->|monthly| GLACIER
    PROD -.->|replication| DR_REGION
    
    style PROD fill:#4caf50
    style S3 fill:#2196f3
    style DR_REGION fill:#ff9800
```

### 11.1 Yedekleme Politikası

| Bileşen | Sıklık | Saklama Süresi | Konum |
|---------|--------|----------------|-------|
| Checkpoints | Her değişiklikte | Süresiz | S3 + Glacier |
| Konfigürasyon | Günlük | 90 gün | S3 |
| Loglar | Günlük | 30 gün | CloudWatch |
| XAI Artifacts | Her tahmin | 7 gün | S3 |

---

> **Not:** Bu deployment diyagramı hem yerel geliştirme hem de cloud-based üretim ortamlarını kapsamaktadır. Gerçek deployment, organizasyonun güvenlik gereksinimleri ve bütçesine göre özelleştirilmelidir.
