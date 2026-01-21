# CardioGuard-AI
# Kullanım Senaryoları Dokümanı

---

**Proje Adı:** CardioGuard-AI  
**Doküman Tipi:** Kullanım Senaryoları (Use Case)  
**Versiyon:** 1.0.0  
**Tarih:** 21 Ocak 2026

---

## 1. Kullanım Senaryoları Diyagramı

```mermaid
graph TB
    subgraph Aktorler["Aktörler"]
        A1(["Klinisyen"])
        A2(["ML Mühendisi"])
        A3(["Sistem Yöneticisi"])
    end
    
    subgraph Sistem["CardioGuard-AI"]
        UC1(("UC1: EKG Yükleme"))
        UC2(("UC2: Patoloji Tespiti"))
        UC3(("UC3: MI Lokalizasyonu"))
        UC4(("UC4: Sonuç Raporlama"))
        UC5(("UC5: XAI Görüntüleme"))
        UC6(("UC6: Model Eğitimi"))
        UC7(("UC7: Checkpoint Yönetimi"))
        UC8(("UC8: Sağlık Kontrolü"))
        UC9(("UC9: Triaj Belirleme"))
    end
    
    A1 --> UC1
    A1 --> UC4
    A1 --> UC5
    
    A2 --> UC6
    A2 --> UC7
    
    A3 --> UC7
    A3 --> UC8
    
    UC1 -.->|«include»| UC2
    UC2 -.->|«include»| UC9
    UC2 -.->|«extend»| UC3
    UC2 -.->|«include»| UC4
    UC4 -.->|«extend»| UC5
```

---

## 2. Aktörler

| Aktör | Tip | Açıklama |
|-------|-----|----------|
| Klinisyen | Birincil | EKG analizi talep eder, sonuçları yorumlar |
| ML Mühendisi | Birincil | Model eğitimi ve optimizasyonu yapar |
| Sistem Yöneticisi | Birincil | Sistem izleme ve bakım yapar |

---

## 3. Kullanım Senaryoları Özeti

| ID | Senaryo | Aktör | Öncelik | Durum |
|----|---------|-------|---------|-------|
| UC1 | EKG Sinyali Yükleme | Klinisyen | Yüksek | Tamamlandı |
| UC2 | Patoloji Tespiti | Sistem | Yüksek | Tamamlandı |
| UC3 | MI Lokalizasyonu | Sistem | Yüksek | Tamamlandı |
| UC4 | Sonuç Raporlama | Klinisyen | Yüksek | Tamamlandı |
| UC5 | XAI Görüntüleme | Klinisyen | Orta | Tamamlandı |
| UC6 | Model Eğitimi | ML Mühendisi | Yüksek | Tamamlandı |
| UC7 | Checkpoint Yönetimi | ML Müh./SysAdmin | Yüksek | Tamamlandı |
| UC8 | Sağlık Kontrolü | Sistem Yöneticisi | Orta | Tamamlandı |
| UC9 | Triaj Belirleme | Sistem | Yüksek | Tamamlandı |

---

## 4. Senaryo İlişkileri

| Kaynak | Hedef | İlişki | Açıklama |
|--------|-------|--------|----------|
| UC1 | UC2 | «include» | EKG yükleme, tespiti tetikler |
| UC2 | UC9 | «include» | Tespit, triaj belirler |
| UC2 | UC3 | «extend» | MI varsa lokalizasyon çalışır |
| UC2 | UC4 | «include» | Tespit sonrası raporlama |
| UC4 | UC5 | «extend» | XAI etkinse açıklamalar |

---

## 5. Senaryo Detayları

### UC1: EKG Sinyali Yükleme

| Özellik | Değer |
|---------|-------|
| Ön Koşul | Kullanıcı sisteme bağlı |
| Son Koşul | Sinyal normalize edilmiş |
| Akış | Dosya seçimi → Format kontrolü → Normalizasyon |

### UC2: Patoloji Tespiti

| Özellik | Değer |
|---------|-------|
| Ön Koşul | Sinyal yüklenmiş |
| Son Koşul | Olasılıklar üretilmiş |
| Sınıflar | MI, STTC, CD, HYP |

### UC3: MI Lokalizasyonu

| Özellik | Değer |
|---------|-------|
| Tetikleme | MI tespit edilirse |
| Bölgeler | AMI, ASMI, ALMI, IMI, LMI |

### UC9: Triaj Belirleme

| Tahmin | Triaj Seviyesi |
|--------|----------------|
| MI Tespit | YÜKSEK |
| Diğer Patoloji | ORTA |
| Normal | DÜŞÜK |
| Model Uyumsuzluğu | İNCELEME |

---

## 6. Planlanan Senaryolar (v2.0)

| ID | Senaryo | Açıklama |
|----|---------|----------|
| UC10 | RAG Entegrasyonu | Klinik kılavuzlarla zenginleştirme |
| UC11 | Belirsizlik Tahmini | Monte Carlo Dropout |
| UC12 | LLM Rapor Üretimi | Otomatik klinik rapor |

---

## Onay Sayfası

| Rol | Ad Soyad | Tarih | İmza |
|-----|----------|-------|------|
| Proje Yöneticisi | | | |
| Teknik Lider | | | |

---

**Doküman Sonu**
