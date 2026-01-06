# CardioGuard-AI: Holistik Sistem Analizi (Deep-6 Protokolü)

**Dil:** Türkçe  
**Rol:** Lead AI Architect & Principal Investigator  
**Kapsam:** Kod tabanı + akademik raporların sentezi  

---

## 1. Mimari Eleştiri (Hybrid CNN + XGBoost)

### 1.1 Neden Uçtan Uca CNN değil de Hibrit Mimari?
CardioGuard-AI, **temsil öğrenme** (CNN) ile **tablosal karar verme**yi (XGBoost) ayrıştıran bir hibrit yapı kullanır. Bu kararın teknik gerekçesi hem veri boyutu hem de klinik karar mantığıdır:

- **PTB-XL veri ölçeği (~21k örnek)** uçtan uca CNN için yeterli görünse de, sınıf dengesizliği (MI azınlık) ve klinik kararların “if-then” yapısı, saf softmax kafasının genelleştirme gücünü sınırlayabilir. Bu nedenle CNN yalnızca yüksek boyutlu sinyal temsili üretir, sınıflandırma ise karar ağaçlarıyla yapılır (bkz. `src/models/xgb.py`, `XGBConfig`).
- **CNN**; QRS kompleksi, ST segmenti ve T dalgası gibi **zamansal morfolojileri** yakalayan güçlü bir desen tanıyıcıdır (`src/models/cnn.py`, `ECGBackbone`).
- **XGBoost** ise **tablosal embedding** üzerinde “karar verme”de daha kararlıdır: `scale_pos_weight`, `max_depth=4`, `subsample=0.8` gibi ayarlar küçük/orta ölçekli dengesiz veri için daha güvenli davranır (`src/models/xgb.py`, `XGBConfig`).

### 1.2 İki Fazlı Eğitim (Backbone Freeze → XGBoost)
Kod ve raporlar, eğitim stratejisini iki fazda tanımlar:

1. **Faz 1 – Temsil Öğrenme:** CNN omurgası, görev başlığı ile eğitilip “sinyali en iyi özetleyen” embedding öğrenir (rapor: *Derinlemesine Teknik Model Raporu*).  
2. **Faz 2 – Karar Mekanizması:** CNN dondurulur (`requires_grad=False` mantığı raporda); embedding matrisleri ile XGBoost eğitilir (`XGBClassifier.fit`).

Bu ayrım, **overfitting riskini azaltır** ve küçük veri kümelerinde **kararlı bir karar sınırı** üretir. Ayrıca CNN eğitimi bittiğinde XGB hiperparametreleriyle esnekçe oynanabilir.

---

## 2. Sinyal İşleme ve Özellik Çıkarımı

### 2.1 ECGCNN Omurgası ve Kernel=7 Kararı
`src/models/cnn.py` dosyasında yer alan `ECGBackbone` iki ardışık `Conv1d` bloğuyla başlar. **Kernel size=7**, stride varsayılan olarak **1**’dir. 100Hz örnekleme hızında 7 örnek **0.07 sn**’ye karşılık gelir:

- Tipik **QRS süresi 0.06–0.10 sn** aralığındadır. Bu nedenle 7 örnekli çekirdek, QRS’in hızlı yükselme/düşüş morfolojisini yakalamak için fizyolojik olarak tutarlıdır.
- `padding = kernel_size // 2` ile uzunluk korunur; bu, QRS’nin zaman ekseni boyunca kaymasına rağmen özellik yakalanmasını kolaylaştırır.

### 2.2 Kanal Bazlı Z-Score Normalizasyonu
`src/data/signals.py` içinde `normalize_signal` ve `normalize_with_stats` fonksiyonları kanal bazlı Z-score uygular. Bu işlem, 12 derivasyon arasında voltaj ölçeklerinin farklı olmasının öğrenmeyi bozmasını engeller.

- **Kanal bağımsız normalize etmek**, özellikle **lead-agnostic** temsil öğrenmede kritiktir. Aksi halde model V1/V6 gibi genlik olarak baskın derivasyonlara aşırı uyum sağlar.
- Eğitim pipeline’ında kanal istatistikleri (`compute_channel_stats_streaming`) eğitim setinden hesaplanarak normalize edilir; bu, veri sızıntısını önler (bkz. `src/data/signals.py`).

---

## 3. XAI Güven Çerçevesi

### 3.1 Combined Explainer Mantığı
`src/xai/combined.py` içindeki `CombinedExplainer`, **SHAP (semantik)** ve **Grad-CAM (uzamsal/zamansal)** açıklamalarını tek raporda birleştirir.

- **SHAP:** XGBoost’un embedding boyutları üzerindeki karar katkısını verir.  
- **Grad-CAM:** CNN’in EKG zaman ekseni üzerindeki odak bölgelerini gösterir.
- **Birleşim:** SHAP ile önem kazanan embedding boyutları, Grad-CAM haritasına ağırlık verir. Böylece **“hangi zaman bölgeleri, hangi semantik öznitelikleri tetikledi”** sorusu cevaplanır.

Bu tasarım, **Interpretability Gap**’i kapatır: tek başına Grad-CAM “nereye bakıldığını”, SHAP ise “hangi özniteliklerin etkili olduğunu” söyler; birleşik yapı ise klinik yorum için ikisini bağlar.

### 3.2 Sanity Checks: Faithfulness + Randomization
XAI raporları, `src/xai/sanity.py` modülü ile doğrulanır (rapor: *Derinlemesine XAI Raporu*):

- **Faithfulness:** Önemli bölgeler maskelenince skor düşmelidir (raporda Faithfulness=0.82). Bu, açıklamanın model kararına sadık olduğunu gösterir.
- **Cascading Randomization:** Model ağırlıkları rastgeleleştirilince Grad-CAM haritalarının bozulması beklenir. Korelasyonun düşmesi, açıklayıcının “basit edge detector” olmadığını ispatlar.

---

## 4. Klinik Relevans ve Performans Metrikleri

### 4.1 AUC=0.976 ve Accuracy=93.6
Sentez raporunda test seti metrikleri açıkça verilmiştir:

- **Binary Accuracy: %93.6**, **ROC AUC: 0.976**  
- **MI Precision: %91.9**, **MI Recall: %81.9**  

Bu metrikler **tarama (screening)** bağlamında güçlüdür. Özellikle AUC=0.976, ayırt edici gücün yüksek olduğunu gösterir. Ancak **MI Recall %81.9** klinik bir tarama için sınırda değerlendirilebilir; pratikte eşik ayarının (“threshold tuning”) klinik risk toleransına göre optimize edilmesi gerekir.

### 4.2 Sınıf Dengesizliği ve `scale_pos_weight`
`src/models/xgb.py` içinde `XGBConfig.scale_pos_weight`, MI/NORM dengesizliğini telafi etmek için tasarlanmıştır. Bu yaklaşım:

- MI’nin **azınlık** olduğu durumlarda recall’ı artırır.
- Modelin “normal” sınıfına aşırı kaymasını engeller.
- CNN’den gelen embeddinglerin, sınıf ağırlıklarıyla daha dengeli kullanılmasını sağlar.

---

## 5. Teknik Zorluklar ve Çözümler

### 5.1 PyTorch Binary “Scalar Output” Hatası
`src/pipeline/generate_xai_report.py` içinde binary tahminlerde CNN çıktısı bazen **skaler** olabilir. Bu durum `cnn_probs.size == 1` kontrolüyle ele alınır:

- Eğer skaler çıktı geliyorsa **MI olasılığı tek değer olarak alınır** ve **NORM=1−MI** olarak hesaplanır.
- Bu düzeltme, **tek-nöronlu binary head** kullanan modellerin (bkz. `BinaryHead` ve `squeeze(-1)` mantığı) downstream XAI raporlamayı kırmasını engeller.

### 5.2 SHAP Wrapper Unwrapping (Calibrated Model)
SHAP, `CalibratedClassifierCV` veya `ManualCalibratedModel` gibi wrapper’larla doğrudan çalışmaz. Bu yüzden `src/xai/shap_xgb.py` ve `src/xai/combined.py` içinde:

```python
if hasattr(model, "base_model"):
    model = model.base_model
```

şeklinde **unwrap** işlemi yapılır. Böylece SHAP yalnızca “saf XGBClassifier” görür ve ağaç yapısı üzerinden doğru Shapley hesaplar.

---

## 6. Gelecek Entegrasyon: LLM-RAG Şeması

### 6.1 JSON Çıktılarının RAG İçin Yapılandırılması
`src/xai/combined.py` içindeki `ExplanationCard` ve `SampleExplanation` yapıları, RAG için doğal bir **ara format** sunar:

Önerilen RAG şeması:

```json
{
  "sample_id": "ecg_000123",
  "task": "binary",
  "prediction": {"class": "MI", "prob": 0.98},
  "gradcam": {
    "vector": [0.02, 0.8, ...],
    "top_windows": [{"start_ms": 120, "end_ms": 200, "score": 0.81}]
  },
  "shap": {
    "top_features": [{"feature_idx": 124, "shap_value": 0.45, "direction": "positive"}],
    "expected_value": 0.15
  },
  "combined": {
    "top_windows": [{"start_ms": 120, "end_ms": 200, "score": 0.91}]
  },
  "sanity": {"faithfulness": 0.82, "randomization_corr": 0.14}
}
```

### 6.2 Grad-CAM Vektörlerinin LLM Tarafından Yorumlanması
`[0.02, 0.8, ...]` gibi Grad-CAM vektörleri, **zamansal dikkat dağılımı** olarak yorumlanmalıdır:

- Vektör **1000 örnek = 10 saniye** sinyalin her zaman dilimine karşılık gelir.
- LLM, bu vektörü **pencerelere bölerek** (ör. 80ms pencereler) “en önemli QRS/ST bölgelerini” çıkarabilir.
- `CombinedExplainer`’ın `window_ms=80` parametresi, bu segmentasyonu zaten klinik zaman aralığına uyumlu hale getirir.

---

## 7. Sonuç

CardioGuard-AI, **CNN’in morfolojik öğrenme gücü** ile **XGBoost’un kararlı karar mekanizmasını** birleştirerek hem performans hem de açıklanabilirlik hedeflerini karşılar. XAI modülü yalnızca görsel değil, **matematiksel olarak doğrulanmış** açıklamalar sunar. Özellikle **Combined Explainer + Sanity Checks** yaklaşımı, klinik güvenilirliği artıran güçlü bir çerçeve sağlar.

Bu rapor, kod tabanındaki gerçek mimari kararlarla (kernel=7, `scale_pos_weight`, Z-score normalizasyonu, wrapper unwrapping vb.) akademik raporların bulgularını tutarlı biçimde birleştirmektedir.
