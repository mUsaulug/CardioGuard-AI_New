# CardioGuard-AI: Algoritma Adımları
## (Hibrit CNN-XGBoost EKG Sınıflandırma Sistemi)

---

## Algoritma 1: CardioGuard-AI Hibrit Çok-Etiketli EKG Sınıflandırma

```
Algorithm 1: CardioGuard-AI Hybrid Multi-Label ECG Classification

Input:  ECG signal X ∈ ℝ^(12×1000), 12-lead ECG at 100Hz
Output: Prediction Y = {pathology labels, probabilities, triage level}

 1   Load pretrained models: M_super (4-class CNN), M_binary (binary CNN), M_loc (5-class CNN)
 2   Load XGBoost ensemble: {XGB_MI, XGB_STTC, XGB_CD, XGB_HYP}
 3   Load calibrators and scaler from artifacts
 4   Load optimized thresholds θ = {θ_MI, θ_STTC, θ_CD, θ_HYP}
 5   
 6   // Phase 1: Preprocessing
 7   X_norm ← MinMaxNormalize(X) per lead
 8   X_tensor ← ConvertToTensor(X_norm)
 9   
10   // Phase 2: CNN Feature Extraction
11   E ← M_super.backbone(X_tensor)                    // E ∈ ℝ^64 embedding vector
12   Z_super ← M_super.head(E)                         // Z_super ∈ ℝ^4 logits
13   P_cnn ← Sigmoid(Z_super)                          // P_cnn ∈ [0,1]^4
14   
15   // Phase 3: XGBoost Prediction
16   E_scaled ← Scaler.transform(E)
17   for each class c in {MI, STTC, CD, HYP} do
18       P_xgb_raw[c] ← XGB[c].predict_proba(E_scaled)
19       P_xgb[c] ← Calibrator[c].transform(P_xgb_raw[c])
20   end for
21   
22   // Phase 4: Ensemble Combination
23   α ← 0.15                                          // Ensemble weight (optimized)
24   for each class c in {MI, STTC, CD, HYP} do
25       P_ensemble[c] ← α × P_cnn[c] + (1 - α) × P_xgb[c]
26   end for
27   
28   // Phase 5: NORM Derivation (not a classifier output)
29   P_NORM ← 1.0 - max(P_ensemble)
30   
31   // Phase 6: Binary MI Verification (Consistency Guard)
32   Z_binary ← M_binary(X_tensor)
33   P_binary_MI ← Sigmoid(Z_binary)
34   
35   // Consistency Check
36   decision_super ← (P_ensemble[MI] ≥ θ_super)       // θ_super = 0.01 for high recall
37   decision_binary ← (P_binary_MI ≥ 0.5)
38   
39   if decision_super AND decision_binary then
40       agreement ← "AGREE_MI"
41       triage ← "HIGH"
42   else if NOT decision_super AND NOT decision_binary then
43       agreement ← "AGREE_NO_MI"
44       triage ← "LOW"
45   else if decision_super AND NOT decision_binary then
46       agreement ← "DISAGREE_TYPE_1"
47       triage ← "REVIEW"                             // Low confidence MI
48   else
49       agreement ← "DISAGREE_TYPE_2"
50       triage ← "REVIEW"                             // Missed by superclass
51   end if
52   
53   // Phase 7: MI Localization (Gated)
54   if agreement = "AGREE_MI" OR agreement = "DISAGREE_TYPE_1" then
55       Z_loc ← M_loc(X_tensor)
56       P_loc ← Sigmoid(Z_loc)                        // P_loc ∈ [0,1]^5
57       for each region r in {AMI, ASMI, ALMI, IMI, LMI} do
58           detected[r] ← (P_loc[r] ≥ 0.5)
59       end for
60   else
61       P_loc ← null
62       detected ← {}
63   end if
64   
65   // Phase 8: Multi-label Decision
66   Y_labels ← {}
67   for each class c in {MI, STTC, CD, HYP} do
68       if P_ensemble[c] ≥ θ[c] then
69           Y_labels ← Y_labels ∪ {c}
70       end if
71   end for
72   if Y_labels = ∅ then
73       Y_labels ← {NORM}
74   end if
75   
76   // Phase 9: Primary Label Selection (MI-first priority)
77   if MI ∈ Y_labels then
78       primary ← MI
79   else if STTC ∈ Y_labels then
80       primary ← STTC
81   else if CD ∈ Y_labels then
82       primary ← CD
83   else if HYP ∈ Y_labels then
84       primary ← HYP
85   else
86       primary ← NORM
87   end if
88   
89   return Y = {labels, probabilities, primary, triage, consistency, localization}
```

---

## Algoritma 2: EKG Sinyali Ön İşleme

```
Algorithm 2: ECG Signal Preprocessing

Input:  Raw ECG signal X_raw ∈ ℝ^(12×1000)
Output: Normalized signal X_norm ∈ ℝ^(12×1000)

 1   for each lead l in {I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6} do
 2       x_min ← min(X_raw[l, :])
 3       x_max ← max(X_raw[l, :])
 4       if x_max - x_min > ε then                     // ε = 1e-8 avoid division by zero
 5           X_norm[l, :] ← (X_raw[l, :] - x_min) / (x_max - x_min)
 6       else
 7           X_norm[l, :] ← 0
 8       end if
 9   end for
10   return X_norm
```

---

## Algoritma 3: CNN Omurga İleri Geçişi

```
Algorithm 3: ECGBackbone CNN Forward Pass

Input:  Normalized ECG tensor X ∈ ℝ^(B×12×1000), batch size B
Output: Feature embedding E ∈ ℝ^(B×64)

 1   // Layer 1: First Convolution Block
 2   H1 ← Conv1d(X, filters=64, kernel_size=7, padding=3)
 3   H1 ← BatchNorm1d(H1)
 4   H1 ← ReLU(H1)
 5   H1 ← Dropout(H1, p=0.3)
 6   
 7   // Layer 2: Second Convolution Block
 8   H2 ← Conv1d(H1, filters=64, kernel_size=7, padding=3)
 9   H2 ← BatchNorm1d(H2)
10   H2 ← ReLU(H2)
11   H2 ← Dropout(H2, p=0.3)
12   
13   // Global Average Pooling
14   E ← AdaptiveAvgPool1d(H2, output_size=1)
15   E ← Squeeze(E, dim=-1)                           // E ∈ ℝ^(B×64)
16   
17   return E
```

---

## Algoritma 4: MI Lokalizasyon Etiketi Türetimi

```
Algorithm 4: MI Localization Label Derivation from PTB-XL SCP Codes

Input:  SCP codes dictionary {code: likelihood}
Output: Multi-hot label vector y_loc ∈ {0,1}^5 for [AMI, ASMI, ALMI, IMI, LMI]

 1   Initialize y_loc ← [0, 0, 0, 0, 0]
 2   Define mapping M:
 3       M[AMI]   ← {AMI}
 4       M[ASMI]  ← {ASMI, INJAS}
 5       M[ALMI]  ← {ALMI, INJAL}
 6       M[IMI]   ← {IMI, ILMI, IPLMI, IPMI, INJIN, INJIL}
 7       M[LMI]   ← {LMI, ILMI, IPLMI, INJIL, INJLA}
 8   
 9   for each code c in SCP_codes do
10       if likelihood[c] > min_likelihood then       // min_likelihood = 0
11           for each region r in {AMI, ASMI, ALMI, IMI, LMI} do
12               if c ∈ M[r] then
13                   y_loc[r] ← 1
14               end if
15           end for
16       end if
17   end for
18   
19   // Note: PMI (Posterior MI) is excluded from mapping
20   return y_loc
```

---

## Algoritma 5: Kontrol Noktası Doğrulama (Fail-Fast)

```
Algorithm 5: Checkpoint Validation at Startup

Input:  Checkpoint paths {binary, superclass, localization}
Output: Validation result (PASS / FAIL with error)

 1   expected_dims ← {binary: 1, superclass: 4, localization: 5}
 2   
 3   for each task t in {binary, superclass, localization} do
 4       checkpoint ← LoadCheckpoint(path[t])
 5       state_dict ← checkpoint["model_state_dict"]
 6       
 7       // Extract output dimension from classifier weights
 8       if "head.classifier.weight" in state_dict then
 9           out_dim ← shape(state_dict["head.classifier.weight"])[0]
10       else if "head.weight" in state_dict then
11           out_dim ← shape(state_dict["head.weight"])[0]
12       else if "1.classifier.weight" in state_dict then
13           out_dim ← shape(state_dict["1.classifier.weight"])[0]
14       else
15           raise Error("Cannot determine output dimension")
16       end if
17       
18       // Dimension validation
19       if out_dim ≠ expected_dims[t] then
20           raise CheckpointMismatchError(t, expected_dims[t], out_dim)
21       end if
22       
23       // Regression head rejection (for localization)
24       if t = "localization" AND "localization_head.regressor.weight" in state_dict then
25           if shape(state_dict["localization_head.regressor.weight"])[0] = 2 then
26               raise Error("Regression head cannot be used for MI localization")
27           end if
28       end if
29   end for
30   
31   // Mapping fingerprint validation
32   fingerprint ← ComputeMappingFingerprint(MI_CODE_TO_REGIONS)
33   if fingerprint ≠ "8ab274e06afa1be8" then
34       raise MappingDriftError("MI localization mapping has changed")
35   end if
36   
37   return PASS
```

---

## Karmaşıklık Analizi

| Algoritma | Zaman Karmaşıklığı | Uzay Karmaşıklığı |
|-----------|-------------------|-------------------|
| Algoritma 1 (Ana) | O(n) | O(n) |
| Algoritma 2 (Ön İşleme) | O(12 × T) = O(T) | O(T) |
| Algoritma 3 (CNN İleri) | O(C × K × T) | O(C × T) |
| Algoritma 4 (Etiket Türetimi) | O(\|SCP kodları\|) | O(1) |
| Algoritma 5 (Doğrulama) | O(\|kontrol noktaları\|) | O(1) |

n = örnek sayısı, T = sinyal uzunluğu (1000), C = filtre sayısı (64), K = çekirdek boyutu (7)

---

## Temel Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| Girdi boyutu | (12, 1000) | 12 derivasyon × 1000 örnek |
| Örnekleme hızı | 100 Hz | 10 saniyelik kayıt |
| CNN filtreleri | 64 | Öznitelik boyutu |
| Çekirdek boyutu | 7 | Zamansal konvolüsyon |
| Dropout | 0.3 | Düzenlileştirme |
| Topluluk α | 0.15 | CNN-XGB topluluğunda CNN ağırlığı |
| θ_MI (süpersınıf) | 0.01 | Yüksek duyarlılık eşiği |
| θ_binary | 0.5 | İkili MI eşiği |
| Lokalizasyon eşiği | 0.5 | Bölge başına tespit |
