# Engineering Audit: Fab-7 In-Line Wafer Excursion Detector

## 1. Silent Defects Identified & Technical Remediation

### Defect 1: Temporal Data Leakage & Synthetic Balancing Artifacts
* **Finding:** The baseline pipeline performed global median imputation and z-score standard scaling across the entire dataset prior to splitting, followed by SMOTE synthetic oversampling on the evaluation split.
* **Impact:** 
  1. Future sensor distributions ($t' > t$) leaked into the past, violating physical causality.
  2. Testing on an artificially balanced split masked real false-positive excursions, creating an illusion of $F1 \ge 0.90$.
* **Fix:** Enforced strict chronological splitting (first 80% train, subsequent 20% test). Imputation medians and scaling parameters were computed exclusively from the training partition and transformed forward. Evaluation was conducted strictly on the untouched test partition at its natural defect prevalence (~5.41%).

### Defect 2: Corrupted Loss Gradients from Silent Tensor Broadcasting
* **Finding:** Model classification logits were shaped `(N, 1)` or `(N,)` while target labels were structured with conflicting rank during loss calculation.
* **Impact:** PyTorch implicitly broadcast tensor dimensions into an `(N, N)` cross-sample matrix inside `BCEWithLogitsLoss`. The gradient updates were computed across $N^2$ pairwise sample comparisons instead of sample-wise comparisons, corrupting the backpropagation updates and producing false loss curves.
* **Fix:** Enforced explicit tensor alignment using `.view(-1, 1)` and added strict runtime assertions (`assert logits.shape == batch_y.shape` and `assert x_rec.shape == batch_x.shape`) prior to computing `l_recon + 2.0 * l_class`.

### Defect 3: Invariant Blindness on Legacy Probe Station PS-L
* **Finding:** Telemetry from legacy Probe Station PS-L wrote `sensor_0 = 0.0` as a hardware sentinel flag, and stored categorical firmware error codes (`100.0`, `200.0`, `300.0`) in `sensor_1`. The original notebook treated both columns as continuous floating-point measurements.
* **Impact:** Standard scaling over discrete hardware error codes distorted mean/variance estimates and destroyed the discrete diagnostic signal.
* **Fix:** Extracted a dedicated tool indicator `is_ps_l = (sensor_0 == 0.0)`. One-hot encoded the categorical status codes into explicit binary channels (`ps_l_err_100`, `ps_l_err_200`, `ps_l_err_300`), and masked out legacy `sensor_0` and `sensor_1` values with `NaN` so they were excluded from continuous feature normalization.

---

## 2. True Evaluation Metrics at Natural Prevalence (5.41%)

The repaired model was evaluated strictly on the temporally isolated test split (314 wafers: 297 normal, 17 defective):

| Metric | Measured Value |
| :--- | :--- |
| **Natural Defect Prevalence** | **5.41%** (17 / 314 wafers) |
| **Optimal Decision Threshold** | **0.7823** |
| **Defect Class (1) F1 Score** | **0.3077** (~0.31) |
| **PR-AUC (Average Precision)** | **0.2354** |
| **ROC-AUC Score** | **0.7851** |
| **Overall Accuracy** | **0.94** |

### Detailed Classification Breakdown

```text
              precision    recall  f1-score   support

  Normal (0)       0.96      0.98      0.97       297
Defective (1)      0.44      0.24      0.31        17

    accuracy                           0.94       314
   macro avg       0.70      0.61      0.64       314
weighted avg       0.93      0.94      0.93       314