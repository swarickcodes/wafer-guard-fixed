# WaferGuard: Production-Grade Semiconductor Excursion Detector

An audited and repaired machine learning pipeline for semiconductor manufacturing anomaly detection on inline telemetry data (`secom_telemetry_modified.csv`).

---

## 1. Project Overview & Business Context

In high-volume semiconductor fabrication (Fab-7), early detection of defective wafers prevents downstream packaging waste and line excursions. This repository audits and repairs a line-monitoring anomaly detection pipeline:
- **Baseline Defect:** The initial model claimed an artificial $F1 \ge 0.90$ caused by future-to-past data leakage, evaluation on a synthetically balanced test set (SMOTE), and silent tensor broadcasting errors.
- **Remediation:** Enforces strict physical causality, tensor dimension verification, domain-specific telemetry routing for legacy probe stations, and honest evaluation at the natural test defect prevalence of ~5.41%.

---

## 2. System Architecture

The pipeline uses a dual-objective **Supervised Autoencoder (SAE)**:
- **Encoder-Decoder Network:** Maps high-dimensional continuous telemetry features into a compact 32-dimensional latent bottleneck representation and reconstructs them via Mean Squared Error (MSE) loss.
- **Classification Head:** Directly utilizes the regularized latent manifold to classify defective wafers using positive-class weighted Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **Station PS-L Telemetry Router:** Isolates sentinel `0.0` values on `sensor_0` and one-hot encodes firmware error codes (`100.0`, `200.0`, `300.0`) from `sensor_1`, preventing synthetic continuous scaling over hardware flags.

---

## 3. Repository Structure

```text
wafer-guard-audit/
├── .gitignore               # Ignores local environments, data, and checkpoints
├── requirements.txt         # Pinned project dependencies
├── README.md                # System documentation and instructions
├── AUDIT.md                 # Technical audit report and measured metrics
└── wafer_guard_fixed.ipynb  # Fully executed, repaired PyTorch notebook