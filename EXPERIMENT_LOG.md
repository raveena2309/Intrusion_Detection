# Experiment Log

## Intelligent Network Intrusion Detection System

---

## E1 — Random Forest vs XGBoost (Multiclass Baseline)

**Date:** 2026-11-08  
**Phase:** 2  
**Script:** `experiments/train_classifiers.py`  
**Dataset split:** Official UNSW-NB15 train (175,341) / test (82,332)  
**Random seed:** 42  

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train Time |
|-------|----------|----------|-------------|------------|
| Random Forest | 0.6838 | 0.4945 | 0.7363 | 4.1s |
| XGBoost | 0.7636 | 0.5130 | 0.7801 | 19.1s |

**XGBoost per-class F1:** Analysis=0.033, Backdoor=0.053, DoS=0.197, Exploits=0.711, Fuzzers=0.389, Generic=0.984, Normal=0.846, Reconnaissance=0.866, Shellcode=0.516, Worms=0.535  
**Artifacts:** `models/classifier/xgboost_metadata.yaml`, `models/classifier/random_forest_metadata.yaml`  
**Results:** `results/metrics/classifier_results.yaml`  

---

## E3 — Isolation Forest Anomaly Detector

**Date:** 2026-11-08  
**Phase:** 3  
**Script:** `experiments/train_anomaly_detector.py`  
**Training data:** Normal-only records from training set (47,600 normal)  
**Threshold selection:** 5% target FPR on normal-only validation set (8,400 records)  
**Random seed:** 42  

| Metric | Value |
|--------|-------|
| AUROC | 0.7641 |
| AUPRC | 0.7753 |
| Precision | 0.8387 |
| Recall | 0.2645 |
| F1 | 0.4022 |
| FPR on Normal | 0.0624 |
| Threshold | 0.5009 |

**Per-class detection rates:** Generic=54.3%, Worms=13.6%, Fuzzers=10.2%, Exploits=7.5%  
**Artifacts:** `models/anomaly/anomaly_metadata.yaml`  
**Results:** `results/metrics/anomaly_results.yaml`  

---

## E4 — Decision Fusion (Classifier + Anomaly Detector)

**Date:** 2026-11-08  
**Phase:** 4  
**Script:** `experiments/train_fusion.py`  
**Calibration:** Isotonic regression on validation set (26,301 records)  
**Threshold selection:** Grid search on validation set  
**Random seed:** 42  

**Calibration metrics:** ECE=0.032, Brier=0.014  
**Selected thresholds:** conf=0.85, anomaly=-0.1318  

| Condition | Known Accuracy | Unknown Recall | FPR |
|-----------|---------------|----------------|-----|
| Classifier-only | 0.8202 | N/A | N/A |
| Anomaly-only | 0.5187 | N/A | N/A |
| Simple Fusion | 0.9387 | 0.0133 | 0.0457 |
| Tuned Fusion | 0.9857 | 0.4803 | 0.6630 |

**Open-set (Worms withheld, 44 test samples):** Recall=0.932, FPR=0.775, Precision=0.0009  
**Operational:** 24,581 flows/sec, P95 latency=0.045ms  
**Artifacts:** `models/calibration/`, `results/figures/`  
**Results:** `results/metrics/fusion_results.yaml`  

---

## E5 — SHAP Explainability

**Date:** 2026-11-08  
**Phase:** 5  
**Script:** `experiments/generate_shap.py`  
**Model:** XGBoost (from E1)  
**Sample:** 500 test records  

**Top-10 features by mean |SHAP|:**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | sbytes | 0.703456 |
| 2 | smean | 0.617742 |
| 3 | sttl | 0.522556 |
| 4 | service | 0.365681 |
| 5 | ct_srv_dst | 0.318394 |
| 6 | proto | 0.312695 |
| 7 | ct_dst_src_ltm | 0.284784 |
| 8 | ct_srv_src | 0.259175 |
| 9 | ct_dst_sport_ltm | 0.217146 |
| 10 | ct_src_dport_ltm | 0.203164 |

**Artifacts:** `results/figures/shap_summary.png`, `results/figures/shap_importance.png`, `results/metrics/shap_artifacts.pkl`  

---

## Pending Experiments

| ID | Description | Status |
|----|-------------|--------|
| E2 | CNN-LSTM vs Tabular ML | Pending (requires temporal sequence justification) |
| E6 | Multiple withheld attack families (Backdoor, Shellcode) | Pending |
| E7 | Feature engineering contribution analysis | Pending |
| E8 | PCAP/Live vs Batch operational comparison | Pending |
