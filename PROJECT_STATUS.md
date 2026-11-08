# Project Status

## Current Phase: Phase 8 Complete - All Phases Finished (100%)

**Last Updated:** 2026-11-09

---

## Phase 0 - Environment and Repository [COMPLETE]

- Python 3.13.1, Windows 11, 16 GB RAM
- Virtual environment with all dependencies installed
- Full project structure created
- `.gitignore`, `README.md`, configuration files, tracking files

---

## Phase 1 - Dataset Understanding & Preprocessing [COMPLETE]

- **Dataset:** UNSW-NB15 (official real benchmark, downloaded directly from source; NO synthetic data)
- Training: 175,341 rows | Testing: 82,332 rows | 45 columns
- No missing values, no duplicates, no infinities
- **Leakage risk identified:** `id` column (0.73 Pearson correlation with label) - rigorously excluded
- 10.37% train/test overlap documented as inherent benchmark artifact
- Severe class imbalance: 431:1 ratio (Generic:Worms)
- Preprocessing pipeline: 42 features, 3 categorical + 39 numerical, StandardScaler
- Unit test suite: 16 tests passing in `tests/test_preprocessing.py`

---

## Phase 2 - Multiclass Classifiers [COMPLETE]

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train Time |
|-------|----------|----------|-------------|------------|
| Random Forest | 0.6838 | 0.4945 | 0.7363 | 4.1s |
| **XGBoost** | **0.7636** | **0.5130** | **0.7801** | 19.1s |

XGBoost selected as primary classifier. Rare classes (Analysis, Backdoor) remain challenging due to natural data distribution.

---

## Phase 3 - Anomaly Detection [COMPLETE]

- Isolation Forest trained on **normal-only** traffic (56,000 records)
- Threshold: 0.5009 (5% target FPR)
- **AUROC: 0.7641**, AUPRC: 0.7753
- Generic detection: 54.3% | Worms detection: 13.6% (pre-fusion)

---

## Phase 4 - Confidence Calibration + Decision Fusion [COMPLETE]

- Calibrated XGBoost with isotonic regression (ECE=0.032, Brier=0.014)
- Best thresholds: conf=0.85, anomaly=-0.1318 (selected on validation set)

| Condition | Known Accuracy | Unknown Recall | FPR | Precision (Unknown) |
|-----------|---------------|----------------|-----|---------------------|
| Classifier-only | 0.8202 | - | - | - |
| Anomaly-only | 0.5187 | - | - | - |
| Simple Fusion | 0.9387 | 0.013 | 0.046 | 0.3826 |
| **Tuned Fusion** | **0.9857** | **0.480** | 0.663 | 0.6069 |

- **Open-set (Worms withheld): Recall=93.2%, FPR=77.5%**
- Throughput: 24,581 flows/sec | P95 latency: 0.045 ms/record

---

## Phase 5 - Explainability [COMPLETE]

- TreeSHAP computed across test set. Top features (mean |SHAP|):
  1. sbytes (7.035) | 2. smean (6.177) | 3. sttl (5.226)
  4. service (3.657) | 5. ct_srv_dst (3.184)
- Generated figures: `shap_summary.png`, `shap_importance.png`, `reliability_diagram.png`, `anomaly_auroc.png`, `fusion_confusion_matrix.png`

---

## Phase 6 - Traffic Processing & Ingestion [COMPLETE]

- `traffic/schema_mapping.py` - 42 UNSW-NB15 features, validation, PCAP-mappable subset
- `traffic/flow_extractor.py` - Scapy-based flow extraction with explicit unavailable feature marking
- `traffic/pcap_replay.py` - PCAP replay pipeline with schema compatibility check
- `traffic/packet_capture.py` - Authorized local capture with start/stop controls

---

## Phase 7 - Streamlit Dashboard [COMPLETE]

8 interactive pages implemented:
1. Overview - KPIs, attack distribution, recent alerts
2. Data Input - CSV upload, schema validation, sample dataset
3. Live Monitoring - Interface selector, capture controls (live disabled by default)
4. Attack Analytics - Predicted distribution, per-class metrics, confusion matrix
5. Anomaly Detection - Score histogram, threshold, open-set evaluation
6. Explainability - Global SHAP importance, local explanations
7. Model Performance - All metrics, calibration, operational benchmarks
8. Alert History - Full prediction log with export to CSV

Run command: `streamlit run dashboard/app.py`

---

## Phase 8 - Research Paper & Deliverables [COMPLETE]

- **Academic Paper**: `research_paper/paper.md` and `research_paper/paper.tex` (IEEE conference format)
- **Literature Review**: `research_paper/literature_review/literature_table.md` (12 verified references with DOIs)
- **Publication Tables**: LaTeX and CSV tables in `research_paper/tables/` and `results/tables/`
- **Publication Figures**: All 5 high-resolution evaluation figures in `research_paper/figures/`
- **Experiment Configs**: `experiments/configs/` with reproducible YAML configs for E1, E3, E4, E5
- **Notebooks**: Full suite `notebooks/01_dataset_exploration.ipynb` through `notebooks/06_final_analysis.ipynb`
- **Unit Tests**: 41 passed across all test modules (`tests/test_preprocessing.py`, `tests/test_inference.py`, `tests/test_decision_fusion.py`, `tests/test_schema_validation.py`)

---

## Verification & Test Summary

```
tests/test_decision_fusion.py: 9 passed
tests/test_inference.py: 6 passed
tests/test_preprocessing.py: 16 passed
tests/test_schema_validation.py: 10 passed
Total: 41 passed in 3.23s
```
