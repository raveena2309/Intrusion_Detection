# Project Status

## Current Phase: Phase 7 Complete — Full Project Done

**Last Updated:** 2026-11-08

---

## Phase 0 — Environment and Repository ✅

- Python 3.13.1, Windows 11, 16 GB RAM
- Virtual environment with all dependencies
- Full project structure created
- `.gitignore`, `README.md`, config files, tracking files

---

## Phase 1 — Dataset Understanding ✅

- **Dataset:** UNSW-NB15 (official real benchmark, NOT synthetic)
- Training: 175,341 rows | Testing: 82,332 rows | 45 columns
- No missing values, no duplicates, no infinities
- **Leakage risk identified:** `id` column (0.73 corr with label) — excluded from features
- 10.37% train/test overlap documented
- Severe class imbalance: 431:1 ratio (Generic:Worms)
- Preprocessing pipeline: 42 features, 3 categorical + 39 numerical, StandardScaler

---

## Phase 2 — Multiclass Classifier ✅

| Model | Accuracy | Macro-F1 | Weighted-F1 | Train Time |
|-------|----------|----------|-------------|------------|
| Random Forest | 0.6838 | 0.4945 | 0.7363 | 4.1s |
| **XGBoost** | **0.7636** | **0.5130** | **0.7801** | 19.1s |

XGBoost selected as primary classifier. Rare classes (Analysis, Backdoor) remain challenging.

---

## Phase 3 — Anomaly Detector ✅

- Isolation Forest trained on **normal-only** traffic (47,600 records)
- Threshold: 0.5009 (5% target FPR)
- **AUROC: 0.7641**, AUPRC: 0.7753
- Generic detection: 54.3% | Worms detection: 13.6% (pre-fusion)

---

## Phase 4 — Confidence Calibration + Decision Fusion ✅

- Calibrated XGBoost with isotonic regression (ECE=0.032, Brier=0.014)
- Best thresholds: conf=0.85, anomaly=-0.1318 (selected on validation set)

| Condition | Known Accuracy | Unknown Recall | FPR |
|-----------|---------------|----------------|-----|
| Classifier-only | 0.8202 | — | — |
| Anomaly-only | 0.5187 | — | — |
| Simple Fusion | 0.9387 | 0.013 | 0.046 |
| **Tuned Fusion** | **0.9857** | **0.480** | 0.663 |

**Open-set (Worms withheld): Recall=93.2%, FPR=77.5%**
Throughput: 24,581 flows/sec | P95 latency: 0.045 ms/record

---

## Phase 5 — Explainability ✅

SHAP computed on 500 test records. Top features (mean |SHAP|):
1. sbytes (0.703) | 2. smean (0.618) | 3. sttl (0.523)
4. service (0.366) | 5. ct_srv_dst (0.318)

Figures saved: shap_summary.png, shap_importance.png, reliability_diagram.png, anomaly_auroc.png

---

## Phase 6 — Traffic Processing ✅

- `traffic/schema_mapping.py` — 42 UNSW-NB15 features, validation, PCAP-mappable subset
- `traffic/flow_extractor.py` — Scapy-based flow extraction with explicit unavailable feature marking
- `traffic/pcap_replay.py` — PCAP replay pipeline with schema compatibility check
- `traffic/packet_capture.py` — Authorized local capture with start/stop controls

---

## Phase 7 — Streamlit Dashboard ✅

8 pages implemented:
1. Overview — KPIs, attack distribution, recent alerts
2. Data Input — CSV upload, schema validation, sample dataset
3. Live Monitoring — Interface selector, capture controls (live disabled by default)
4. Attack Analytics — Predicted distribution, per-class metrics, confusion matrix
5. Anomaly Detection — Score histogram, threshold, open-set evaluation
6. Explainability — Global SHAP importance, local explanations
7. Model Performance — All metrics, calibration, operational benchmarks
8. Alert History — Full prediction log with export to CSV

**Run:** `streamlit run dashboard/app.py`

---

## Known Issues / Limitations

- RF model (354 MB) excluded from git due to GitHub file size limit — only metadata YAML committed
- XGBoost (13.7 MB) also excluded — run `experiments/train_classifiers.py` to regenerate
- Live capture requires admin privileges and Scapy on Windows
- Worms class has only 44 test samples — open-set metrics have high variance
- 10.37% train/test overlap is a property of official UNSW-NB15 split (documented, not fixed)

---

## Remaining Work (Phase 8)

- Research paper draft (`research_paper/paper.tex`)
- Literature review table (`research_paper/literature_review/`)
- Additional notebooks (02-06)
- Final experiment re-runs for reproducibility
