# Intelligent Network Intrusion Detection System

**A Confidence-Aware Hybrid Machine Learning Framework for Explainable Multi-Class and Anomalous Network Intrusion Detection with Real-Time Monitoring**

## Overview

This project implements a complete, modular Network Intrusion Detection System (NIDS) that:

1. **Classifies** network traffic into multiple known classes (normal + 9 attack categories)
2. **Detects anomalies** in traffic that differ from normal network behaviour
3. **Identifies unknown attacks** using confidence-aware hybrid decision fusion
4. **Explains** predictions using SHAP and interpretable anomaly indicators
5. **Monitors** traffic through an interactive Streamlit dashboard
6. **Validates** novel attack detection via an open-set evaluation protocol

**Primary benchmark dataset:** UNSW-NB15 (Real benchmark dataset, strictly without synthetic generation)

## Research Questions & Findings

- **RQ1:** How accurately can conventional ML baselines classify known network traffic categories?
  - *Finding:* XGBoost achieves 76.36% accuracy and 0.5130 macro-F1 (0.7801 weighted-F1) on the natural 10-class distribution, outperforming Random Forest (68.38% acc, 0.4945 macro-F1).
- **RQ2:** Can a detector trained only on normal traffic identify attack families withheld from supervised training?
  - *Finding:* Isolation Forest trained exclusively on normal traffic achieves AUROC = 0.7641 and AUPRC = 0.7753 on the full test set.
- **RQ3:** Does confidence-aware fusion improve unknown-attack recognition vs. classifier alone?
  - *Finding:* Tuned 5-rule fusion raises known-attack accuracy to 98.57% while flagging 48.0% of general anomalies and 93.18% of withheld novel attacks.
- **RQ4:** What trade-off exists between unknown-attack recall and false-positive/false-acceptance rates?
  - *Finding:* High novel-attack recall (93.18%) comes with elevated FPR (77.54%) in zero-day regimes, suitable for tiered SOC alert triage.
- **RQ5:** Can SHAP explanations help users understand feature influences on predictions?
  - *Finding:* TreeSHAP attributes highest influence to `sbytes` (7.035), `smean` (6.177), `sttl` (5.226), and `service` (3.657), validating network-domain semantics.
- **RQ6:** Can the trained pipeline process replayed or captured traffic with practical latency?
  - *Finding:* End-to-end CPU throughput reaches 24,581 flows/sec with 0.040 ms median latency and 0.045 ms P95 latency.

## Setup Instructions

### Prerequisites

- Python 3.10+ (tested with Python 3.13.1)
- Windows 11 / Linux / macOS
- 16 GB RAM recommended
- CPU-only execution fully supported

### Installation

```bash
# Clone the repository
git clone https://github.com/raveena2309/Intrusion_Detection.git
cd Intrusion_Detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

This project uses the official **UNSW-NB15** dataset.

1. Download the training and testing CSV files from the official ACCS repository:
   - `UNSW_NB15_training-set.csv` (175,341 rows)
   - `UNSW_NB15_testing-set.csv` (82,332 rows)
2. Place the downloaded files in `datasets/raw/`
3. Verify and explore the dataset:
   ```bash
   python src/data_loader.py
   ```

### Running Experiments

```bash
# Phase 2: Train multiclass classifiers (RF + XGBoost)
python experiments/train_classifiers.py

# Phase 3: Train anomaly detector (Isolation Forest on normal traffic)
python experiments/train_anomaly_detector.py

# Phase 4: Calibrate confidence & optimize decision fusion
python experiments/train_fusion.py

# Phase 5: Compute TreeSHAP explanations
python experiments/generate_shap.py

# Phase 8: Export publication tables
python experiments/export_tables.py
```

### Running Tests

```bash
python -m pytest tests/ -v
# 41 unit tests covering preprocessing, inference, decision fusion, and schema validation
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
Intrusion_Detection/
├── README.md                    # Project overview and instructions
├── PROJECT_STATUS.md            # Detailed progress tracking across all 8 phases
├── EXPERIMENT_LOG.md            # Verified real experiment logs and metrics
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules (.joblib models excluded)
├── config/                      # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── experiment_config.yaml
├── datasets/                    # Data directory (raw files ignored in git)
│   ├── raw/                     # Original downloaded benchmark files
│   ├── processed/               # Preprocessed data copies
│   └── sample/                  # Stratified 500-row sample for quick testing
├── notebooks/                   # Jupyter analysis notebooks (01 to 06)
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_multiclass_baselines.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_confidence_fusion.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_final_analysis.ipynb
├── src/                         # Core modular source code
│   ├── data_loader.py           # Dataset loading and validation
│   ├── preprocessing.py         # 42-feature preprocessor pipeline
│   ├── feature_engineering.py   # Ratio and domain feature engineering
│   ├── multiclass_classifier.py # RF and XGBoost implementations
│   ├── anomaly_detector.py      # Isolation Forest anomaly detector
│   ├── confidence_calibration.py# Isotonic / Platt calibration
│   ├── decision_fusion.py       # 5-rule confidence-aware fusion engine
│   ├── explainability.py        # TreeSHAP explainer and local plots
│   ├── evaluation.py            # Open-set and operational metrics
│   └── inference.py             # Real-time inference engine
├── traffic/                     # PCAP and live traffic modules
│   ├── schema_mapping.py        # Feature mapping and validation
│   ├── flow_extractor.py        # Scapy-based network flow extraction
│   ├── pcap_replay.py           # PCAP replay engine
│   └── packet_capture.py        # Local network capture
├── dashboard/                   # 8-page Streamlit monitoring application
│   ├── app.py                   # Main dashboard entrypoint
│   └── pages/                   # Interactive dashboard pages
├── experiments/                 # Experiment execution scripts and configs
│   └── configs/                 # E1, E3, E4, E5 experiment definitions
├── results/                     # Reproducible outputs
│   ├── metrics/                 # Metric YAMLs and SHAP rankings
│   ├── figures/                 # AUROC, calibration, SHAP, and fusion plots
│   └── tables/                  # Formatted LaTeX / CSV result tables
├── research_paper/              # Complete research paper package
│   ├── paper.md                 # Full manuscript draft in Markdown
│   ├── paper.tex                # IEEE conference format LaTeX manuscript
│   ├── literature_review/       # Literature matrix (12 verified papers)
│   ├── figures/                 # High-resolution figures for publication
│   └── tables/                  # Publication-ready LaTeX tables
└── tests/                       # Unit test suite (41 tests, 100% passing)
    ├── test_preprocessing.py
    ├── test_inference.py
    ├── test_decision_fusion.py
    └── test_schema_validation.py
```

## Development Phases

| Phase | Description | Status |
|---|---|---|
| 0 | Environment and repository setup | Completed |
| 1 | Dataset understanding & preprocessing pipeline | Completed |
| 2 | Multiclass baseline classifiers (RF & XGBoost) | Completed |
| 3 | Normal-only anomaly detector (Isolation Forest) | Completed |
| 4 | Confidence calibration & 5-rule decision fusion | Completed |
| 5 | TreeSHAP explainability & feature attributions | Completed |
| 6 | Traffic processing (PCAP replay & live capture) | Completed |
| 7 | Streamlit monitoring dashboard (8 interactive pages) | Completed |
| 8 | Research paper (IEEE .tex & .md), literature review & validation | Completed |

## System Architecture

```
Dataset CSV / PCAP Replay / Authorized Live Capture
                         │
                         ▼
              Traffic and Flow Processing
                         │
                         ▼
          Schema Validation and Preprocessing
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
    Multiclass Classifier     Normal-only Anomaly Detector
    (XGBoost / RF)            (Isolation Forest)
             │                       │
             ▼                       ▼
    Class Probabilities       Anomaly Score / Flag
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
             Confidence-aware Decision Fusion
                         │
       ┌─────────────────┼───────────────────┐
       │                 │                   │
       ▼                 ▼                   ▼
     Normal          Known Attack     Unknown Suspicious
                         │
                         ▼
              Explainability / SHAP
                         │
                         ▼
                Monitoring Dashboard
```

## Hardware and Performance Specifications

- **OS:** Windows 11 Home
- **Python:** 3.13.1
- **RAM:** 16 GB
- **Execution Mode:** CPU-only
- **Inference Throughput:** 24,581 samples/sec
- **Median Latency:** 0.040 ms/flow
- **P95 Latency:** 0.045 ms/flow

## License and Ethics

This project is developed for cybersecurity research and educational purposes. Raw UNSW-NB15 benchmark data is distributed according to its original license by the Australian Centre for Cyber Security (ACCS).

## References

1. N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems," in *2015 Military Communications and Information Systems Conference (MilCIS)*, IEEE, 2015. DOI: 10.1109/MilCIS.2015.7348942
2. F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *2008 Eighth IEEE International Conference on Data Mining*, 2008. DOI: 10.1109/ICDM.2008.17
3. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *ICML*, 2017.
4. S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, 2020. DOI: 10.1038/s42256-019-0138-9
