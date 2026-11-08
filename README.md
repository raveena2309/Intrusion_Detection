# Intelligent Network Intrusion Detection System

**A Confidence-Aware Hybrid Machine Learning Framework for Explainable Multi-Class and Anomalous Network Intrusion Detection with Real-Time Monitoring**

## Overview

This project implements a complete, modular Network Intrusion Detection System (NIDS) that:

1. **Classifies** network traffic into multiple known classes (normal + 9 attack categories)
2. **Detects anomalies** in traffic that differ from normal network behaviour
3. **Identifies unknown attacks** using confidence-aware hybrid decision fusion
4. **Explains** predictions using SHAP and interpretable anomaly indicators
5. **Monitors** traffic through a Streamlit dashboard

**Primary benchmark dataset:** UNSW-NB15

## Research Questions

- **RQ1:** How accurately can conventional ML baselines classify known network traffic categories?
- **RQ2:** Can a detector trained only on normal traffic identify attack families withheld from supervised training?
- **RQ3:** Does confidence-aware fusion improve unknown-attack recognition vs. classifier alone?
- **RQ4:** What trade-off exists between unknown-attack recall and false-positive/false-acceptance rates?
- **RQ5:** Can SHAP explanations help users understand feature influences on predictions?
- **RQ6:** Can the trained pipeline process replayed or captured traffic with practical latency?

## Setup Instructions

### Prerequisites

- Python 3.10+ (tested with Python 3.13.1)
- Windows 11 (primary development platform)
- 16 GB RAM recommended
- GPU optional (CPU-only pipeline supported)

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

This project uses the **UNSW-NB15** dataset. The dataset is NOT included in this repository.

1. Visit the official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download the training and testing CSV files:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
3. Place the downloaded files in `datasets/raw/`
4. Run the dataset exploration script:
   ```bash
   python src/data_loader.py
   ```

### Running the Dashboard (Phase 7+)

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
Intrusion_Detection/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Current progress tracking
├── EXPERIMENT_LOG.md            # Experiment records
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── config/                      # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── experiment_config.yaml
├── datasets/                    # Data directory (raw files not in git)
│   ├── raw/                     # Original downloaded files
│   ├── processed/               # Preprocessed data
│   └── sample/                  # Small samples for testing
├── notebooks/                   # Jupyter notebooks for exploration
├── src/                         # Core source code
│   ├── data_loader.py           # Dataset loading and validation
│   ├── preprocessing.py         # Feature preprocessing pipeline
│   └── ...                      # Additional modules (added per phase)
├── traffic/                     # PCAP and live capture modules
├── dashboard/                   # Streamlit dashboard
├── models/                      # Saved model artifacts
├── experiments/                 # Experiment configs, logs, reports
├── results/                     # Metrics, figures, tables
├── research_paper/              # Paper drafts and literature review
└── tests/                       # Unit tests
```

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment and repository setup | ✅ Complete |
| 1 | Dataset understanding | 🔄 In Progress |
| 2 | Multiclass baseline | ⏳ Pending |
| 3 | Anomaly detector | ⏳ Pending |
| 4 | Fusion and open-set evaluation | ⏳ Pending |
| 5 | Explainability | ⏳ Pending |
| 6 | Traffic processing | ⏳ Pending |
| 7 | Dashboard | ⏳ Pending |
| 8 | Research paper and final validation | ⏳ Pending |

## System Architecture

```
Dataset CSV / PCAP Replay / Authorized Live Capture
                         |
                         v
              Traffic and Flow Processing
                         |
                         v
          Schema Validation and Preprocessing
                         |
             +-----------+-----------+
             |                       |
             v                       v
    Multiclass Classifier     Normal-only Anomaly Detector
    (XGBoost / RF)            (Isolation Forest)
             |                       |
             v                       v
    Class Probabilities       Anomaly Score / Flag
             |                       |
             +-----------+-----------+
                         |
                         v
             Confidence-aware Decision Fusion
                         |
       +-----------------+-------------------+
       |                 |                   |
       v                 v                   v
     Normal          Known Attack     Unknown Suspicious
                         |
                         v
              Explainability / SHAP
                         |
                         v
                Monitoring Dashboard
```

## Environment

- **OS:** Windows 11 Home
- **Python:** 3.13.1
- **RAM:** 16 GB
- **GPU:** CPU-only primary pipeline

## License

This project is developed for research and educational purposes.

## Acknowledgements

- UNSW-NB15 dataset: Moustafa, N. and Slay, J. (2015). UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems. Military Communications and Information Systems Conference (MilCIS).
