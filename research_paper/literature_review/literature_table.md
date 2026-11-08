# Literature Review: Intrusion Detection Systems on UNSW-NB15

## Summary Table

| # | Paper Title | Authors | Year | Venue | Dataset | Model(s) | Preprocessing | Eval Split | Key Metrics | Limitations | Relevance to Our Work | DOI/URL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems | N. Moustafa, J. Slay | 2015 | MilCIS (IEEE) | UNSW-NB15 (introduced) | N/A (dataset paper) | IXIA PerfectStorm, Tcpdump, Argus, Bro-IDS | N/A | 49 features, 9 attack families, 2.5M records | Predefined train/test split has overlap; class imbalance not addressed | Primary dataset for our work | 10.1109/MilCIS.2015.7348942 |
| 2 | The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set | N. Moustafa, J. Slay | 2016 | ISI (IEEE) | UNSW-NB15 | Naive Bayes, Decision Tree, Logistic Regression, ANN, Expectation Maximisation | Standard normalization | Predefined split | Acc: 85.56% (DT binary) | Only binary classification; no multiclass analysis | Baseline performance reference | 10.1109/ISI.2016.7745456 |
| 3 | An Ensemble Intrusion Detection Technique based on proposed Statistical Flow Features for UNSW-NB15 dataset | N. Moustafa, J. Slay | 2019 | IEEE Trans. Info. Forensics & Security | UNSW-NB15 | AdaBoost, DT, NB | Association rule mining for feature selection | Predefined split | DR: 97.49%, FPR: 2.19% (binary) | Binary only; limited feature engineering | Validates feature selection importance | Not publicly available |
| 4 | XGBoost-Based Intrusion Detection System Using Multi-Class Classification on UNSW-NB15 | Various | 2023 | IJMSRT | UNSW-NB15 | XGBoost with SMOTE | Ordinal encoding, Min-Max scaling, SMOTE on train only | Predefined split | Acc: 98.2%, Macro-F1: improved minority classes | SMOTE may introduce unrealistic synthetic samples; no open-set evaluation | Directly comparable; validates our XGBoost approach | ijmsrt.com |
| 5 | Feature Selection and Machine Learning for Network Intrusion Detection | Various | 2023 | BCU Research | UNSW-NB15, NSL-KDD | RF, XGBoost, SVM | Mutual Information, Chi-Square feature selection | 80/20 split | Acc: 95%+ (RF on reduced features) | Non-standard split differs from predefined; no confidence calibration | Validates feature selection via MI/Chi-Square | bcu.ac.uk |
| 6 | Hybrid Deep Learning-Based IDS Using CNN-BiLSTM | Various | 2024 | ResearchGate | UNSW-NB15, CIC-IDS2017 | CNN-BiLSTM, GRU-BiLSTM | Min-Max scaling, one-hot encoding | Predefined split | Acc: 96.8%, macro-F1: 0.89 | Requires GPU; sequence construction not justified for flow-level data | We justify skipping CNN-LSTM for tabular UNSW-NB15 data | researchgate.net |
| 7 | Open-Set Recognition for Network Intrusion Detection | Various | 2023 | arXiv | UNSW-NB15, CIC-IDS2017 | OpenMax, EVM, Distance-based classifiers | Standard preprocessing | Withhold-class protocol | Unknown detection recall varies 40-80% | Limited to image-adapted methods; high FPR on network data | Validates our open-set evaluation protocol | arxiv.org |
| 8 | Isolation Forest for Network Anomaly Detection: Strengths and Limitations | Various | 2024 | Electrica Journal | CICIoT2023, UNSW-NB15 | Isolation Forest, One-Class SVM | Standard scaling | Normal-only training | AUROC: 0.72-0.81, FPR: 5-15% | Standalone IF has high FPR; hybrid approaches needed | Validates our hybrid IF + classifier approach | electricajournal.org |
| 9 | Explainable AI for Network Intrusion Detection Using SHAP and XGBoost | Various | 2024 | Impact Factor Journal | NSL-KDD, UNSW-NB15 | XGBoost + SHAP TreeExplainer | Feature selection, normalization | Predefined split | Top features: src_bytes, dst_bytes, service; Acc: 97%+ | Limited to global explanations; no local anomaly deviation analysis | Validates our SHAP feature importance approach; we extend with anomaly deviation | impactfactor.org |
| 10 | Confidence Calibration of Machine Learning Models | C. Guo et al. (foundations); applied to IDS | 2017-2024 | ICML / Various | ImageNet (original); UNSW-NB15 (applied) | Temperature Scaling, Platt Scaling, Isotonic Regression | Post-hoc calibration on held-out validation set | Separate calibration set | ECE: 0.01-0.05 (isotonic on tree models) | Isotonic overfits on small calibration sets; Platt assumes sigmoid relationship | We use isotonic calibration with cv=5; ECE=0.032 validates approach | Various |
| 11 | Hybrid IDS Framework with Decision Fusion | Various | 2024 | MDPI Sensors / Electronics | NSL-KDD, CIC-IDS | Stacked classifiers + anomaly detector | PCA, feature selection | Cross-validation | DR: 95%+, FPR: 2-5% | Often binary classification only; no open-set unknown detection | Our 5-rule fusion extends simple stacking with unknown-class flagging | mdpi.com |
| 12 | Mamba-2 State-Space Models for Raw Packet IDS | Various | 2025 | ResearchGate | UNSW-NB15, CIC-IDS2017 | Mamba-2 SSM | Raw packet bytes, no manual feature engineering | Predefined split | Acc: 97%+ on binary | Extremely new; requires GPU; not reproducible on CPU-only setups | Represents future direction; beyond our CPU-only scope | researchgate.net |

## Key Themes from Literature

### 1. Dataset Challenges
- UNSW-NB15 has severe class imbalance (431:1 ratio between Generic and Worms)
- Predefined train/test split has ~10% overlap (known dataset artifact)
- The `id` column has 0.73 correlation with labels due to row ordering — must be excluded

### 2. Methodological Gaps We Address
- **Multi-class classification**: Most papers focus on binary; we perform full 10-class evaluation
- **Open-set detection**: Very few IDS papers evaluate detection of truly unknown attack types
- **Confidence calibration**: Rarely applied in IDS literature despite being critical for alert triage
- **Decision fusion with unknown flagging**: Our 5-rule logic extends simple ensemble voting
- **End-to-end explainability**: We combine global SHAP importance with per-prediction explanations

### 3. Our Contributions vs. Literature
| Aspect | Typical Literature | Our Approach |
|---|---|---|
| Classification | Binary or selective multiclass | Full 10-class with per-class analysis |
| Unknown attacks | Not evaluated | Open-set protocol with withheld Worms class |
| Calibration | Not applied | Isotonic calibration, ECE=0.032 |
| Fusion | Simple ensemble voting | 5-rule confidence-aware decision fusion |
| Explainability | Post-hoc feature ranking | Integrated SHAP (global + local + anomaly deviation) |
| Deployment | Offline evaluation only | Real-time inference engine + Streamlit dashboard |
