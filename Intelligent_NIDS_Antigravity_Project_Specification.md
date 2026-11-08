# Intelligent Network Intrusion Detection System

## Detailed Project Specification for Antigravity

**Project type:** Machine Learning mini-project with research-paper
objective\
**Working title:** *A Confidence-Aware Hybrid Machine Learning Framework
for Explainable Multi-Class and Anomalous Network Intrusion Detection
with Real-Time Monitoring*\
**Primary benchmark:** UNSW-NB15\
**Implementation:** Python, scikit-learn, XGBoost, PyTorch (optional
Autoencoder), SHAP, Scapy, Streamlit

------------------------------------------------------------------------

# 1. Project Overview

Build a complete, modular Network Intrusion Detection System (NIDS) that
can:

1.  Classify network traffic into multiple known classes (normal traffic
    and known attack categories).
2.  Detect anomalous traffic that differs from normal network behaviour.
3.  Identify traffic that may belong to an attack category withheld from
    supervised training.
4.  Combine classifier confidence and anomaly scores in a transparent
    decision-fusion module.
5.  Explain supervised predictions using SHAP and provide interpretable
    anomaly indicators.
6.  Monitor uploaded datasets, replayed PCAP traffic, and---where
    technically supported---live captured traffic through a dashboard.
7.  Produce reproducible experiments and results suitable for a research
    paper.

The dashboard is the demonstration interface. The research contribution
is the carefully evaluated hybrid detection methodology---not the
dashboard itself.

## 1.1 Core research problem

Conventional supervised IDS models learn a fixed set of labels. When
traffic from an unfamiliar attack family appears, a closed-set
classifier may force it into one of its known classes. This project
investigates whether a supervised multiclass classifier combined with a
normal-only anomaly detector and a confidence-aware rejection mechanism
can improve recognition of withheld attack families while controlling
false alarms.

## 1.2 Proposed research questions

-   **RQ1:** How accurately can conventional ML and deep-learning
    baselines classify known network traffic categories?
-   **RQ2:** Can a detector trained only on normal traffic identify
    attack families withheld from supervised training?
-   **RQ3:** Does confidence-aware fusion of classifier probabilities
    and anomaly scores improve unknown-attack recognition compared with
    a classifier alone?
-   **RQ4:** What trade-off exists between unknown-attack recall and
    false-positive/false-acceptance rates?
-   **RQ5:** Can SHAP explanations help users understand the features
    influencing known-class predictions?
-   **RQ6:** Can the trained pipeline process replayed or captured
    traffic with practical latency and throughput?

Do not claim that the system detects real-world zero-day
vulnerabilities. The defensible experimental claim is recognition of
attack families withheld from the model's supervised training.

------------------------------------------------------------------------

# 2. Scope and Expected Deliverables

## 2.1 Required capabilities

-   Dataset exploration and preprocessing.
-   Binary and multiclass classification.
-   Normal-only anomaly detection.
-   Open-set/withheld-attack evaluation.
-   Confidence calibration.
-   Hybrid decision fusion.
-   Explainable predictions.
-   CSV upload and prediction.
-   PCAP traffic replay and compatible flow extraction.
-   Optional live packet capture on an authorized test network.
-   Streamlit dashboard.
-   Evaluation metrics, plots, experiment logs and reproducible
    configuration.
-   Research-paper-ready tables and methodology notes.

## 2.2 Final deliverables

1.  Complete source code.
2.  Trained model artifacts and preprocessing pipeline.
3.  Streamlit dashboard.
4.  Dataset preparation and experiment scripts.
5.  Results, figures, metrics and experiment configurations.
6.  README with setup and execution instructions.
7.  Research manuscript draft in IEEE-style structure.
8.  Project presentation and demonstration instructions.

## 2.3 Scope boundaries

-   Do not build offensive attack-generation functionality.
-   Do not scan, capture or test networks without authorization.
-   Do not claim that a dataset replay is equivalent to live network
    deployment.
-   Do not claim production-grade prevention; this is a detection and
    monitoring system.
-   Do not fabricate performance numbers or claim novelty without
    literature verification.
-   Keep the project runnable on a CPU-only Windows laptop where
    possible. GPU support may be optional.

------------------------------------------------------------------------

# 3. Dataset Strategy

## 3.1 Primary dataset: UNSW-NB15

Use the official UNSW-NB15 dataset:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

The dataset contains normal traffic and nine attack categories:

-   Analysis
-   Backdoor
-   DoS
-   Exploits
-   Fuzzers
-   Generic
-   Reconnaissance
-   Shellcode
-   Worms

The dataset is commonly distributed with predefined training and testing
CSV files. Verify the exact files, columns, row counts, labels and
versions after download. Do not assume every mirror has identical
contents.

### Required dataset handling

-   Keep raw files unchanged in `datasets/raw/`.
-   Record source URL, download date, file names, sizes and checksums
    where possible.
-   Inspect column names and target columns before coding.
-   Identify categorical, numerical, label and identifier columns.
-   Check missing values, duplicates, infinities, invalid values and
    class distribution.
-   Check for duplicate or overlapping records across official
    train/test partitions.
-   Never silently discard records or features; document every decision.

## 3.2 Optional secondary dataset: CIC-IDS2017

Official source: https://www.unb.ca/cic/datasets/ids-2017.html

Use only after the UNSW-NB15 pipeline is complete. Cross-dataset
evaluation is optional and must account for different feature
definitions, units, flow-generation methods and label taxonomies. Do not
directly feed one dataset's columns into a model trained on another
dataset without a justified feature mapping.

## 3.3 Dataset tracks

Maintain two separate evaluation tracks.

### Track A: Standard closed-set multiclass evaluation

-   Use the official training partition for model development.
-   Use the official test partition for final evaluation.
-   Keep test data untouched during model selection.
-   Report class-wise metrics and macro-F1, not accuracy alone.

### Track B: Open-set / withheld-attack evaluation

-   Select one or more attack families to withhold from supervised
    training.
-   Choose withheld classes based on sufficient sample counts and
    meaningful evaluation.
-   Train the supervised classifier without withheld attack labels or
    examples.
-   Train the anomaly detector using normal training traffic only.
-   Use validation data that does not include withheld attack examples
    for threshold selection, unless a separate explicitly labelled
    open-set calibration protocol is justified.
-   Evaluate on normal traffic, known attacks and withheld attack
    families.
-   Repeat with multiple withheld attack families when sample sizes
    permit.
-   Document the split protocol and random seeds.

Do not use withheld attack records for supervised training, feature
selection, threshold tuning, calibration, early stopping or model
selection. If a validation procedure uses unknown-class examples,
explicitly label it as an additional setting and keep it separate from
the strict unknown-class test.

------------------------------------------------------------------------

# 4. Proposed System Architecture

The system should have the following logical pipeline:

``` text
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
                         |
                         v
                Alert and Report Logs
```

Keep the model pipeline independent of Streamlit. The dashboard must
call reusable service functions rather than contain training code.

------------------------------------------------------------------------

# 5. Model Development

## 5.1 Baseline models

Implement and evaluate:

1.  Random Forest --- conventional baseline.
2.  XGBoost --- primary tabular multiclass candidate.
3.  CNN-LSTM --- comparison with the supplied 2025 reference-paper
    direction, only if a defensible sequence construction is designed.

Important: UNSW-NB15 is tabular flow data. Do not arbitrarily reshape
individual records into sequences to make an LSTM run. A sequence model
requires meaningful ordering and grouping, such as flows grouped by a
justified entity and ordered by time. If the source data cannot support
valid temporal sequences, state this limitation and use tabular models
as the primary approach.

## 5.2 Multiclass classification

Train a model to predict the normal class and the nine known attack
categories.

Requirements:

-   Use a consistent label mapping saved with the model.
-   Fit encoders and scalers on training data only.
-   Handle unseen categorical values safely.
-   Investigate class weighting before oversampling.
-   If resampling is used, apply it only inside training folds.
-   Evaluate on the untouched test split.
-   Save predicted probabilities as well as predicted labels.
-   Record model configuration, random seed and software versions.

## 5.3 Anomaly detection

Primary model: Isolation Forest.

Training protocol:

-   Train using normal training records only.
-   Apply exactly the same feature schema and fitted preprocessing used
    by the anomaly pipeline.
-   Establish an anomaly score convention and document it clearly.
-   Tune the decision threshold using normal-only validation data or a
    clearly documented permitted calibration procedure.
-   Evaluate on normal, known attack and withheld attack test records.

Optional model: Autoencoder.

-   Train only on normal training traffic for the primary one-class
    setting.
-   Use reconstruction error as an anomaly score.
-   Choose architecture and threshold using validation data.
-   Compare with Isolation Forest.
-   Do not add this model until the Isolation Forest pipeline is correct
    and reproducible.

## 5.4 Confidence calibration

The multiclass classifier's raw maximum probability is not automatically
a reliable confidence estimate.

Investigate calibration methods such as:

-   Temperature scaling, where appropriate.
-   Platt scaling or isotonic calibration for suitable setups.
-   Reliability diagrams.
-   Expected Calibration Error (ECE) or Brier score.

Calibration must use a validation set separate from the final test set.
For strict open-set evaluation, do not use withheld attack examples to
calibrate the known-class classifier.

## 5.5 Confidence-aware decision fusion

Implement the fusion mechanism as a separate, testable module.

Inputs:

-   Predicted known class.
-   Calibrated maximum class probability.
-   Full class probability vector, if useful.
-   Anomaly score.
-   Anomaly threshold.
-   Confidence threshold.

Initial conceptual rules:

1.  If the classifier predicts a known attack with sufficiently high
    calibrated confidence, output that known attack.
2.  If the classifier predicts normal but the anomaly detector flags the
    record, output `unknown_suspicious`.
3.  If classifier confidence is below the calibrated rejection threshold
    and anomaly score is high, output `unknown_suspicious`.
4.  If classifier confidence is low and anomaly score is not high,
    output `uncertain_review` rather than falsely asserting normality.
5.  Output `normal` only when the classifier supports normal traffic and
    the anomaly detector does not flag it, subject to the validated
    decision policy.

Do not hard-code arbitrary values such as 0.5 or 0.8 and call them
optimal. Select thresholds on permitted validation data and document the
optimization objective.

Compare at least:

-   Classifier alone.
-   Anomaly detector alone.
-   Simple rule-based fusion.
-   Tuned confidence-aware fusion.

Investigate the trade-off between unknown attack recall and false
alarms. Keep the fusion formula interpretable and provide an ablation
study.

------------------------------------------------------------------------

# 6. Explainable AI

Use SHAP for the supervised classifier.

Required outputs:

-   Global feature importance.
-   Summary plot.
-   Local explanation for an individual prediction.
-   Feature contribution values in the dashboard.
-   Explanation text that does not overstate causal meaning.

For anomaly detection, provide a separate explanation approach. Options
include:

-   Showing the largest deviations from normal feature ranges.
-   Comparing a record with normal training distributions.
-   Using model-agnostic explanations where technically justified.

Do not label classifier SHAP values as explanations of the Isolation
Forest unless an appropriate explainer is separately implemented and
validated.

Dashboard explanations should state that feature contributions describe
model behaviour, not proof of malicious intent.

------------------------------------------------------------------------

# 7. Traffic Processing, PCAP Replay and Live Capture

This module is required for the full demonstration but must be developed
after the dataset model pipeline works.

## 7.1 Common prediction interface

Create a shared interface that accepts a flow record with the expected
feature schema and returns:

``` json
{
  "prediction": "unknown_suspicious",
  "known_class": "Normal",
  "confidence": 0.42,
  "anomaly_score": 0.81,
  "is_anomaly": true,
  "explanation_available": true
}
```

This is an illustrative response schema, not a real prediction.

## 7.2 PCAP replay

-   Accept a PCAP file.
-   Parse packets and construct bidirectional or otherwise consistently
    defined flows.
-   Define flow timeout and aggregation behaviour.
-   Extract only features that can be computed consistently.
-   Handle incomplete flows and unsupported protocols.
-   Map extracted features to the trained model's exact feature schema.
-   Clearly mark unavailable dataset features.
-   Do not invent values for unavailable features.
-   If a compatible mapping cannot be established, show that live/PCAP
    inference is unsupported for that model rather than returning
    misleading predictions.

## 7.3 Live capture

Use Scapy for authorized local capture.

-   Provide explicit start and stop controls.
-   Capture only on an interface selected by the user.
-   Explain required permissions.
-   Do not capture payloads unnecessarily.
-   Prefer metadata and flow statistics.
-   Do not transmit packets or generate attacks.
-   Store captures only when the user explicitly requests it.
-   Show capture errors and unsupported feature warnings.

A practical implementation may use a dedicated flow-generation tool such
as CICFlowMeter, provided its output schema is reconciled with the
model's training features. The project must document differences between
benchmark flow features and live extracted features.

------------------------------------------------------------------------

# 8. Streamlit Dashboard Requirements

Create a clean, professional cybersecurity dashboard. Prefer a dark
security-operations visual style, readable charts, consistent spacing
and clear alert severity. Avoid clutter.

## 8.1 Overview page

Show:

-   Total processed flows.
-   Normal flows.
-   Known attack flows.
-   Unknown suspicious flows.
-   Uncertain/review flows.
-   Recent alerts.
-   Traffic timeline.
-   Attack distribution.

Clearly distinguish actual dataset labels from model predictions.

## 8.2 Data input page

Allow:

-   Uploading supported CSV files.
-   Uploading PCAP files for replay if supported.
-   Selecting a sample dataset.
-   Viewing schema validation.
-   Downloading prediction results.

Show clear validation errors for missing or incompatible features.

## 8.3 Live monitoring page

Show:

-   Selected network interface.
-   Start/stop capture.
-   Capture status.
-   Incoming flow count.
-   Recent flow predictions.
-   Alert severity and timestamps.
-   Latency and throughput.

Do not imply that monitoring is active when only static data is being
displayed.

## 8.4 Attack analytics page

Show:

-   Predicted attack distribution.
-   Per-class counts.
-   Confusion matrix when ground truth is available.
-   Class-wise precision, recall and F1.
-   Time-based trends when timestamps are available.

## 8.5 Anomaly detection page

Show:

-   Anomaly score distribution.
-   Threshold indicator.
-   Flagged records.
-   Top unusual feature values.
-   Known versus withheld attack evaluation results when in research
    mode.

## 8.6 Explainability page

Show:

-   Global SHAP summary.
-   Feature importance.
-   Local explanation for selected flow.
-   Plain-language explanation of top contributions.
-   Clear distinction between model explanation and security certainty.

## 8.7 Model performance page

Show:

-   Accuracy.
-   Macro-F1 and weighted-F1.
-   Per-class metrics.
-   Confusion matrix.
-   AUROC/AUPRC where valid.
-   Open-set metrics.
-   Calibration plot and ECE/Brier score where implemented.
-   Inference latency and throughput.
-   Experiment configuration and model version.

## 8.8 Alert history page

Show:

-   Timestamp.
-   Source/destination metadata only where appropriate.
-   Predicted category.
-   Confidence.
-   Anomaly score.
-   Final decision.
-   Explanation availability.
-   Export to CSV.

Protect sensitive network metadata. Do not expose packet payloads or
unnecessary personal data in the dashboard.

------------------------------------------------------------------------

# 9. Evaluation Methodology

## 9.1 Classification metrics

Report:

-   Accuracy.
-   Precision.
-   Recall.
-   Macro-F1.
-   Weighted-F1.
-   Per-class precision, recall and F1.
-   Confusion matrix.

Macro-F1 is important because attack categories may be imbalanced.

## 9.2 Anomaly and open-set metrics

Report:

-   Unknown-attack precision, recall and F1.
-   AUROC.
-   AUPRC.
-   False-positive rate on normal traffic.
-   False acceptance rate for unknown attacks classified as
    known/normal.
-   Detection rate by withheld attack family.

Define every metric mathematically and specify the positive class.

## 9.3 Operational metrics

Report:

-   Average and median inference latency.
-   95th-percentile latency where possible.
-   Throughput in flows/records per second.
-   Model file size.
-   Peak memory use where measurable.
-   Capture-to-alert latency for the live/replay pipeline.

Do not describe offline batch inference as live real-time performance.

## 9.4 Planned experiments

  -----------------------------------------------------------------------
  Experiment              Comparison              Purpose
  ----------------------- ----------------------- -----------------------
  E1                      Random Forest vs        Multiclass baseline
                          XGBoost                 

  E2                      CNN-LSTM vs tabular ML  Compare with reference
                                                  direction

  E3                      Isolation Forest vs     Anomaly detection
                          Autoencoder             

  E4                      Classifier alone vs     Benefit of fusion
                          hybrid                  

  E5                      Different confidence    Rejection behaviour
                          thresholds              

  E6                      Multiple withheld       Open-set generalization
                          attack families         

  E7                      With/without selected   Feature contribution
                          feature engineering     

  E8                      Batch vs replay/live    Operational performance
                          processing              
  -----------------------------------------------------------------------

Run repeated experiments with fixed and reported seeds where practical.
Report variability or confidence intervals when appropriate. Never
invent or pre-fill performance numbers.

------------------------------------------------------------------------

# 10. Research Contribution and Novelty Positioning

The project should not claim that CNN-LSTM, SHAP, Isolation Forest,
multiclass IDS or dashboards are individually novel.

The proposed contribution to investigate is:

**A confidence-aware hybrid decision mechanism that combines calibrated
multiclass predictions with a normal-only anomaly score to distinguish
known attacks, normal traffic, uncertain traffic and potentially unseen
attack traffic, evaluated under controlled withheld-attack protocols.**

Potential contributions to validate experimentally:

1.  A reproducible open-set evaluation protocol on UNSW-NB15 with
    multiple withheld attack families.
2.  A transparent fusion strategy combining calibrated class confidence
    and normal-only anomaly scores.
3.  An empirical analysis of the trade-off between unknown-attack recall
    and false alarms.
4.  A unified explanation and monitoring workflow for offline benchmark
    traffic and compatible replay/live flows.

These are proposed contributions, not guaranteed novelty claims. Conduct
a current literature review before asserting that the exact combination
is new. Compare the closest prior work's data splits, detectors, fusion
logic, thresholds and metrics.

------------------------------------------------------------------------

# 11. Literature Review and Research Paper

## 11.1 Literature review table

Create a CSV or Markdown table with:

-   Paper title.
-   Authors.
-   Year.
-   Venue.
-   Dataset.
-   Model/architecture.
-   Preprocessing.
-   Evaluation split.
-   Metrics and reported results.
-   Limitations stated by authors.
-   Our relevance.
-   DOI or official URL.

Prioritize peer-reviewed papers on:

-   UNSW-NB15 multiclass IDS.
-   CNN-LSTM IDS.
-   Open-set intrusion detection.
-   One-class and anomaly-based IDS.
-   Confidence calibration and selective classification.
-   Explainable IDS.
-   Real-time flow-based monitoring.
-   Dataset shift and cross-dataset generalization.

Do not invent paper titles, authors, DOIs, results or limitations.
Verify bibliographic information from publisher or author sources.

## 11.2 Proposed paper structure

1.  Abstract.
2.  Keywords.
3.  Introduction.
4.  Related Work.
5.  Problem Formulation and Research Gap.
6.  Dataset and Preprocessing.
7.  Proposed Confidence-Aware Hybrid Framework.
8.  Experimental Setup.
9.  Results.
10. Ablation and Error Analysis.
11. Discussion.
12. Limitations and Threats to Validity.
13. Conclusion and Future Work.
14. References.

Use IEEE-style formatting unless the target venue requires a different
template. Do not claim acceptance or guaranteed publication.

------------------------------------------------------------------------

# 12. Recommended Repository Structure

Create this structure:

``` text
Intelligent_NIDS/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── experiment_config.yaml
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_multiclass_models.ipynb
│   ├── 04_anomaly_detection.ipynb
│   ├── 05_open_set_evaluation.ipynb
│   └── 06_final_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── multiclass_classifier.py
│   ├── anomaly_detector.py
│   ├── confidence_calibration.py
│   ├── decision_fusion.py
│   ├── explainability.py
│   ├── evaluation.py
│   └── inference.py
│
├── traffic/
│   ├── packet_capture.py
│   ├── flow_extractor.py
│   ├── pcap_replay.py
│   └── schema_mapping.py
│
├── dashboard/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Data_Input.py
│   │   ├── 3_Live_Monitoring.py
│   │   ├── 4_Attack_Analytics.py
│   │   ├── 5_Anomaly_Detection.py
│   │   ├── 6_Explainability.py
│   │   ├── 7_Model_Performance.py
│   │   └── 8_Alert_History.py
│   └── components/
│
├── models/
│   ├── classifier/
│   ├── anomaly/
│   ├── calibration/
│   └── preprocessing/
│
├── experiments/
│   ├── configs/
│   ├── logs/
│   └── reports/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── confusion_matrices/
│   └── tables/
│
├── research_paper/
│   ├── literature_review/
│   ├── figures/
│   ├── tables/
│   └── paper.tex
│
└── tests/
    ├── test_preprocessing.py
    ├── test_inference.py
    ├── test_decision_fusion.py
    └── test_schema_validation.py
```

The structure can be adjusted if the actual implementation needs it. Do
not create empty files merely to match the tree; create modules when
they have a clear responsibility.

------------------------------------------------------------------------

# 13. Development Phases and Acceptance Criteria

Develop in the following order. Do not skip ahead when a phase is
failing.

## Phase 0 --- Environment and repository

Tasks:

-   Inspect the current workspace before creating files.
-   Check Python version and available hardware.
-   Create a virtual environment.
-   Create the repository structure.
-   Create `requirements.txt`.
-   Add a README and `.gitignore`.
-   Record package versions.

Acceptance criteria:

-   Project environment can be recreated.
-   Imports work.
-   README contains setup instructions.
-   No large datasets, secrets or generated model files are accidentally
    committed.

## Phase 1 --- Dataset understanding

Tasks:

-   Obtain the official dataset or guide the user to download it.
-   Inspect CSV files and schemas.
-   Generate data-quality and class-distribution reports.
-   Identify target columns and possible leakage columns.
-   Save a reproducible data-preparation configuration.

Acceptance criteria:

-   Dataset source and version documented.
-   Class counts and feature types reported.
-   Missing values, duplicates and invalid values analyzed.
-   No labels or target-derived fields leak into input features.

## Phase 2 --- Multiclass baseline

Tasks:

-   Implement preprocessing using training-only fit.
-   Train Random Forest.
-   Train XGBoost.
-   Save models and label mappings.
-   Evaluate on the untouched official test set.

Acceptance criteria:

-   Reproducible training script.
-   Saved model and preprocessing artifacts.
-   Classification report, confusion matrix and metrics.
-   Per-class results, not just accuracy.

## Phase 3 --- Anomaly detector

Tasks:

-   Build a normal-only training subset.
-   Train Isolation Forest.
-   Define score direction and threshold selection.
-   Evaluate on normal, known attack and withheld attack examples.

Acceptance criteria:

-   No attack labels or attack examples used to train the normal-only
    detector.
-   Threshold-selection protocol documented.
-   Unknown detection and false-alarm metrics reported.

## Phase 4 --- Fusion and open-set evaluation

Tasks:

-   Implement confidence calibration.
-   Implement decision fusion.
-   Compare classifier-only and hybrid results.
-   Test multiple withheld attack categories.
-   Conduct threshold sensitivity analysis.

Acceptance criteria:

-   Unit tests for every decision branch.
-   No test-set leakage.
-   Confusion matrix includes normal, known attack and unknown/uncertain
    outcomes as appropriate.
-   Trade-offs are clearly reported.

## Phase 5 --- Explainability

Tasks:

-   Generate global SHAP analysis.
-   Generate local explanations.
-   Add anomaly-specific feature deviation explanations.

Acceptance criteria:

-   Explanations can be generated for a selected prediction.
-   Feature names are readable.
-   Explanation outputs are saved or reproducible.
-   The UI distinguishes feature contribution from causal proof.

## Phase 6 --- Traffic processing

Tasks:

-   Implement PCAP replay.
-   Implement flow extraction.
-   Validate the extracted feature schema.
-   Add optional authorized live capture.
-   Connect the common inference interface.

Acceptance criteria:

-   Unsupported features are detected rather than fabricated.
-   Flow extraction is tested on known PCAP examples.
-   The pipeline reports incomplete or unsupported flows.
-   Live capture can be stopped safely.

## Phase 7 --- Dashboard

Tasks:

-   Build the pages specified above.
-   Connect them to saved models and inference services.
-   Add upload validation and error handling.
-   Add alert history and export.
-   Add clear mode labels: dataset, replay or live.

Acceptance criteria:

-   Dashboard launches from documented command.
-   All pages work with sample data.
-   No fake live metrics or fabricated predictions.
-   Model and preprocessing versions are visible.

## Phase 8 --- Research paper and final validation

Tasks:

-   Freeze final experimental configurations.
-   Re-run experiments from scripts.
-   Export result tables and plots.
-   Write methodology and results from actual outputs.
-   Verify all references.
-   Document limitations and reproducibility details.

Acceptance criteria:

-   Every reported result is traceable to an experiment log.
-   Paper claims match the implementation.
-   No fabricated metrics or citations.
-   Code, README, results and manuscript agree.

------------------------------------------------------------------------

# 14. Engineering and Quality Requirements

-   Use modular Python code with type hints where useful.
-   Add docstrings for public functions.
-   Keep training, inference, traffic capture and UI code separate.
-   Use configuration files rather than scattering constants through
    source code.
-   Set and record random seeds.
-   Save model metadata, feature order, label mapping and preprocessing
    version together.
-   Validate input schema before prediction.
-   Handle missing columns, extra columns, invalid types and NaN/Inf
    values explicitly.
-   Never fit preprocessing on the test set.
-   Avoid data leakage through duplicate records, identifiers,
    timestamps or target-derived features.
-   Add unit tests for preprocessing, inference, fusion and schema
    validation.
-   Log exceptions clearly; do not silently return normal predictions
    after errors.
-   Do not place credentials, private IP information or sensitive
    captures in Git.
-   Use safe file-upload handling and reasonable upload-size limits.
-   Make the dashboard usable on a laptop screen.
-   Ensure the system can run without a GPU for the main pipeline.

------------------------------------------------------------------------

# 15. Instructions to Antigravity: How to Work on This Project

Act as a senior ML engineer, cybersecurity researcher and careful
software architect.

**Work incrementally. Do not generate the entire project in one
unverified step.**

1.  Inspect the existing workspace and report what is already present.
2.  Read this specification fully.
3.  Propose a short implementation plan for the current phase.
4.  Implement only the current phase.
5.  Run the code and tests.
6.  Inspect errors and fix them.
7.  Summarize files created or modified, commands run, test results and
    remaining issues.
8.  Wait for approval before moving to a major subsequent phase.

Additional rules:

-   Do not invent dataset files or pretend downloads succeeded.
-   Do not use random synthetic data as a substitute for benchmark
    results. Synthetic data may be used only for clearly labelled unit
    tests and UI previews.
-   Do not fabricate research papers, citations, metrics or experimental
    outcomes.
-   Do not silently change the proposed research problem.
-   Explain major methodological decisions and data-leakage risks.
-   Prefer simple, maintainable implementations over unnecessary
    complexity.
-   Keep a `PROJECT_STATUS.md` file updated with completed work, current
    phase, known issues and next steps.
-   Keep an `EXPERIMENT_LOG.md` with experiment ID, date, dataset split,
    configuration, random seed, model version and result artifact paths.
-   If a design choice is scientifically questionable, explain the issue
    and propose a defensible alternative before implementing it.
-   Never claim that an unknown attack is a confirmed zero-day attack.
-   Never claim live monitoring works until packet capture, flow
    extraction, feature compatibility and inference have all been
    tested.

------------------------------------------------------------------------

# 16. First Task: Start with Phase 0 and Phase 1 Only

When this specification is provided, do not immediately build the full
dashboard or train every model.

First:

1.  Inspect the workspace.
2.  Check Python and hardware.
3.  Create or adapt the project structure.
4.  Create the environment and dependencies.
5.  Create README and status tracking files.
6.  Check whether UNSW-NB15 files already exist.
7.  If they do not exist, provide the official download instructions; do
    not invent local dataset paths.
8.  Inspect the dataset once available.
9.  Produce a dataset summary and research-methodology risks report.

Then stop and present:

-   Workspace summary.
-   Environment setup results.
-   Files created.
-   Dataset status.
-   Dataset schema and class distribution if available.
-   Potential leakage and split concerns.
-   Recommended next implementation step.

**Do not start Phase 2 until the dataset and preprocessing plan have
been reviewed.**

------------------------------------------------------------------------

# 17. Useful Official Resources

-   UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
-   CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
-   Scikit-learn: https://scikit-learn.org/stable/
-   XGBoost: https://xgboost.readthedocs.io/
-   SHAP: https://shap.readthedocs.io/
-   Streamlit: https://docs.streamlit.io/
-   Scapy: https://scapy.readthedocs.io/
-   PyTorch: https://pytorch.org/docs/stable/

------------------------------------------------------------------------

## Final Project Goal

Deliver a reproducible, research-oriented and working Intelligent
Network Intrusion Detection System that combines multiclass
classification, normal-only anomaly detection, confidence-aware open-set
decisions, explainability and traffic monitoring.

The final project must include both:

1.  **A functional application** demonstrating the detection pipeline.
2.  **A scientifically defensible research paper** supported by
    reproducible experiments, comparisons, ablations and honest
    limitations.

Build the research contribution first, validate it rigorously, and then
demonstrate it through the dashboard.
