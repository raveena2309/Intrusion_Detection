"""
Train Multiclass Classifiers – Phase 2
======================================

Trains Random Forest and XGBoost on the official UNSW-NB15 train/test split,
evaluates on the held-out test set, and persists models + metrics.
"""

import sys
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import NIDSPreprocessor, load_config
from multiclass_classifier import (
    train_random_forest,
    train_xgboost,
    evaluate_classifier,
    save_classifier,
    print_comparison_table,
    ClassifierResult,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_CFG = load_config("data_config.yaml")
MODEL_CFG = load_config("model_config.yaml")

TRAIN_CSV = PROJECT_ROOT / DATA_CFG["dataset"]["paths"]["raw_dir"] / DATA_CFG["dataset"]["expected_files"]["training"]
TEST_CSV  = PROJECT_ROOT / DATA_CFG["dataset"]["paths"]["raw_dir"] / DATA_CFG["dataset"]["expected_files"]["testing"]

METRICS_DIR   = PROJECT_ROOT / "results" / "metrics"
CM_DIR        = PROJECT_ROOT / "results" / "confusion_matrices"
MODEL_DIR     = PROJECT_ROOT / "models" / "classifier"
PREPROC_DIR   = PROJECT_ROOT / "models" / "preprocessing"

for d in [METRICS_DIR, CM_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
print("Loading raw datasets …")
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
print(f"  Train shape: {df_train.shape}")
print(f"  Test  shape: {df_test.shape}")


# ---------------------------------------------------------------------------
# Preprocess  (fit on training only)
# ---------------------------------------------------------------------------
print("\nFitting NIDSPreprocessor on training data …")
preprocessor = NIDSPreprocessor(config=DATA_CFG)
X_train = preprocessor.fit_transform(df_train)
y_train = preprocessor.get_labels(df_train, label_type="multiclass")

X_test = preprocessor.transform(df_test)
y_test = preprocessor.get_labels(df_test, label_type="multiclass")

label_names: list = preprocessor.label_encoder.classes_.tolist()
feature_names: list = preprocessor.feature_names

print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  Classes ({len(label_names)}): {label_names}")

# Save / overwrite preprocessor
preprocessor.save(str(PREPROC_DIR))


# ---------------------------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------------------------
print("\n[1/2] Training Random Forest …")
t0 = time.time()
rf_model = train_random_forest(X_train, y_train, config=MODEL_CFG)
rf_time = time.time() - t0
print(f"  Done in {rf_time:.1f}s")

rf_result = evaluate_classifier(
    rf_model, X_test, y_test,
    label_names=label_names,
    model_name="Random Forest",
    feature_names=feature_names,
    training_time=rf_time,
)

print(f"  Accuracy   : {rf_result.accuracy:.4f}")
print(f"  Macro-F1   : {rf_result.macro_f1:.4f}")
print(f"  Weighted-F1: {rf_result.weighted_f1:.4f}")

save_classifier(
    rf_model,
    model_name="random_forest",
    metadata={
        "model_name": "Random Forest",
        "accuracy": rf_result.accuracy,
        "macro_f1": rf_result.macro_f1,
        "weighted_f1": rf_result.weighted_f1,
        "training_time_s": rf_time,
        "n_estimators": 200,
        "class_weight": "balanced",
        "random_state": 42,
        "label_names": label_names,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    },
    directory=str(MODEL_DIR),
)


# ---------------------------------------------------------------------------
# Train XGBoost
# ---------------------------------------------------------------------------
print("\n[2/2] Training XGBoost …")
t0 = time.time()
xgb_model = train_xgboost(X_train, y_train, config=MODEL_CFG)
xgb_time = time.time() - t0
print(f"  Done in {xgb_time:.1f}s")

xgb_result = evaluate_classifier(
    xgb_model, X_test, y_test,
    label_names=label_names,
    model_name="XGBoost",
    feature_names=feature_names,
    training_time=xgb_time,
)

print(f"  Accuracy   : {xgb_result.accuracy:.4f}")
print(f"  Macro-F1   : {xgb_result.macro_f1:.4f}")
print(f"  Weighted-F1: {xgb_result.weighted_f1:.4f}")

save_classifier(
    xgb_model,
    model_name="xgboost",
    metadata={
        "model_name": "XGBoost",
        "accuracy": xgb_result.accuracy,
        "macro_f1": xgb_result.macro_f1,
        "weighted_f1": xgb_result.weighted_f1,
        "training_time_s": xgb_time,
        "n_estimators": 300,
        "objective": "multi:softprob",
        "random_state": 42,
        "label_names": label_names,
        "feature_names": feature_names,
        "n_features": len(feature_names),
    },
    directory=str(MODEL_DIR),
)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
results = [rf_result, xgb_result]
print_comparison_table(results)


# ---------------------------------------------------------------------------
# Save classifier_results.yaml
# ---------------------------------------------------------------------------
def _per_class_to_serializable(report: dict) -> dict:
    """Convert classification_report dict values to plain Python floats."""
    out = {}
    for k, v in report.items():
        if isinstance(v, dict):
            out[k] = {kk: float(vv) if hasattr(vv, "item") else vv for kk, vv in v.items()}
        else:
            out[k] = float(v) if hasattr(v, "item") else v
    return out


metrics_payload = {}
for res in results:
    key = res.model_name.lower().replace(" ", "_")
    metrics_payload[key] = {
        "accuracy": float(res.accuracy),
        "macro_f1": float(res.macro_f1),
        "weighted_f1": float(res.weighted_f1),
        "training_time_s": float(res.training_time),
        "per_class_report": _per_class_to_serializable(res.per_class_report),
    }

metrics_path = METRICS_DIR / "classifier_results.yaml"
with open(metrics_path, "w") as fh:
    yaml.dump(metrics_payload, fh, default_flow_style=False, sort_keys=False)
print(f"\nMetrics saved → {metrics_path}")


# ---------------------------------------------------------------------------
# Save confusion matrices
# ---------------------------------------------------------------------------
for res in results:
    key = res.model_name.lower().replace(" ", "_")
    cm_path = CM_DIR / f"{key}_confusion_matrix.csv"
    cm_df = pd.DataFrame(
        res.confusion_matrix,
        index=res.label_names,
        columns=res.label_names,
    )
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved → {cm_path}")

print("\nPhase 2 complete.")
