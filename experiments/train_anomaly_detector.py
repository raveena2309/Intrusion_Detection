"""
Train Anomaly Detector – Phase 3
=================================

Trains an Isolation Forest on normal-only training traffic, selects a decision
threshold using a held-out normal validation split, then evaluates on the full
test set (including withheld Worms class).
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
from anomaly_detector import (
    train_isolation_forest,
    compute_anomaly_scores,
    select_threshold,
    evaluate_anomaly_detector,
    save_anomaly_detector,
    load_anomaly_detector,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_CFG  = load_config("data_config.yaml")
MODEL_CFG = load_config("model_config.yaml")

TRAIN_CSV = PROJECT_ROOT / DATA_CFG["dataset"]["paths"]["raw_dir"] / DATA_CFG["dataset"]["expected_files"]["training"]
TEST_CSV  = PROJECT_ROOT / DATA_CFG["dataset"]["paths"]["raw_dir"] / DATA_CFG["dataset"]["expected_files"]["testing"]

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
MODEL_DIR   = PROJECT_ROOT / "models" / "anomaly"
PREPROC_DIR = PROJECT_ROOT / "models" / "preprocessing"

for d in [METRICS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NORMAL_VAL_FRACTION = 0.15  # 15 % of normal training records → threshold selection
RANDOM_SEED         = 42

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
print("Loading raw datasets …")
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
print(f"  Train shape : {df_train.shape}")
print(f"  Test  shape : {df_test.shape}")

# ---------------------------------------------------------------------------
# Preprocess  (fit on training only)
# ---------------------------------------------------------------------------
print("\nFitting NIDSPreprocessor on training data …")
preprocessor = NIDSPreprocessor(config=DATA_CFG)
X_train_full = preprocessor.fit_transform(df_train)

# Binary labels for train
y_train_binary = df_train["label"].values.astype(int)
# Multiclass labels for test
y_test_binary  = df_test["label"].values.astype(int)
attack_cat_test = df_test["attack_cat"].astype(str).str.strip().values

X_test = preprocessor.transform(df_test)

label_names: list = preprocessor.label_encoder.classes_.tolist()
print(f"  X_train_full: {X_train_full.shape}  |  X_test: {X_test.shape}")

# ---------------------------------------------------------------------------
# Split normal training records into fit / validation sets
# ---------------------------------------------------------------------------
normal_mask_train = y_train_binary == 0
X_normal_all = X_train_full[normal_mask_train]
print(f"\nNormal training records : {X_normal_all.shape[0]}")

rng = np.random.default_rng(RANDOM_SEED)
n_val = max(1, int(NORMAL_VAL_FRACTION * len(X_normal_all)))
val_indices = rng.choice(len(X_normal_all), size=n_val, replace=False)
fit_indices = np.setdiff1d(np.arange(len(X_normal_all)), val_indices)

X_normal_fit = X_normal_all[fit_indices]
X_normal_val = X_normal_all[val_indices]

print(f"  Normal fit records     : {X_normal_fit.shape[0]}")
print(f"  Normal val records     : {X_normal_val.shape[0]}")

# ---------------------------------------------------------------------------
# Train Isolation Forest
# ---------------------------------------------------------------------------
print("\nTraining Isolation Forest on normal-only data …")
t0 = time.time()
if_model = train_isolation_forest(X_normal_fit, config=MODEL_CFG)
train_time = time.time() - t0
print(f"  Done in {train_time:.1f}s")

# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
TARGET_FPR = 0.05
threshold = select_threshold(if_model, X_normal_val, target_fpr=TARGET_FPR)
print(f"\nSelected threshold (target FPR={TARGET_FPR:.0%}): {threshold:.6f}")

# ---------------------------------------------------------------------------
# Evaluate on full test set
# ---------------------------------------------------------------------------
print("\nEvaluating on full test set …")
eval_full = evaluate_anomaly_detector(
    if_model,
    threshold=threshold,
    X=X_test,
    y_binary=y_test_binary,
    label_names_per_sample=attack_cat_test.tolist(),
)

print(f"  AUROC         : {eval_full['auroc']:.4f}")
print(f"  AUPRC         : {eval_full['auprc']:.4f}")
print(f"  Precision     : {eval_full['precision']:.4f}")
print(f"  Recall        : {eval_full['recall']:.4f}")
print(f"  F1            : {eval_full['f1']:.4f}")
print(f"  FPR on Normal : {eval_full['fpr_on_normal']:.4f}")
print(f"  Samples       : {eval_full['n_total']} total  |  {eval_full['n_normal']} normal  |  {eval_full['n_attacks']} attacks")

if "per_class_detection_rate" in eval_full:
    print("\n  Per-class detection rate:")
    for cls, rate in sorted(eval_full["per_class_detection_rate"].items()):
        print(f"    {cls:<20} {rate:.4f}")

# ---------------------------------------------------------------------------
# Withheld-attack evaluation: Worms (index 9)
# ---------------------------------------------------------------------------
WITHHELD_CLASS = "Worms"
print(f"\nEvaluating specifically on withheld class: {WITHHELD_CLASS}")

# Keep only normal + worms from test set
worms_mask = attack_cat_test == WITHHELD_CLASS
normal_mask_test = y_test_binary == 0

subset_mask = normal_mask_test | worms_mask
X_subset    = X_test[subset_mask]
y_subset    = y_test_binary[subset_mask]
cats_subset = attack_cat_test[subset_mask]

print(f"  Normal in subset : {int((y_subset == 0).sum())}")
print(f"  Worms in subset  : {int((y_subset == 1).sum())}")

eval_worms = evaluate_anomaly_detector(
    if_model,
    threshold=threshold,
    X=X_subset,
    y_binary=y_subset,
    label_names_per_sample=cats_subset.tolist(),
)

print(f"  AUROC  : {eval_worms['auroc']:.4f}")
print(f"  Recall : {eval_worms['recall']:.4f}")
print(f"  F1     : {eval_worms['f1']:.4f}")
print(f"  FPR    : {eval_worms['fpr_on_normal']:.4f}")

# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------
save_anomaly_detector(
    if_model,
    threshold=threshold,
    metadata={
        "model": "IsolationForest",
        "n_estimators": 200,
        "random_state": RANDOM_SEED,
        "normal_fit_records": int(X_normal_fit.shape[0]),
        "normal_val_records": int(X_normal_val.shape[0]),
        "target_fpr": TARGET_FPR,
        "training_time_s": float(train_time),
        "n_features": int(X_train_full.shape[1]),
        "feature_names": preprocessor.feature_names,
    },
    directory=str(MODEL_DIR),
)

# ---------------------------------------------------------------------------
# Save results/metrics/anomaly_results.yaml
# ---------------------------------------------------------------------------
def _floatify(d: dict) -> dict:
    """Recursively convert numpy scalars to native Python types."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _floatify(v)
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out

anomaly_metrics = {
    "threshold": float(threshold),
    "target_fpr": TARGET_FPR,
    "training_time_s": float(train_time),
    "full_test_set": _floatify(eval_full),
    "withheld_worms_subset": _floatify(eval_worms),
}

metrics_path = METRICS_DIR / "anomaly_results.yaml"
with open(metrics_path, "w") as fh:
    yaml.dump(anomaly_metrics, fh, default_flow_style=False, sort_keys=False)

print(f"\nAnomalY results saved → {metrics_path}")
print("\nPhase 3 complete.")
