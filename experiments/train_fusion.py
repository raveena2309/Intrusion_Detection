"""
experiments/train_fusion.py
============================

Phase 4 experiment: confidence calibration + decision fusion evaluation.

Pipeline
--------
1. Load pre-trained classifier and anomaly detector from models/.
2. Split training set: 70% train (already used) / 15% val / 15% held-out.
3. Calibrate classifier probabilities on the validation split.
4. Select confidence and anomaly thresholds on the validation split.
5. Evaluate four conditions on the held-out split:
   a) classifier-only
   b) anomaly-only
   c) simple-fusion (default thresholds)
   d) tuned-fusion  (thresholds from step 4)
6. Open-set evaluation: Worms is withheld from the classifier test pool.
7. Save all results to results/metrics/fusion_results.yaml.
"""

from __future__ import annotations

import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from confidence_calibration import (
    calibrate_classifier,
    compute_ece,
    compute_brier_score,
    plot_reliability_diagram,
    save_calibrator,
)
from decision_fusion import DecisionFusion
from evaluation import (
    compute_openset_metrics,
    compute_operational_metrics,
    plot_confusion_matrix,
    plot_auroc_curve,
)
from preprocessing import NIDSPreprocessor

RANDOM_SEED = 42
NORMAL_CLASS_IDX = 6
WITHHELD_CLASSES = ["Worms"]

LABEL_NAMES = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
]


def _check_prerequisites():
    clf_dir = PROJECT_ROOT / "models" / "classifier"
    anom_dir = PROJECT_ROOT / "models" / "anomaly"
    clf_meta = list(clf_dir.glob("*.yaml")) if clf_dir.exists() else []
    anom_meta = list(anom_dir.glob("*.yaml")) if anom_dir.exists() else []
    missing = []
    if not clf_meta:
        missing.append("models/classifier/ (no metadata yaml found)")
    if not anom_meta:
        missing.append("models/anomaly/ (no metadata yaml found)")
    if missing:
        return False, "Missing: " + ", ".join(missing)
    return True, "OK"


def load_classifier(clf_dir):
    for prefix in ("xgboost", "random_forest", ""):
        candidates = list(clf_dir.glob(f"{prefix}*.joblib"))
        if candidates:
            model = joblib.load(candidates[0])
            print(f"[train_fusion] Classifier loaded: {candidates[0].name}")
            return model
    raise FileNotFoundError("No classifier .joblib found")


def load_anomaly_detector(anom_dir):
    candidates = list(anom_dir.glob("*.joblib"))
    if not candidates:
        raise FileNotFoundError("No anomaly detector .joblib found")
    model = joblib.load(candidates[0])
    print(f"[train_fusion] Anomaly detector loaded: {candidates[0].name}")
    return model


def load_data():
    preprocessor = NIDSPreprocessor.load(str(PROJECT_ROOT / "models" / "preprocessing"))
    print("[train_fusion] Preprocessor loaded")

    train_csv = PROJECT_ROOT / "datasets" / "raw" / "UNSW_NB15_training-set.csv"
    test_csv = PROJECT_ROOT / "datasets" / "raw" / "UNSW_NB15_testing-set.csv"

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    X_train = preprocessor.transform(df_train)
    y_train = preprocessor.get_labels(df_train, "multiclass")
    X_test = preprocessor.transform(df_test)
    y_test = preprocessor.get_labels(df_test, "multiclass")

    return X_train, y_train, X_test, y_test


def get_anomaly_scores(anomaly_model, X):
    if hasattr(anomaly_model, "decision_function"):
        return -anomaly_model.decision_function(X)
    elif hasattr(anomaly_model, "score_samples"):
        return -anomaly_model.score_samples(X)
    else:
        raise AttributeError("Anomaly model has no decision_function or score_samples")


def eval_classifier_only(clf, X_test, y_test, label_names):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    prfs = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(label_names))))
    recs = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(label_names))))
    f1s = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(label_names))))
    report = {}
    for i, cls in enumerate(label_names):
        report[cls] = {"precision": float(prfs[i]), "recall": float(recs[i]), "f1": float(f1s[i])}
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "per_class": report}


def eval_anomaly_only(anomaly_model, X_test, y_test, threshold, normal_class_idx=6):
    scores = get_anomaly_scores(anomaly_model, X_test)
    y_pred_binary = (scores > threshold).astype(int)
    y_true_binary = (y_test != normal_class_idx).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "threshold": float(threshold),
    }


def main():
    print("=" * 70)
    print("Phase 4 - Decision Fusion Experiment")
    print("=" * 70)

    ok, msg = _check_prerequisites()
    if not ok:
        print(f"\n[PENDING] Models not yet trained. {msg}")
        sys.exit(0)

    clf_dir = PROJECT_ROOT / "models" / "classifier"
    anom_dir = PROJECT_ROOT / "models" / "anomaly"
    clf = load_classifier(clf_dir)
    anomaly_model = load_anomaly_detector(anom_dir)

    X_train_full, y_train_full, X_test, y_test = load_data()

    # Split training -> val (15%) + held (15%)
    X_rest, _, y_rest, _ = train_test_split(
        X_train_full, y_train_full, test_size=0.70,
        random_state=RANDOM_SEED, stratify=y_train_full,
    )
    X_val, X_held, y_val, y_held = train_test_split(
        X_rest, y_rest, test_size=0.50,
        random_state=RANDOM_SEED, stratify=y_rest,
    )
    print(f"[train_fusion] Val size: {len(X_val)} | Held size: {len(X_held)}")

    # Calibrate classifier
    print("\n--- Calibrating classifier ---")
    try:
        calibrated_clf = calibrate_classifier(clf, X_val, y_val, method="isotonic")
        save_calibrator(calibrated_clf, str(PROJECT_ROOT / "models" / "calibration"))
        y_prob_val = calibrated_clf.predict_proba(X_val)
        ece = compute_ece((y_val == NORMAL_CLASS_IDX).astype(int), y_prob_val[:, NORMAL_CLASS_IDX])
        brier = compute_brier_score(y_val, y_prob_val)
        print(f"[calibration] ECE={ece:.4f}  Brier={brier:.4f}")
        plot_reliability_diagram(
            y_val, y_prob_val, LABEL_NAMES,
            save_path=str(PROJECT_ROOT / "results" / "figures" / "reliability_diagram.png"),
        )
    except Exception as exc:
        print(f"[calibration] WARNING: calibration failed ({exc}). Using raw classifier.")
        calibrated_clf = clf
        ece = None
        brier = None

    # Anomaly scores
    print("\n--- Computing anomaly scores ---")
    anom_scores_val = get_anomaly_scores(anomaly_model, X_val)
    anom_scores_held = get_anomaly_scores(anomaly_model, X_held)
    anom_scores_test = get_anomaly_scores(anomaly_model, X_test)

    normal_val_mask = y_val == NORMAL_CLASS_IDX
    default_anom_threshold = float(
        np.percentile(anom_scores_val[normal_val_mask], 95)
        if normal_val_mask.sum() > 0
        else np.percentile(anom_scores_val, 80)
    )
    print(f"[fusion] Default anomaly threshold (95th pct normal): {default_anom_threshold:.4f}")

    # Select thresholds on validation set
    print("\n--- Selecting thresholds on validation set ---")
    clf_probs_val = calibrated_clf.predict_proba(X_val)
    fusion = DecisionFusion(confidence_threshold=0.6, anomaly_threshold=default_anom_threshold)
    best_conf, best_anom = fusion.select_thresholds(
        clf_probs_val, anom_scores_val, y_val,
        normal_class_idx=NORMAL_CLASS_IDX, label_names=LABEL_NAMES,
    )
    fusion.save(str(PROJECT_ROOT / "models" / "calibration"))

    # Evaluate four conditions on held-out set
    print("\n--- Evaluating four conditions on held-out set ---")
    results = {}

    # (a) Classifier-only
    clf_only = eval_classifier_only(calibrated_clf, X_held, y_held, LABEL_NAMES)
    results["classifier_only"] = clf_only
    print(f"  (a) Classifier-only  acc={clf_only['accuracy']:.4f}  macro_f1={clf_only['macro_f1']:.4f}")

    # (b) Anomaly-only
    anom_only = eval_anomaly_only(anomaly_model, X_held, y_held, default_anom_threshold)
    results["anomaly_only"] = anom_only
    print(f"  (b) Anomaly-only     acc={anom_only['accuracy']:.4f}  f1={anom_only['f1']:.4f}")

    # (c) Simple fusion
    simple_fusion = DecisionFusion(confidence_threshold=0.6, anomaly_threshold=default_anom_threshold)
    simple_res = simple_fusion.evaluate_fusion(
        X_held, y_held, calibrated_clf, anomaly_model,
        anomaly_threshold=default_anom_threshold, conf_threshold=0.6,
        label_names=LABEL_NAMES,
    )
    results["simple_fusion"] = {k: v for k, v in simple_res.items() if k != "decisions"}
    print(f"  (c) Simple-fusion    known_acc={simple_res.get('known_accuracy', 0):.4f}  "
          f"unknown_recall={simple_res.get('unknown_recall', 0):.4f}  "
          f"fpr={simple_res.get('false_positive_rate', 0):.4f}")

    # (d) Tuned fusion
    tuned_res = fusion.evaluate_fusion(
        X_held, y_held, calibrated_clf, anomaly_model,
        anomaly_threshold=best_anom, conf_threshold=best_conf,
        label_names=LABEL_NAMES,
    )
    results["tuned_fusion"] = {k: v for k, v in tuned_res.items() if k != "decisions"}
    print(f"  (d) Tuned-fusion     known_acc={tuned_res.get('known_accuracy', 0):.4f}  "
          f"unknown_recall={tuned_res.get('unknown_recall', 0):.4f}  "
          f"fpr={tuned_res.get('false_positive_rate', 0):.4f}")

    # Open-set evaluation: Worms withheld
    print("\n--- Open-set evaluation (Worms withheld) ---")
    y_test_str = np.array([LABEL_NAMES[i] for i in y_test])
    clf_probs_test = calibrated_clf.predict_proba(X_test)
    clf_preds_test = np.argmax(clf_probs_test, axis=1)
    test_decisions = []
    for i in range(len(X_test)):
        r = fusion.fuse(
            int(clf_preds_test[i]), clf_probs_test[i],
            float(anom_scores_test[i]), LABEL_NAMES, NORMAL_CLASS_IDX,
        )
        test_decisions.append(r.final_decision)

    openset_metrics = compute_openset_metrics(
        y_true=y_test_str,
        y_pred_fusion=np.array(test_decisions),
        withheld_class_names=WITHHELD_CLASSES,
    )
    results["openset_worms"] = openset_metrics
    print(f"  Worms unknown_recall   : {openset_metrics.get('unknown_recall', 0):.4f}")
    print(f"  False positive rate    : {openset_metrics.get('false_positive_rate', 0):.4f}")
    print(f"  Precision unknown      : {openset_metrics.get('precision_unknown', 0):.4f}")
    print(f"  #Withheld samples      : {openset_metrics.get('n_withheld_samples', 0)}")

    # Operational metrics
    print("\n--- Operational metrics (classifier) ---")
    op_metrics = compute_operational_metrics(calibrated_clf, X_test, batch_size=1000)
    results["operational"] = op_metrics
    print(f"  Throughput: {op_metrics['throughput_samples_per_sec']:.0f} samples/sec")
    print(f"  P95 latency/sample: {op_metrics['p95_latency_ms']:.4f} ms")

    # Confusion matrix for tuned fusion
    print("\n--- Generating confusion matrix ---")
    decision_map = {name: i for i, name in enumerate(LABEL_NAMES)}
    decision_map["unknown_suspicious"] = len(LABEL_NAMES)
    decision_map["uncertain_review"] = len(LABEL_NAMES) + 1
    decision_map["normal"] = NORMAL_CLASS_IDX

    tuned_decisions_held = tuned_res.get("decisions", [])
    if tuned_decisions_held:
        y_pred_cm = np.array([decision_map.get(d, NORMAL_CLASS_IDX) for d in tuned_decisions_held])
        known_mask_cm = y_pred_cm < len(LABEL_NAMES)
        if known_mask_cm.sum() > 0:
            cm = confusion_matrix(
                y_held[known_mask_cm], y_pred_cm[known_mask_cm],
                labels=list(range(len(LABEL_NAMES))),
            )
            plot_confusion_matrix(
                cm, LABEL_NAMES,
                title="Tuned Fusion - Held-out Confusion Matrix",
                save_path=str(PROJECT_ROOT / "results" / "figures" / "fusion_confusion_matrix.png"),
            )

    # AUROC
    y_true_binary_test = (y_test != NORMAL_CLASS_IDX).astype(int)
    auroc = plot_auroc_curve(
        y_true_binary_test, anom_scores_test,
        save_path=str(PROJECT_ROOT / "results" / "figures" / "anomaly_auroc.png"),
        title="Anomaly Detector AUROC",
    )
    results["anomaly_auroc"] = float(auroc)
    print(f"  Anomaly AUROC: {auroc:.4f}")

    if ece is not None:
        results["calibration"] = {"ece": float(ece), "brier_score": float(brier), "method": "isotonic"}

    # Save results
    out_path = PROJECT_ROOT / "results" / "metrics" / "fusion_results.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    print(f"\n[train_fusion] Results saved to {out_path}")
    print("=" * 70)
    print("Phase 4 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
