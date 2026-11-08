"""
Anomaly Detector Module for Intelligent NIDS
=============================================

Implements an Isolation Forest-based anomaly detector trained exclusively on
*normal* traffic.  Provides threshold selection, evaluation, and persistence
utilities suitable for open-set / unknown-attack detection.
"""

import yaml
import joblib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_normal_train: np.ndarray,
    config: Optional[dict] = None,
) -> IsolationForest:
    """Train an Isolation Forest on normal-class traffic only.

    The model learns the distribution of benign traffic; anomaly scores are
    later inverted so that *higher score -> more anomalous*.

    Args:
        X_normal_train: Feature matrix containing **only** normal-class records.
        config: Optional model configuration dict.  Parameters are read from
                ``config['anomaly_detectors']['isolation_forest']``.

    Returns:
        Fitted :class:`~sklearn.ensemble.IsolationForest` instance.
    """
    if_cfg: dict = {}
    if config is not None:
        if_cfg = config.get("anomaly_detectors", {}).get("isolation_forest", {})

    n_estimators = if_cfg.get("n_estimators", 200)
    contamination = if_cfg.get("contamination", "auto")
    max_samples = if_cfg.get("max_samples", "auto")
    n_jobs = if_cfg.get("n_jobs", -1)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(X_normal_train)
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
) -> np.ndarray:
    """Compute anomaly scores for every sample in *X*.

    Isolation Forest's ``score_samples`` returns *negative* average path
    lengths (more negative -> more anomalous).  We negate the values so that
    **higher score = more anomalous**, which is intuitive for downstream use.

    Args:
        model: Fitted :class:`~sklearn.ensemble.IsolationForest`.
        X: Feature matrix to score.

    Returns:
        1-D float array of anomaly scores (shape ``(n_samples,)``).
        Higher values indicate more anomalous samples.
    """
    # score_samples: higher (less negative) -> more normal -> negate to flip
    return -model.score_samples(X)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(
    model: IsolationForest,
    X_normal_val: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Select an anomaly-score threshold that achieves a target FPR on normal data.

    The threshold is chosen as the ``(1 - target_fpr)``-quantile of anomaly
    scores computed on a held-out set of *normal* samples.  At this threshold
    at most ``target_fpr`` of normal traffic is flagged as anomalous.

    Args:
        model: Fitted :class:`~sklearn.ensemble.IsolationForest`.
        X_normal_val: Validation feature matrix containing **only** normal records.
        target_fpr: Desired false-positive rate on normal traffic (default 5 %).

    Returns:
        Float threshold value.  Samples whose anomaly score exceeds this
        value are classified as attacks.
    """
    scores = compute_anomaly_scores(model, X_normal_val)
    threshold = float(np.quantile(scores, 1.0 - target_fpr))
    return threshold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_anomaly_detector(
    model: IsolationForest,
    threshold: float,
    X: np.ndarray,
    y_binary: np.ndarray,
    label_names_per_sample: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate the anomaly detector on a mixed (normal + attack) dataset.

    Args:
        model: Fitted :class:`~sklearn.ensemble.IsolationForest`.
        threshold: Decision threshold returned by :func:`select_threshold`.
        X: Feature matrix (normal + attack records mixed).
        y_binary: Binary ground-truth labels: ``0`` = normal, ``1`` = attack.
        label_names_per_sample: Optional per-sample attack-category strings
            used to compute per-class detection rates.

    Returns:
        Dictionary with the following keys:

        * ``auroc`` - Area under ROC curve.
        * ``auprc`` - Area under precision-recall curve.
        * ``precision`` - Precision at the selected threshold.
        * ``recall`` - Recall (TPR) at the selected threshold.
        * ``f1`` - F1 score at the selected threshold.
        * ``fpr_on_normal`` - FPR measured on normal-only samples.
        * ``threshold`` - The threshold value used.
        * ``n_total`` - Total number of evaluated samples.
        * ``n_attacks`` - Number of attack samples.
        * ``n_normal`` - Number of normal samples.
        * ``per_class_detection_rate`` - Dict mapping category name -> detection
          rate (only when *label_names_per_sample* is provided).
    """
    scores = compute_anomaly_scores(model, X)
    y_pred = (scores >= threshold).astype(int)

    # Guard against single-class edge case
    unique_classes = np.unique(y_binary)
    if len(unique_classes) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_binary, scores))
        auprc = float(average_precision_score(y_binary, scores))

    precision = float(precision_score(y_binary, y_pred, zero_division=0))
    recall = float(recall_score(y_binary, y_pred, zero_division=0))
    f1 = float(f1_score(y_binary, y_pred, zero_division=0))

    # FPR on normal traffic specifically
    normal_mask = y_binary == 0
    fpr_on_normal = float(y_pred[normal_mask].mean()) if normal_mask.any() else float("nan")

    n_total = len(y_binary)
    n_attacks = int(y_binary.sum())
    n_normal = int((y_binary == 0).sum())

    result: Dict[str, Any] = {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr_on_normal": fpr_on_normal,
        "threshold": threshold,
        "n_total": n_total,
        "n_attacks": n_attacks,
        "n_normal": n_normal,
    }

    # Per-class detection rate
    if label_names_per_sample is not None:
        label_arr = np.array(label_names_per_sample)
        per_class: Dict[str, float] = {}
        for cls in np.unique(label_arr):
            mask = label_arr == cls
            if mask.sum() == 0:
                continue
            detected = int(y_pred[mask].sum())
            total_cls = int(mask.sum())
            per_class[str(cls)] = round(detected / total_cls, 4)
        result["per_class_detection_rate"] = per_class

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_anomaly_detector(
    model: IsolationForest,
    threshold: float,
    metadata: dict,
    directory: str,
) -> None:
    """Persist the anomaly detector model and threshold to disk.

    Args:
        model: Fitted :class:`~sklearn.ensemble.IsolationForest`.
        threshold: Selected decision threshold.
        metadata: Arbitrary key-value metadata dict.
        directory: Destination directory; created automatically if absent.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = dir_path / "isolation_forest.joblib"
    joblib.dump(model, model_path)

    # Save threshold + metadata
    payload = {"threshold": float(threshold), **metadata}
    meta_path = dir_path / "anomaly_metadata.yaml"
    with open(meta_path, "w") as fh:
        yaml.dump(payload, fh, default_flow_style=False, sort_keys=False)

    print(f"[save_anomaly_detector] Saved model -> {model_path}")
    print(f"[save_anomaly_detector] Saved metadata -> {meta_path}")


def load_anomaly_detector(directory: str) -> Tuple[IsolationForest, float]:
    """Load a saved anomaly detector from disk.

    Args:
        directory: Directory that contains the saved artefacts.

    Returns:
        Tuple of ``(model, threshold)``.

    Raises:
        FileNotFoundError: If the model or metadata file is missing.
    """
    dir_path = Path(directory)

    model_path = dir_path / "isolation_forest.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model: IsolationForest = joblib.load(model_path)

    meta_path = dir_path / "anomaly_metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    with open(meta_path, "r") as fh:
        meta = yaml.safe_load(fh)

    threshold = float(meta["threshold"])
    return model, threshold
