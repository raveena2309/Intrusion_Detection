"""
Evaluation Module for Intelligent NIDS
========================================

Provides comprehensive evaluation utilities:
- Open-set metrics (unknown-attack recall, FPR, etc.)
- Operational metrics (latency, throughput)
- Confusion matrix / AUROC plotting
- Full evaluation report aggregation
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# Open-set metrics
# ---------------------------------------------------------------------------

def compute_openset_metrics(
    y_true: np.ndarray,
    y_pred_fusion: np.ndarray,
    withheld_class_names: List[str],
    label_names: Optional[List[str]] = None,
) -> dict:
    """Compute open-set evaluation metrics for the fusion system.

    For each withheld class (classes not seen during classifier training),
    measures how often the fusion system correctly flags samples as
    ``unknown_suspicious`` instead of silently mis-classifying them.

    Args:
        y_true: Ground-truth string class labels, shape ``(n,)``.
        y_pred_fusion: Fusion decision strings, shape ``(n,)``.
        withheld_class_names: Class names that were withheld from the
            classifier (the classes we want to detect as unknown).
        label_names: All known class names (used for context in the report).

    Returns:
        Dictionary with the following keys:

        ``unknown_recall`` (float)
            Fraction of withheld-class samples correctly flagged as
            ``unknown_suspicious``.
        ``false_positive_rate`` (float)
            Fraction of normal traffic flagged as ``unknown_suspicious``.
        ``precision_unknown`` (float)
            Precision of the ``unknown_suspicious`` decision.
        ``n_withheld_samples`` (int)
            Number of samples from withheld classes in the evaluation set.
        ``per_class_recall`` (dict)
            Per withheld class unknown recall.
    """
    y_true = np.asarray(y_true)
    y_pred_fusion = np.asarray(y_pred_fusion)

    # -- Unknown recall on withheld classes ----------------------------------
    withheld_mask = np.isin(y_true, withheld_class_names)
    n_withheld = int(withheld_mask.sum())

    if n_withheld > 0:
        unknown_recall = float(
            (y_pred_fusion[withheld_mask] == "unknown_suspicious").mean()
        )
    else:
        unknown_recall = 0.0

    # -- Per-class recall ----------------------------------------------------
    per_class_recall: Dict[str, float] = {}
    for cls in withheld_class_names:
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            per_class_recall[cls] = float(
                (y_pred_fusion[cls_mask] == "unknown_suspicious").mean()
            )
        else:
            per_class_recall[cls] = 0.0

    # -- False-positive rate on normal traffic --------------------------------
    normal_mask = y_true == "Normal"
    if normal_mask.sum() > 0:
        fpr = float((y_pred_fusion[normal_mask] == "unknown_suspicious").mean())
    else:
        fpr = 0.0

    # -- Precision of unknown_suspicious flag ---------------------------------
    unknown_flag_mask = y_pred_fusion == "unknown_suspicious"
    if unknown_flag_mask.sum() > 0:
        # Positive = sample is from a withheld class
        precision_unknown = float(
            np.isin(y_true[unknown_flag_mask], withheld_class_names).mean()
        )
    else:
        precision_unknown = 0.0

    return {
        "unknown_recall": unknown_recall,
        "false_positive_rate": fpr,
        "precision_unknown": precision_unknown,
        "n_withheld_samples": n_withheld,
        "n_unknown_suspicious_flags": int(unknown_flag_mask.sum()),
        "per_class_recall": per_class_recall,
    }


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------

def compute_operational_metrics(
    model,
    X: np.ndarray,
    batch_size: int = 1000,
) -> dict:
    """Measure inference latency and throughput of *model*.

    Uses ``predict`` if available, otherwise ``predict_proba``.

    Args:
        model: Fitted scikit-learn compatible model.
        X: Feature matrix, shape ``(n, p)``.
        batch_size: Number of samples per batch for throughput measurement.

    Returns:
        Dictionary with:
        ``mean_latency_ms``, ``p50_latency_ms``, ``p95_latency_ms``,
        ``p99_latency_ms``, ``throughput_samples_per_sec``.
    """
    predict_fn = getattr(model, "predict", None) or getattr(model, "predict_proba")

    n = len(X)
    batch_latencies = []

    idx = 0
    while idx < n:
        batch = X[idx: idx + batch_size]
        t0 = time.perf_counter()
        predict_fn(batch)
        t1 = time.perf_counter()
        batch_latencies.append((t1 - t0) * 1000.0)  # ms per batch
        idx += batch_size

    latencies = np.array(batch_latencies)
    per_sample = latencies / batch_size  # ms per sample

    # Total elapsed for throughput
    total_ms = latencies.sum()
    throughput = (n / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    return {
        "mean_latency_ms": float(per_sample.mean()),
        "p50_latency_ms": float(np.percentile(per_sample, 50)),
        "p95_latency_ms": float(np.percentile(per_sample, 95)),
        "p99_latency_ms": float(np.percentile(per_sample, 99)),
        "throughput_samples_per_sec": float(throughput),
        "total_batches": len(batch_latencies),
        "n_samples_evaluated": n,
    }


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: List[str],
    title: str,
    save_path: str,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array of shape ``(n_classes, n_classes)``.
        label_names: Class name list indexed by integer label.
        title: Figure title string.
        save_path: Path to save the PNG figure.
        normalize: Whether to normalise rows to sum to 1 (default ``True``).
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0.0)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.8)))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=label_names,
        yticklabels=label_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_plot[i, j]
            text = f"{val:{fmt}}" if val > 0 else ""
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=6, color="white" if val > thresh else "black")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Confusion matrix saved -> {save_path}")


def plot_auroc_curve(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
    save_path: str,
    title: str = "AUROC Curve",
) -> float:
    """Plot and save a binary AUROC curve.

    Args:
        y_true_binary: Binary ground-truth labels (0 / 1).
        scores: Continuous decision scores (higher -> more positive).
        save_path: Path to save the PNG figure.
        title: Figure title string.

    Returns:
        AUROC value as a float.
    """
    fpr_arr, tpr_arr, _ = roc_curve(y_true_binary, scores)
    roc_auc = auc(fpr_arr, tpr_arr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, lw=2, label=f"AUROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] AUROC curve saved -> {save_path}")
    return float(roc_auc)


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def generate_full_evaluation_report(
    clf_result: dict,
    anomaly_result: dict,
    fusion_result: dict,
) -> dict:
    """Aggregate classifier, anomaly, and fusion results into a single report.

    Args:
        clf_result: Metrics dict from the classifier evaluation step.
            Expected keys: ``accuracy``, ``macro_f1``, ``classification_report``.
        anomaly_result: Metrics dict from the anomaly detector evaluation step.
            Expected keys: ``auroc``, ``threshold``, ``precision``, ``recall``,
            ``f1``.
        fusion_result: Metrics dict from :meth:`DecisionFusion.evaluate_fusion`.

    Returns:
        Merged report dictionary with top-level sections
        ``classifier``, ``anomaly_detector``, ``decision_fusion``, ``summary``.
    """
    summary = {
        "classifier_accuracy": clf_result.get("accuracy"),
        "classifier_macro_f1": clf_result.get("macro_f1"),
        "anomaly_auroc": anomaly_result.get("auroc"),
        "fusion_known_accuracy": fusion_result.get("known_accuracy"),
        "fusion_unknown_recall": fusion_result.get("unknown_recall"),
        "fusion_false_positive_rate": fusion_result.get("false_positive_rate"),
        "fusion_precision_unknown": fusion_result.get("precision_unknown"),
    }

    return {
        "classifier": clf_result,
        "anomaly_detector": anomaly_result,
        "decision_fusion": fusion_result,
        "summary": summary,
    }
