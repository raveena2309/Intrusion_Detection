"""
Confidence Calibration Module for Intelligent NIDS
====================================================

Provides probability calibration for classifier outputs using Platt scaling
(sigmoid) or isotonic regression. Also computes calibration metrics such as
Expected Calibration Error (ECE) and Brier score, and generates reliability
diagrams.
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Calibration fitting
# ---------------------------------------------------------------------------

def calibrate_classifier(
    base_model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
    cv=5,
) -> CalibratedClassifierCV:
    """Calibrate a pre-fitted classifier using Platt scaling or isotonic regression.

    Args:
        base_model: A pre-fitted scikit-learn compatible classifier.
        X_val: Validation feature matrix (used for fitting the calibrator).
        y_val: Validation labels (integer-encoded).
        method: Calibration method - ``'sigmoid'`` (Platt scaling) or
            ``'isotonic'`` (isotonic regression).
        cv: Cross-validation strategy. Use ``'prefit'`` when *base_model* is
            already trained.

    Returns:
        A fitted ``CalibratedClassifierCV`` instance.

    Raises:
        ValueError: If *method* is not ``'sigmoid'`` or ``'isotonic'``.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ValueError(f"method must be 'sigmoid' or 'isotonic', got '{method}'")

    calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    calibrated.fit(X_val, y_val)
    return calibrated


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the weighted average gap between predicted confidence and
    empirical accuracy across probability bins.

    Args:
        y_true: Binary ground-truth labels (0 or 1) for the positive class.
        y_prob: Predicted probabilities for the positive class (shape ``(n,)``).
        n_bins: Number of equally-spaced probability bins.

    Returns:
        ECE as a float in ``[0, 1]``.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_weight = mask.sum() / n_samples
        ece += bin_weight * abs(bin_confidence - bin_accuracy)

    return float(ece)


def compute_brier_score(
    y_true: np.ndarray,
    y_prob_matrix: np.ndarray,
) -> float:
    """Compute the multi-class Brier score (macro-averaged over classes).

    For each class *c* the binary Brier score is computed with one-vs-rest
    binarisation, then the scores are averaged.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob_matrix: Predicted probability matrix, shape ``(n, n_classes)``.

    Returns:
        Mean Brier score across all classes.
    """
    n_classes = y_prob_matrix.shape[1]
    scores = []
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        scores.append(brier_score_loss(binary_true, y_prob_matrix[:, c]))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    save_path: str,
    n_bins: int = 10,
) -> None:
    """Plot and save a reliability (calibration) diagram.

    One reliability curve is drawn per class using a one-vs-rest binarisation.

    Args:
        y_true: Integer class labels, shape ``(n,)``.
        y_prob: Predicted probability matrix, shape ``(n, n_classes)``.
        label_names: Human-readable class names indexed by integer label.
        save_path: Absolute or relative path where the PNG will be saved.
        n_bins: Number of probability bins for the calibration curve.
    """
    n_classes = y_prob.shape[1]
    fig, ax = plt.subplots(figsize=(8, 7))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")

    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        if binary_true.sum() == 0:
            continue
        fraction_of_positives, mean_predicted = calibration_curve(
            binary_true, y_prob[:, c], n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_predicted, fraction_of_positives, marker="o", lw=1.2,
                label=label_names[c] if c < len(label_names) else str(c))

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[calibration] Reliability diagram saved -> {save_path}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_calibrator(model: CalibratedClassifierCV, directory: str) -> None:
    """Serialise a fitted calibrated classifier to *directory*.

    Args:
        model: Fitted ``CalibratedClassifierCV`` instance.
        directory: Directory path.  Created if it does not exist.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibrated_classifier.joblib"
    joblib.dump(model, path)
    print(f"[calibration] Calibrator saved -> {path}")


def load_calibrator(directory: str) -> CalibratedClassifierCV:
    """Load a calibrated classifier previously saved with :func:`save_calibrator`.

    Args:
        directory: Directory containing ``calibrated_classifier.joblib``.

    Returns:
        Fitted ``CalibratedClassifierCV`` instance.

    Raises:
        FileNotFoundError: If the expected file is missing.
    """
    path = Path(directory) / "calibrated_classifier.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Calibrator not found at {path}")
    model = joblib.load(path)
    print(f"[calibration] Calibrator loaded <- {path}")
    return model
