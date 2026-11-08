"""
Explainability Module for Intelligent NIDS
===========================================

Provides SHAP-based global and local explanations for the classifier,
plus statistical anomaly explanations for the anomaly detector output.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Global SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: int = 500,
) -> dict:
    """Compute global SHAP values for *model* using :class:`shap.TreeExplainer`.

    Suitable for tree-based models such as Random Forest and XGBoost.

    Args:
        model: Fitted tree-based classifier (scikit-learn or XGBoost API).
        X: Feature matrix used for SHAP computation.  Rows are sampled
            down to *max_samples* if necessary.
        feature_names: Ordered list of feature names matching *X* columns.
        max_samples: Maximum number of rows to pass to the explainer.
            Larger values increase runtime; default is 500.

    Returns:
        Dictionary with keys:

        ``shap_values``
            SHAP value array of shape ``(n_samples, n_features)`` for
            multi-class models this is the mean-absolute values collapsed
            over classes, or the raw 3-D array when ``raw=True``.
        ``shap_values_raw``
            Raw SHAP array - 3-D ``(n_samples, n_features, n_classes)`` for
            multi-class, 2-D ``(n_samples, n_features)`` for binary.
        ``expected_value``
            Model base value(s) from the explainer.
        ``feature_names``
            Echo of the input feature names.
        ``n_samples``
            Number of samples actually used.

    Raises:
        ImportError: If the *shap* package is not installed.
        RuntimeError: If the explainer cannot be initialised for *model*.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    # Down-sample if needed
    rng = np.random.default_rng(42)
    n = len(X)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
    except Exception as exc:
        raise RuntimeError(f"Could not create TreeExplainer: {exc}") from exc

    shap_raw = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value

    # Collapse to 2-D mean-absolute for ranking purposes
    if isinstance(shap_raw, list):
        # Multi-class: list of (n_samples, n_features) arrays
        shap_3d = np.stack(shap_raw, axis=2)  # (n, f, C)
        shap_collapsed = np.abs(shap_3d).mean(axis=2)  # (n, f)
    else:
        shap_3d = shap_raw
        shap_collapsed = np.abs(shap_raw)

    return {
        "shap_values": shap_collapsed,
        "shap_values_raw": shap_raw,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "n_samples": len(X_sample),
        "X_sample": X_sample,
    }


# ---------------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_dict: dict,
    X: np.ndarray,
    feature_names: List[str],
    save_path: str,
    max_display: int = 20,
) -> None:
    """Generate and save a SHAP beeswarm (summary) plot.

    Args:
        shap_dict: Dictionary returned by :func:`compute_shap_values`.
        X: Feature matrix aligned with *shap_dict* (same rows).
        feature_names: Feature name list.
        save_path: Path to save the PNG figure.
        max_display: Maximum number of features to display.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is not installed.")

    shap_raw = shap_dict["shap_values_raw"]
    X_sample = shap_dict.get("X_sample", X)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.4)))

    # For multi-class (list), shap.summary_plot handles it natively
    shap.summary_plot(
        shap_raw,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.gcf().set_size_inches(10, max(6, max_display * 0.4))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[explainability] SHAP summary plot saved -> {save_path}")


def plot_shap_feature_importance(
    shap_dict: dict,
    feature_names: List[str],
    save_path: str,
    top_n: int = 20,
) -> None:
    """Generate and save a SHAP mean-absolute feature importance bar chart.

    Args:
        shap_dict: Dictionary returned by :func:`compute_shap_values`.
        feature_names: Feature name list.
        save_path: Path to save the PNG figure.
        top_n: Number of top features to display.
    """
    shap_collapsed = shap_dict["shap_values"]  # (n_samples, n_features)
    mean_abs = np.abs(shap_collapsed).mean(axis=0)  # (n_features,)

    order = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] for i in order]
    top_vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Feature Importances (SHAP)")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[explainability] SHAP feature importance plot saved -> {save_path}")


# ---------------------------------------------------------------------------
# Local (single-sample) explanations
# ---------------------------------------------------------------------------

def explain_single_prediction(
    explainer,
    X_single: np.ndarray,
    feature_names: List[str],
    label_names: List[str],
    predicted_class: int,
    top_n: int = 10,
) -> dict:
    """Explain one prediction using SHAP values.

    Args:
        explainer: A fitted ``shap.TreeExplainer`` instance.
        X_single: Single-sample feature vector, shape ``(1, n_features)``
            or ``(n_features,)``.
        feature_names: Feature name list.
        label_names: Class name list indexed by integer label.
        predicted_class: Integer index of the predicted class.
        top_n: Number of top contributing features to return.

    Returns:
        Dictionary with:
        ``predicted_class_name`` (str),
        ``top_features`` (list of dicts with ``feature``, ``value``,
        ``shap_value``, ``direction``).
    """
    if not _SHAP_AVAILABLE:
        raise ImportError("shap is not installed.")

    if X_single.ndim == 1:
        X_single = X_single.reshape(1, -1)

    shap_raw = explainer.shap_values(X_single)

    # Extract SHAP values for predicted class
    if isinstance(shap_raw, list):
        cls_shap = shap_raw[predicted_class][0]  # (n_features,)
    else:
        cls_shap = shap_raw[0]

    # Rank by absolute contribution
    order = np.argsort(np.abs(cls_shap))[::-1][:top_n]

    top_features = []
    for i in order:
        top_features.append({
            "feature": feature_names[i],
            "value": float(X_single[0, i]),
            "shap_value": float(cls_shap[i]),
            "direction": "increases_risk" if cls_shap[i] > 0 else "decreases_risk",
        })

    class_name = (
        label_names[predicted_class]
        if predicted_class < len(label_names)
        else str(predicted_class)
    )

    return {
        "predicted_class": predicted_class,
        "predicted_class_name": class_name,
        "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Anomaly explanation (z-score based)
# ---------------------------------------------------------------------------

def explain_anomaly(
    X_single: np.ndarray,
    feature_names: List[str],
    normal_means: np.ndarray,
    normal_stds: np.ndarray,
    top_n: int = 10,
) -> dict:
    """Explain why a sample is flagged as anomalous using z-scores.

    Identifies which features deviate most from the normal (training-set)
    distribution.

    Args:
        X_single: Single-sample feature vector, shape ``(n_features,)`` or
            ``(1, n_features)``.
        feature_names: Feature name list.
        normal_means: Per-feature means of the normal class in the training
            set, shape ``(n_features,)``.
        normal_stds: Per-feature standard deviations, shape ``(n_features,)``.
            Values of 0 are replaced with 1 to avoid division errors.
        top_n: Number of most deviant features to return.

    Returns:
        Dictionary with:
        ``top_deviating_features`` (list of dicts with ``feature``,
        ``value``, ``normal_mean``, ``normal_std``, ``z_score``,
        ``direction``).
    """
    if X_single.ndim == 2:
        X_single = X_single.flatten()

    stds_safe = np.where(normal_stds == 0, 1.0, normal_stds)
    z_scores = (X_single - normal_means) / stds_safe

    order = np.argsort(np.abs(z_scores))[::-1][:top_n]

    top_deviating = []
    for i in order:
        top_deviating.append({
            "feature": feature_names[i],
            "value": float(X_single[i]),
            "normal_mean": float(normal_means[i]),
            "normal_std": float(normal_stds[i]),
            "z_score": float(z_scores[i]),
            "direction": "above_normal" if z_scores[i] > 0 else "below_normal",
        })

    return {"top_deviating_features": top_deviating}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_shap_artifacts(shap_dict: dict, save_dir: str) -> None:
    """Serialise SHAP computation results to disk.

    Saves a compressed pickle of *shap_dict* (excluding ``X_sample`` to save
    space) plus a plain-text feature importance ranking.

    Args:
        shap_dict: Dictionary returned by :func:`compute_shap_values`.
        save_dir: Directory where artefacts will be written.
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)

    pkl_path = out / "shap_artifacts.pkl"
    # Exclude X_sample from serialised artifact (recoverable from test data)
    to_save = {k: v for k, v in shap_dict.items() if k != "X_sample"}
    with open(pkl_path, "wb") as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[explainability] SHAP artifacts saved -> {pkl_path}")

    # Also save a human-readable feature importance ranking
    shap_collapsed = shap_dict["shap_values"]
    feature_names = shap_dict["feature_names"]
    mean_abs = np.abs(shap_collapsed).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    txt_path = out / "shap_feature_ranking.txt"
    with open(txt_path, "w") as f:
        f.write("Rank | Feature | Mean |SHAP|\n")
        f.write("-" * 45 + "\n")
        for rank, i in enumerate(order, 1):
            f.write(f"{rank:4d} | {feature_names[i]:<30s} | {mean_abs[i]:.6f}\n")
    print(f"[explainability] SHAP feature ranking saved -> {txt_path}")
