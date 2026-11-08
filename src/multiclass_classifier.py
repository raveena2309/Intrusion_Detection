"""
Multiclass Classifier Module for Intelligent NIDS
==================================================

Implements Random Forest and XGBoost classifiers for 10-class attack
categorization on the UNSW-NB15 dataset. Provides training, evaluation,
persistence, and result reporting utilities.
"""

import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassifierResult:
    """Container for classifier evaluation results."""

    model_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_report: Dict[str, Any]
    confusion_matrix: np.ndarray
    label_names: List[str]
    feature_names: List[str]
    training_time: float  # seconds


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[dict] = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Args:
        X_train: Feature matrix for training.
        y_train: Encoded integer class labels.
        config: Optional model configuration dictionary.  When supplied the
                function reads parameters from ``config['classifiers']['random_forest']``.

    Returns:
        Fitted ``RandomForestClassifier`` instance.
    """
    rf_cfg: dict = {}
    if config is not None:
        rf_cfg = config.get("classifiers", {}).get("random_forest", {})

    n_estimators = rf_cfg.get("n_estimators", 200)
    max_depth = rf_cfg.get("max_depth", None)
    min_samples_split = rf_cfg.get("min_samples_split", 5)
    min_samples_leaf = rf_cfg.get("min_samples_leaf", 2)
    n_jobs = rf_cfg.get("n_jobs", -1)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[dict] = None,
) -> "XGBClassifier":
    """Train an XGBoost multiclass classifier.

    Args:
        X_train: Feature matrix for training.
        y_train: Encoded integer class labels (0-indexed, contiguous).
        config: Optional model configuration dictionary.  Parameters are read
                from ``config['classifiers']['xgboost']``.

    Returns:
        Fitted ``XGBClassifier`` instance.

    Raises:
        ImportError: If xgboost is not installed.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

    xgb_cfg: dict = {}
    if config is not None:
        xgb_cfg = config.get("classifiers", {}).get("xgboost", {})

    n_estimators = xgb_cfg.get("n_estimators", 300)
    max_depth = xgb_cfg.get("max_depth", 8)
    learning_rate = xgb_cfg.get("learning_rate", 0.1)
    subsample = xgb_cfg.get("subsample", 0.8)
    colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8)
    min_child_weight = xgb_cfg.get("min_child_weight", 3)
    reg_alpha = xgb_cfg.get("reg_alpha", 0.1)
    reg_lambda = xgb_cfg.get("reg_lambda", 1.0)
    n_jobs = xgb_cfg.get("n_jobs", -1)

    n_classes = len(np.unique(y_train))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=n_jobs,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: List[str],
    model_name: str,
    feature_names: Optional[List[str]] = None,
    training_time: float = 0.0,
) -> ClassifierResult:
    """Evaluate a fitted classifier on the test set.

    Args:
        model: Fitted classifier with a ``predict`` method.
        X_test: Feature matrix for the test set.
        y_test: True encoded integer class labels.
        label_names: Ordered list of class name strings (index must match
                     encoded label integers).
        model_name: Human-readable model identifier used in reports.
        feature_names: Optional list of feature names.
        training_time: Elapsed training time in seconds.

    Returns:
        :class:`ClassifierResult` populated with all computed metrics.
    """
    y_pred = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-class report as dict
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    return ClassifierResult(
        model_name=model_name,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_report=report_dict,
        confusion_matrix=cm,
        label_names=label_names,
        feature_names=feature_names or [],
        training_time=training_time,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_classifier(
    model: Any,
    model_name: str,
    metadata: dict,
    directory: str,
) -> None:
    """Persist a fitted classifier and its metadata to disk.

    Args:
        model: Fitted classifier object.
        model_name: Short identifier used as the file stem (e.g. ``"random_forest"``).
        metadata: Arbitrary key-value metadata to store alongside the model.
        directory: Destination directory; created automatically if absent.
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    model_path = dir_path / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    meta_path = dir_path / f"{model_name}_metadata.yaml"
    with open(meta_path, "w") as fh:
        yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

    print(f"[save_classifier] Saved '{model_name}' -> {model_path}")


def load_classifier(model_name: str, directory: str) -> Any:
    """Load a previously saved classifier from disk.

    Args:
        model_name: Short identifier used as the file stem.
        directory: Directory that contains the saved artefacts.

    Returns:
        The deserialized classifier object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(directory) / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[ClassifierResult]) -> None:
    """Print a formatted comparison table for a list of classifier results.

    Args:
        results: List of :class:`ClassifierResult` objects to compare.
    """
    header = f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Weighted-F1':>12} {'Train(s)':>10}"
    print("\n" + "=" * len(header))
    print("CLASSIFIER COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model_name:<25} {r.accuracy:>10.4f} {r.macro_f1:>10.4f} "
            f"{r.weighted_f1:>12.4f} {r.training_time:>10.1f}"
        )
    print("=" * len(header))
