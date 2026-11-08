"""PCAP replay and prediction pipeline.

This module orchestrates the full flow: PCAP → flow extraction →
preprocessing → model prediction → labelled results DataFrame.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from traffic.flow_extractor import extract_flows_from_pcap, flows_to_dataframe
from traffic.schema_mapping import validate_feature_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class IncompatibleFeatureSchema(Exception):
    """Raised when the extracted flow DataFrame is incompatible with the model.

    This typically happens when mandatory features cannot be derived and no
    sensible defaults exist, making inference unreliable.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replay_pcap(
    pcap_path: str,
    preprocessor,  # NIDSPreprocessor (avoid circular import with type hint)
    classifier,    # trained sklearn-compatible classifier (predict_proba)
    anomaly_model,  # trained sklearn-compatible anomaly detector (decision_function / score_samples)
    fusion,         # fusion strategy callable: fusion(clf_proba, anomaly_scores) → dict
    label_names: List[str],
) -> pd.DataFrame:
    """Full PCAP → prediction pipeline.

    Extracts bidirectional flows from *pcap_path*, applies the NIDS preprocessing
    pipeline, runs the classifier and anomaly detector, and returns a results
    DataFrame annotated with predicted labels and scores.

    Args:
        pcap_path: Path to a ``.pcap`` or ``.pcapng`` file.
        preprocessor: Fitted :class:`~src.preprocessing.NIDSPreprocessor` instance.
        classifier: Trained multiclass classifier with a ``predict_proba`` method.
        anomaly_model: Trained anomaly detector.  Should expose either
            ``decision_function`` or ``score_samples``.
        fusion: Callable that accepts ``(clf_proba: np.ndarray, anomaly_scores: np.ndarray)``
            and returns a dict with keys ``prediction``, ``confidence``, ``is_anomaly``.
        label_names: Ordered list of class names matching the classifier's output.

    Returns:
        DataFrame with one row per flow containing:

        - Original UNSW-NB15 feature columns (with defaults for PCAP-unavailable ones)
        - ``prediction`` – predicted class name
        - ``confidence`` – probability of predicted class
        - ``anomaly_score`` – raw anomaly detector output
        - ``is_anomaly`` – boolean flag

    Raises:
        FileNotFoundError: If *pcap_path* does not exist.
        IncompatibleFeatureSchema: If the extracted schema cannot be reconciled
            with what the preprocessor expects.
        RuntimeError: For other processing failures.
    """
    # 1. Extract flows
    logger.info("Extracting flows from PCAP: %s", pcap_path)
    flows = extract_flows_from_pcap(pcap_path)

    if not flows:
        logger.warning("No flows extracted from PCAP — returning empty DataFrame.")
        return pd.DataFrame()

    # 2. Convert to DataFrame
    df_flows = flows_to_dataframe(flows)

    # 3. Validate schema
    validation = validate_feature_schema(df_flows)
    if not validation["valid"]:
        raise IncompatibleFeatureSchema(
            f"Extracted flow schema is incompatible with the UNSW-NB15 model. "
            f"Missing features: {validation['missing']}. "
            f"Issues: {validation['issues']}"
        )

    logger.info("Schema validation passed.  %d flows ready for inference.", len(df_flows))

    # 4. Preprocess
    try:
        X = preprocessor.transform(df_flows)
    except Exception as exc:
        raise RuntimeError(f"Preprocessing failed: {exc}") from exc

    # 5. Classify
    try:
        clf_proba = classifier.predict_proba(X)  # shape (n_flows, n_classes)
    except Exception as exc:
        raise RuntimeError(f"Classifier prediction failed: {exc}") from exc

    # 6. Anomaly scoring
    try:
        if hasattr(anomaly_model, "decision_function"):
            anomaly_scores = anomaly_model.decision_function(X)
        elif hasattr(anomaly_model, "score_samples"):
            anomaly_scores = anomaly_model.score_samples(X)
        else:
            logger.warning("Anomaly model has no decision_function or score_samples; using zeros.")
            anomaly_scores = np.zeros(len(X))
    except Exception as exc:
        logger.warning("Anomaly scoring failed (%s); using zeros.", exc)
        anomaly_scores = np.zeros(len(X))

    # 7. Fusion
    try:
        fusion_results = fusion(clf_proba, anomaly_scores)
    except Exception as exc:
        raise RuntimeError(f"Fusion step failed: {exc}") from exc

    # 8. Assemble results
    results = df_flows.copy()

    pred_indices = np.argmax(clf_proba, axis=1)
    results["prediction"] = [
        label_names[i] if i < len(label_names) else str(i) for i in pred_indices
    ]
    results["confidence"] = np.max(clf_proba, axis=1)
    results["anomaly_score"] = anomaly_scores

    # Fusion outputs (prediction may override classifier if anomaly detected)
    if isinstance(fusion_results, dict):
        if "prediction" in fusion_results:
            results["prediction"] = fusion_results["prediction"]
        if "is_anomaly" in fusion_results:
            results["is_anomaly"] = fusion_results["is_anomaly"]
        else:
            results["is_anomaly"] = False
    else:
        results["is_anomaly"] = False

    logger.info(
        "Replay complete.  %d flows classified.  %d flagged as anomalous.",
        len(results),
        int(results["is_anomaly"].sum()) if "is_anomaly" in results.columns else 0,
    )
    return results
