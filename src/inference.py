"""Shared inference interface -- the prediction API for the NIDS system.

This module provides a unified ``NIDSInferenceEngine`` that wraps the trained
preprocessor, multiclass classifier, anomaly detector, and fusion strategy
behind a single, easy-to-use API.

All heavy model loading is done once at construction time.  Subsequent calls
to ``predict_*`` are stateless and thread-safe (provided the underlying
sklearn/joblib models are thread-safe, which they are for predict operations).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Single-record prediction output from the NIDS inference engine.

    Attributes:
        prediction: Predicted attack category name (e.g. ``'Normal'``, ``'DoS'``).
        known_class: Whether the prediction matches a known training class
            (always ``True`` for classifier output; ``False`` only when the
            anomaly detector flags a record that the classifier puts into
            *Normal* -- indicating a potentially novel threat).
        confidence: Classifier probability for the predicted class (0-1).
        anomaly_score: Raw output of the anomaly detector.  Higher (less
            negative for Isolation Forest) means *more anomalous*.
        is_anomaly: ``True`` if the anomaly detector flags this record as
            suspicious regardless of the classifier's output.
        explanation_available: Whether SHAP explanations can be generated for
            this record (requires the explainability module).
    """

    prediction: str
    known_class: bool
    confidence: float
    anomaly_score: float
    is_anomaly: bool
    explanation_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "prediction": self.prediction,
            "known_class": self.known_class,
            "confidence": float(self.confidence),
            "anomaly_score": float(self.anomaly_score),
            "is_anomaly": bool(self.is_anomaly),
            "explanation_available": self.explanation_available,
        }


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


class NIDSInferenceEngine:
    """Unified inference engine for the Intelligent NIDS.

    Wraps the preprocessor, classifier, anomaly model, and fusion strategy
    behind a consistent prediction interface used by the Streamlit dashboard
    and the PCAP replay pipeline.

    Args:
        preprocessor: Fitted :class:`~src.preprocessing.NIDSPreprocessor`.
        classifier: Trained multiclass classifier with ``predict_proba``.
        anomaly_model: Trained anomaly detector (``decision_function`` or
            ``score_samples``).
        fusion: Callable accepting ``(clf_proba, anomaly_scores)`` and
            returning a dict with keys ``prediction``, ``is_anomaly``,
            ``confidence`` (all arrays of length *n*).
        label_encoder: Fitted ``LabelEncoder`` mapping integer indices to
            class name strings.
    """

    def __init__(
        self,
        preprocessor,
        classifier,
        anomaly_model,
        fusion,
        label_encoder,
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.anomaly_model = anomaly_model
        self.fusion = fusion
        self.label_encoder = label_encoder

        # Derive label names from the label encoder
        self._label_names: List[str] = (
            list(label_encoder.classes_) if label_encoder is not None else []
        )

        # Determine anomaly threshold (default 0; models vary)
        self._anomaly_threshold: float = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Run preprocessing and return feature matrix."""
        return self.preprocessor.transform(df)

    def _anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute raw anomaly scores (higher -> more anomalous)."""
        try:
            if hasattr(self.anomaly_model, "decision_function"):
                scores = self.anomaly_model.decision_function(X)
            elif hasattr(self.anomaly_model, "score_samples"):
                scores = self.anomaly_model.score_samples(X)
            else:
                scores = np.zeros(len(X))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anomaly scoring error: %s -- using zeros.", exc)
            scores = np.zeros(len(X))
        # Negate so higher = more anomalous (Isolation Forest convention)
        return -scores

    def _clf_proba(self, X: np.ndarray) -> np.ndarray:
        """Return classifier probability matrix (n_samples × n_classes)."""
        return self.classifier.predict_proba(X)

    def _build_results(
        self,
        clf_proba: np.ndarray,
        anomaly_scores: np.ndarray,
    ) -> List[PredictionResult]:
        """Fuse classifier + anomaly outputs and build result list."""
        pred_indices = np.argmax(clf_proba, axis=1)
        confidences = np.max(clf_proba, axis=1)

        is_anomaly_flags = anomaly_scores > self._anomaly_threshold

        # Optionally let the fusion strategy override predictions
        fusion_out: Optional[Dict] = None
        try:
            fusion_out = self.fusion(clf_proba, anomaly_scores)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fusion step error: %s -- using raw classifier output.", exc)

        results: List[PredictionResult] = []
        for i in range(len(pred_indices)):
            pred_name = (
                self._label_names[pred_indices[i]]
                if pred_indices[i] < len(self._label_names)
                else str(pred_indices[i])
            )
            confidence = float(confidences[i])
            is_anomaly = bool(is_anomaly_flags[i])

            if fusion_out is not None:
                if "prediction" in fusion_out and hasattr(fusion_out["prediction"], "__len__"):
                    pred_name = fusion_out["prediction"][i]
                if "is_anomaly" in fusion_out and hasattr(fusion_out["is_anomaly"], "__len__"):
                    is_anomaly = bool(fusion_out["is_anomaly"][i])
                if "confidence" in fusion_out and hasattr(fusion_out["confidence"], "__len__"):
                    confidence = float(fusion_out["confidence"][i])

            known_class = not (is_anomaly and pred_name == "Normal")

            results.append(
                PredictionResult(
                    prediction=pred_name,
                    known_class=known_class,
                    confidence=confidence,
                    anomaly_score=float(anomaly_scores[i]),
                    is_anomaly=is_anomaly,
                    explanation_available=True,  # SHAP is available via explainability module
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, record: pd.Series) -> PredictionResult:
        """Predict a single network flow record.

        Args:
            record: A pandas Series with UNSW-NB15 feature names as the index.

        Returns:
            :class:`PredictionResult` for this record.
        """
        df = record.to_frame().T.reset_index(drop=True)
        X = self._preprocess(df)
        clf_proba = self._clf_proba(X)
        anomaly_scores = self._anomaly_scores(X)
        results = self._build_results(clf_proba, anomaly_scores)
        return results[0]

    def predict_batch(self, df: pd.DataFrame) -> List[PredictionResult]:
        """Predict a batch of network flow records.

        Args:
            df: DataFrame with UNSW-NB15 feature columns (one row per flow).

        Returns:
            List of :class:`PredictionResult` instances, one per row.
        """
        if df.empty:
            return []
        X = self._preprocess(df)
        clf_proba = self._clf_proba(X)
        anomaly_scores = self._anomaly_scores(X)
        return self._build_results(clf_proba, anomaly_scores)

    def predict_with_latency(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[PredictionResult], float]:
        """Predict a batch and report per-record inference latency.

        Args:
            df: DataFrame with UNSW-NB15 feature columns.

        Returns:
            Tuple of (results list, latency_ms_per_record).
        """
        if df.empty:
            return [], 0.0

        t0 = time.perf_counter()
        results = self.predict_batch(df)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latency_per_record = elapsed_ms / max(len(df), 1)
        return results, round(latency_per_record, 4)

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded models.

        Returns:
            Dictionary with keys: ``feature_count``, ``classes``,
            ``classifier_type``, ``anomaly_model_type``,
            ``preprocessor_fitted``.
        """
        info: Dict[str, Any] = {
            "feature_count": (
                len(self.preprocessor.feature_names)
                if hasattr(self.preprocessor, "feature_names")
                else "unknown"
            ),
            "classes": self._label_names,
            "n_classes": len(self._label_names),
            "classifier_type": type(self.classifier).__name__,
            "anomaly_model_type": type(self.anomaly_model).__name__,
            "preprocessor_fitted": getattr(self.preprocessor, "is_fitted", False),
        }

        # Append preprocessor metadata if available
        if hasattr(self.preprocessor, "metadata"):
            info["preprocessor_metadata"] = self.preprocessor.metadata

        return info
