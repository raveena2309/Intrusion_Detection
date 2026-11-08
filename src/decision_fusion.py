"""
Decision Fusion Module for Intelligent NIDS
============================================

Combines predictions from the classifier and anomaly detector using a
rule-based fusion strategy. Supports threshold selection on validation data
and open-set detection (unknown / novel attacks).
"""

import os
import yaml
import joblib
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Result produced by :meth:`DecisionFusion.fuse`.

    Attributes:
        final_decision: Human-readable decision string, e.g. ``'DoS'``,
            ``'unknown_suspicious'``, ``'normal'``, or ``'uncertain_review'``.
        known_class: Class name predicted by the base classifier (before
            fusion rules are applied).
        confidence: Maximum class probability from the classifier.
        anomaly_score: Raw anomaly score from the anomaly detector.
        is_anomaly: Whether the anomaly detector flagged this sample.
        explanation_available: Whether SHAP / anomaly explanations should be
            generated for this sample (True for suspicious / uncertain cases).
    """
    final_decision: str
    known_class: str
    confidence: float
    anomaly_score: float
    is_anomaly: bool
    explanation_available: bool


# ---------------------------------------------------------------------------
# DecisionFusion
# ---------------------------------------------------------------------------

class DecisionFusion:
    """Rule-based decision fusion combining classifier and anomaly detector.

    Decision rules (applied in order):
        1. High confidence known attack -> that attack class.
        2. Classifier says *Normal* but anomaly detected -> ``unknown_suspicious``.
        3. Low confidence **and** high anomaly score -> ``unknown_suspicious``.
        4. Low confidence **and** normal anomaly score -> ``uncertain_review``.
        5. Classifier says *Normal* **and** anomaly detector says normal -> ``normal``.

    Args:
        confidence_threshold: Minimum class probability to trust the
            classifier's prediction (default ``0.6``).
        anomaly_threshold: Anomaly score above which a sample is considered
            anomalous (default ``0.5``).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        anomaly_threshold: float = 0.5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.anomaly_threshold = anomaly_threshold

    # ------------------------------------------------------------------
    # Core fusion
    # ------------------------------------------------------------------

    def fuse(
        self,
        predicted_class: int,
        class_probabilities: np.ndarray,
        anomaly_score: float,
        label_names: List[str],
        normal_class_idx: int = 6,
    ) -> FusionResult:
        """Apply fusion rules to a single sample.

        Args:
            predicted_class: Integer class index from the base classifier.
            class_probabilities: 1-D probability array of shape ``(n_classes,)``.
            anomaly_score: Scalar anomaly score (higher -> more anomalous).
            label_names: Class name list indexed by integer label.
            normal_class_idx: Index of the *Normal* class (default ``6``).

        Returns:
            :class:`FusionResult` for the sample.
        """
        confidence = float(class_probabilities[predicted_class])
        is_anomaly = anomaly_score > self.anomaly_threshold
        known_class = (
            label_names[predicted_class]
            if predicted_class < len(label_names)
            else str(predicted_class)
        )

        # Rule 1 - high-confidence known attack
        if confidence >= self.confidence_threshold and predicted_class != normal_class_idx:
            return FusionResult(
                final_decision=known_class,
                known_class=known_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                explanation_available=False,
            )

        # Rule 2 - classifier says Normal but anomaly detector fires
        if predicted_class == normal_class_idx and is_anomaly:
            return FusionResult(
                final_decision="unknown_suspicious",
                known_class=known_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                is_anomaly=True,
                explanation_available=True,
            )

        # Rule 3 - low confidence + high anomaly
        if confidence < self.confidence_threshold and is_anomaly:
            return FusionResult(
                final_decision="unknown_suspicious",
                known_class=known_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                is_anomaly=True,
                explanation_available=True,
            )

        # Rule 4 - low confidence + normal anomaly
        if confidence < self.confidence_threshold and not is_anomaly:
            return FusionResult(
                final_decision="uncertain_review",
                known_class=known_class,
                confidence=confidence,
                anomaly_score=anomaly_score,
                is_anomaly=False,
                explanation_available=True,
            )

        # Rule 5 - classifier Normal + anomaly detector Normal
        return FusionResult(
            final_decision="normal",
            known_class=known_class,
            confidence=confidence,
            anomaly_score=anomaly_score,
            is_anomaly=False,
            explanation_available=False,
        )

    # ------------------------------------------------------------------
    # Threshold selection
    # ------------------------------------------------------------------

    def select_thresholds(
        self,
        clf_probs: np.ndarray,
        anomaly_scores: np.ndarray,
        y_true_fusion: np.ndarray,
        normal_class_idx: int = 6,
        label_names: Optional[List[str]] = None,
    ) -> Tuple[float, float]:
        """Grid-search confidence and anomaly thresholds on validation data.

        Optimises macro-averaged F1 score of the fused decisions.

        Args:
            clf_probs: Classifier probability matrix, shape ``(n, n_classes)``.
            anomaly_scores: Anomaly score array, shape ``(n,)``.
            y_true_fusion: True integer labels, shape ``(n,)``.
            normal_class_idx: Index for the *Normal* class (default ``6``).
            label_names: Optional list of class name strings.

        Returns:
            Tuple ``(best_confidence_threshold, best_anomaly_threshold)``.
        """
        n_classes = clf_probs.shape[1]
        if label_names is None:
            label_names = [str(i) for i in range(n_classes)]

        clf_preds = np.argmax(clf_probs, axis=1)

        # Candidate thresholds
        conf_candidates = np.arange(0.3, 0.95, 0.05)
        anomaly_candidates = np.percentile(
            anomaly_scores, [10, 20, 30, 40, 50, 60, 70, 80, 90]
        )

        best_f1 = -1.0
        best_conf = 0.6
        best_anom = float(np.median(anomaly_scores))

        # Build a simple numeric mapping for evaluation
        # unknown_suspicious -> n_classes, uncertain_review -> n_classes+1, normal -> normal_class_idx
        UNKNOWN_LABEL = n_classes
        UNCERTAIN_LABEL = n_classes + 1

        def _decision_to_int(decision: str, known_cls: int) -> int:
            if decision == "unknown_suspicious":
                return UNKNOWN_LABEL
            if decision == "uncertain_review":
                return UNCERTAIN_LABEL
            if decision == "normal":
                return normal_class_idx
            return known_cls  # named attack class

        for c_thr in conf_candidates:
            for a_thr in anomaly_candidates:
                self.confidence_threshold = float(c_thr)
                self.anomaly_threshold = float(a_thr)

                y_pred_int = []
                for i in range(len(clf_preds)):
                    result = self.fuse(
                        int(clf_preds[i]),
                        clf_probs[i],
                        float(anomaly_scores[i]),
                        label_names,
                        normal_class_idx,
                    )
                    y_pred_int.append(_decision_to_int(result.final_decision, int(clf_preds[i])))

                y_pred_arr = np.array(y_pred_int)
                # Compare on the known classes only (exclude uncertain / unknown)
                mask = y_pred_arr < n_classes
                if mask.sum() == 0:
                    continue
                f1 = f1_score(
                    y_true_fusion[mask], y_pred_arr[mask],
                    average="macro", zero_division=0,
                    labels=list(range(n_classes)),
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_conf = float(c_thr)
                    best_anom = float(a_thr)

        self.confidence_threshold = best_conf
        self.anomaly_threshold = best_anom
        print(f"[fusion] Best thresholds -> conf={best_conf:.2f}, anomaly={best_anom:.4f} (F1={best_f1:.4f})")
        return best_conf, best_anom

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_fusion(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        clf_model,
        anomaly_model,
        anomaly_threshold: float,
        conf_threshold: float,
        label_names: List[str],
        normal_class_idx: int = 6,
    ) -> dict:
        """Evaluate the fusion system on a dataset.

        Args:
            X: Feature matrix, shape ``(n, p)``.
            y_true: True integer labels, shape ``(n,)``.
            clf_model: Fitted classifier with ``predict_proba`` method.
            anomaly_model: Fitted anomaly detector with ``decision_function``
                or ``score_samples`` method.
            anomaly_threshold: Anomaly score cut-off.
            conf_threshold: Classifier confidence cut-off.
            label_names: Class name list indexed by integer label.
            normal_class_idx: Index for the *Normal* class.

        Returns:
            Dictionary with keys:
            ``decisions``, ``n_unknown_suspicious``, ``n_uncertain_review``,
            ``n_known``, ``known_accuracy``, ``unknown_recall``,
            ``false_positive_rate``, ``precision_unknown``.
        """
        self.confidence_threshold = conf_threshold
        self.anomaly_threshold = anomaly_threshold

        clf_probs = clf_model.predict_proba(X)
        clf_preds = np.argmax(clf_probs, axis=1)

        # Anomaly scores - higher means more anomalous
        if hasattr(anomaly_model, "decision_function"):
            raw_scores = anomaly_model.decision_function(X)
        elif hasattr(anomaly_model, "score_samples"):
            raw_scores = anomaly_model.score_samples(X)
        else:
            raise AttributeError("anomaly_model must have decision_function or score_samples")
        # Negate so that higher = more anomalous (IsolationForest convention)
        anomaly_scores = -raw_scores

        decisions = []
        for i in range(len(X)):
            r = self.fuse(
                int(clf_preds[i]),
                clf_probs[i],
                float(anomaly_scores[i]),
                label_names,
                normal_class_idx,
            )
            decisions.append(r.final_decision)

        decisions = np.array(decisions)
        n_unknown = int((decisions == "unknown_suspicious").sum())
        n_uncertain = int((decisions == "uncertain_review").sum())
        n_known = int(~np.isin(decisions, ["unknown_suspicious", "uncertain_review"]).sum() if False else
                      np.isin(decisions, ["unknown_suspicious", "uncertain_review", "normal"]).sum())
        # Simpler: everything that isn't uncertain or unknown
        is_known_decision = ~np.isin(decisions, ["unknown_suspicious", "uncertain_review"])

        # Known-class accuracy (only on samples given a named-class decision)
        known_mask = is_known_decision
        if known_mask.sum() > 0:
            known_preds = clf_preds[known_mask]
            known_true = y_true[known_mask]
            known_acc = float((known_preds == known_true).mean())
        else:
            known_acc = 0.0

        # Unknown-attack recall: fraction of non-normal attacks caught as unknown_suspicious
        attack_mask = y_true != normal_class_idx
        if attack_mask.sum() > 0:
            unknown_recall = float((decisions[attack_mask] == "unknown_suspicious").mean())
        else:
            unknown_recall = 0.0

        # False-positive rate: normal samples flagged as unknown_suspicious
        normal_mask = y_true == normal_class_idx
        if normal_mask.sum() > 0:
            fpr = float((decisions[normal_mask] == "unknown_suspicious").mean())
        else:
            fpr = 0.0

        # Precision for unknown_suspicious decisions
        unknown_flag = decisions == "unknown_suspicious"
        if unknown_flag.sum() > 0:
            prec_unknown = float((y_true[unknown_flag] != normal_class_idx).mean())
        else:
            prec_unknown = 0.0

        return {
            "decisions": decisions.tolist(),
            "n_unknown_suspicious": n_unknown,
            "n_uncertain_review": n_uncertain,
            "n_known_decision": int(is_known_decision.sum()),
            "known_accuracy": known_acc,
            "unknown_recall": unknown_recall,
            "false_positive_rate": fpr,
            "precision_unknown": prec_unknown,
            "conf_threshold": conf_threshold,
            "anomaly_threshold": anomaly_threshold,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save fusion parameters to *directory*.

        Args:
            directory: Target directory path.  Created if necessary.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        params = {
            "confidence_threshold": self.confidence_threshold,
            "anomaly_threshold": self.anomaly_threshold,
        }
        path = out / "fusion_params.yaml"
        with open(path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        print(f"[fusion] Parameters saved -> {path}")

    @classmethod
    def load(cls, directory: str) -> "DecisionFusion":
        """Load fusion parameters previously saved with :meth:`save`.

        Args:
            directory: Directory containing ``fusion_params.yaml``.

        Returns:
            :class:`DecisionFusion` instance with restored thresholds.

        Raises:
            FileNotFoundError: If the params file is missing.
        """
        path = Path(directory) / "fusion_params.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Fusion params not found at {path}")
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        instance = cls(
            confidence_threshold=params["confidence_threshold"],
            anomaly_threshold=params["anomaly_threshold"],
        )
        print(f"[fusion] Parameters loaded <- {path}")
        return instance
