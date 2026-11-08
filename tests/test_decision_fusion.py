"""
Test Decision Fusion Module
=============================

Unit tests for the decision fusion pipeline.
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_fusion import DecisionFusion, FusionResult


LABEL_NAMES = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
]
NORMAL_IDX = 6


@pytest.fixture
def fusion():
    """Create a DecisionFusion with default thresholds."""
    return DecisionFusion(confidence_threshold=0.7, anomaly_threshold=0.5)


class TestDecisionFusion:
    """Tests for DecisionFusion decision logic."""

    def test_high_confidence_known_attack(self, fusion):
        """High confidence known attack should return that attack."""
        probs = np.zeros(10)
        probs[3] = 0.95  # Exploits = index 3
        result = fusion.fuse(3, probs, 0.1, LABEL_NAMES, NORMAL_IDX)
        assert result.final_decision == "Exploits"
        assert result.confidence > 0.7

    def test_normal_with_no_anomaly(self, fusion):
        """Confident normal + no anomaly = normal."""
        probs = np.zeros(10)
        probs[NORMAL_IDX] = 0.9
        result = fusion.fuse(NORMAL_IDX, probs, 0.1, LABEL_NAMES, NORMAL_IDX)
        assert result.final_decision in ("normal", "Normal")

    def test_normal_prediction_with_high_anomaly(self, fusion):
        """Classifier says normal but anomaly score high = unknown_suspicious."""
        probs = np.zeros(10)
        probs[NORMAL_IDX] = 0.8
        result = fusion.fuse(NORMAL_IDX, probs, 0.9, LABEL_NAMES, NORMAL_IDX)
        assert result.final_decision == "unknown_suspicious"
        assert result.is_anomaly is True

    def test_low_confidence_high_anomaly(self, fusion):
        """Low confidence + high anomaly = unknown_suspicious."""
        probs = np.full(10, 0.1)
        predicted = 3  # Exploits
        result = fusion.fuse(predicted, probs, 0.9, LABEL_NAMES, NORMAL_IDX)
        assert result.final_decision == "unknown_suspicious"

    def test_low_confidence_normal_anomaly(self, fusion):
        """Low confidence + normal anomaly = uncertain_review."""
        probs = np.full(10, 0.1)
        predicted = 3
        result = fusion.fuse(predicted, probs, 0.1, LABEL_NAMES, NORMAL_IDX)
        assert result.final_decision == "uncertain_review"

    def test_fusion_result_fields(self, fusion):
        """FusionResult has all expected fields."""
        probs = np.zeros(10)
        probs[5] = 0.85
        result = fusion.fuse(5, probs, 0.2, LABEL_NAMES, NORMAL_IDX)
        assert hasattr(result, "final_decision")
        assert hasattr(result, "known_class")
        assert hasattr(result, "confidence")
        assert hasattr(result, "anomaly_score")
        assert hasattr(result, "is_anomaly")
        assert isinstance(result.confidence, float)
        assert isinstance(result.anomaly_score, float)
        assert isinstance(result.is_anomaly, bool)

    def test_all_classes_handled(self, fusion):
        """Fusion should handle prediction of every class."""
        for cls_idx in range(10):
            probs = np.zeros(10)
            probs[cls_idx] = 0.85
            result = fusion.fuse(cls_idx, probs, 0.1, LABEL_NAMES, NORMAL_IDX)
            assert result.final_decision is not None

    def test_save_and_load(self, fusion, tmp_path):
        """Save and load should preserve fusion parameters."""
        fusion.save(str(tmp_path))
        loaded = DecisionFusion.load(str(tmp_path))
        assert loaded.confidence_threshold == fusion.confidence_threshold
        assert loaded.anomaly_threshold == fusion.anomaly_threshold

    def test_select_thresholds_returns_values(self, fusion):
        """select_thresholds should return two floats."""
        np.random.seed(42)
        n = 200
        clf_probs = np.random.dirichlet(np.ones(10), n)
        anom_scores = np.random.randn(n)
        y_true = np.random.randint(0, 10, n)
        conf_t, anom_t = fusion.select_thresholds(
            clf_probs, anom_scores, y_true,
            normal_class_idx=NORMAL_IDX, label_names=LABEL_NAMES,
        )
        assert isinstance(conf_t, float)
        assert isinstance(anom_t, float)
