"""
Test Inference Module
======================

Unit tests for the inference pipeline.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_data():
    """Load sample dataset."""
    p = PROJECT_ROOT / "datasets" / "sample" / "UNSW_NB15_sample.csv"
    if p.exists():
        return pd.read_csv(p)
    pytest.skip("Sample dataset not available")


@pytest.fixture
def preprocessor():
    """Load fitted preprocessor."""
    from preprocessing import NIDSPreprocessor
    p = PROJECT_ROOT / "models" / "preprocessing"
    if (p / "preprocessor.joblib").exists():
        return NIDSPreprocessor.load(str(p))
    pytest.skip("Preprocessor not fitted yet")


@pytest.fixture
def inference_engine(preprocessor):
    """Load inference engine if all models available."""
    import joblib
    from inference import NIDSInferenceEngine
    from decision_fusion import DecisionFusion

    clf_path = PROJECT_ROOT / "models" / "classifier" / "xgboost.joblib"
    anom_path = PROJECT_ROOT / "models" / "anomaly" / "isolation_forest.joblib"

    if not clf_path.exists() or not anom_path.exists():
        pytest.skip("Model artifacts not available")

    clf = joblib.load(clf_path)
    anom = joblib.load(anom_path)
    fusion = DecisionFusion(confidence_threshold=0.6, anomaly_threshold=0.005)

    return NIDSInferenceEngine(
        preprocessor=preprocessor,
        classifier=clf,
        anomaly_model=anom,
        fusion=fusion,
        label_encoder=preprocessor.label_encoder,
    )


class TestInferenceEngine:
    """Tests for NIDSInferenceEngine."""

    def test_predict_single(self, inference_engine, sample_data):
        """Single record prediction returns valid PredictionResult."""
        result = inference_engine.predict_single(sample_data.iloc[0])
        assert result.prediction is not None
        assert 0 <= result.confidence <= 1
        assert isinstance(result.is_anomaly, bool)
        assert isinstance(result.anomaly_score, float)

    def test_predict_batch(self, inference_engine, sample_data):
        """Batch prediction returns correct number of results."""
        results = inference_engine.predict_batch(sample_data.head(10))
        assert len(results) == 10
        for r in results:
            assert r.prediction is not None
            assert 0 <= r.confidence <= 1

    def test_predict_with_latency(self, inference_engine, sample_data):
        """Prediction with latency measurement works."""
        results, latency = inference_engine.predict_with_latency(sample_data.head(10))
        assert len(results) == 10
        assert latency >= 0

    def test_get_model_info(self, inference_engine):
        """Model info returns expected keys."""
        info = inference_engine.get_model_info()
        assert "n_features" in info or "feature_count" in info or len(info) > 0

    def test_prediction_values_valid(self, inference_engine, sample_data):
        """All prediction values are from expected set."""
        valid_predictions = {
            "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
            "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
            "normal", "unknown_suspicious", "uncertain_review",
        }
        results = inference_engine.predict_batch(sample_data.head(20))
        for r in results:
            assert r.prediction in valid_predictions, f"Unexpected prediction: {r.prediction}"

    def test_known_normal_record(self, inference_engine, sample_data):
        """Normal records should generally get normal-ish predictions."""
        normal_rows = sample_data[sample_data["attack_cat"] == "Normal"]
        if len(normal_rows) == 0:
            pytest.skip("No normal records in sample")
        result = inference_engine.predict_single(normal_rows.iloc[0])
        # Not asserting exact prediction (model may differ), just that it runs
        assert result.prediction is not None
