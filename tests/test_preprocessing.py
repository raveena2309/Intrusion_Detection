"""
Test Preprocessing Module
==========================

Unit tests for the preprocessing pipeline using sample data.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import NIDSPreprocessor, validate_input_schema
from src.data_loader import load_raw_dataset


@pytest.fixture
def sample_data():
    """Load sample dataset for testing."""
    from pathlib import Path
    sample_path = Path(__file__).parent.parent / "datasets" / "sample" / "UNSW_NB15_sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)
    else:
        pytest.skip("Sample dataset not available")


@pytest.fixture
def preprocessor(sample_data):
    """Create a fitted preprocessor using sample data."""
    p = NIDSPreprocessor()
    p.fit(sample_data)
    return p


class TestPreprocessor:
    """Tests for NIDSPreprocessor."""
    
    def test_fit_creates_encoders(self, preprocessor):
        """Preprocessor should create encoders for categorical columns."""
        assert preprocessor.is_fitted
        assert "proto" in preprocessor.categorical_encoders
        assert "service" in preprocessor.categorical_encoders
        assert "state" in preprocessor.categorical_encoders
    
    def test_fit_creates_scaler(self, preprocessor):
        """Preprocessor should create a scaler."""
        assert preprocessor.scaler is not None
    
    def test_fit_creates_label_encoder(self, preprocessor):
        """Preprocessor should create a label encoder."""
        assert preprocessor.label_encoder is not None
        assert len(preprocessor.label_encoder.classes_) == 10
    
    def test_transform_output_shape(self, preprocessor, sample_data):
        """Transform should produce the correct number of features."""
        X = preprocessor.transform(sample_data)
        assert X.shape[0] == len(sample_data)
        assert X.shape[1] == 42  # 45 columns - id - label - attack_cat
    
    def test_transform_no_nan(self, preprocessor, sample_data):
        """Transform output should not contain NaN values."""
        X = preprocessor.transform(sample_data)
        assert not np.isnan(X).any()
    
    def test_transform_no_inf(self, preprocessor, sample_data):
        """Transform output should not contain infinity values."""
        X = preprocessor.transform(sample_data)
        assert not np.isinf(X).any()
    
    def test_excludes_id_column(self, preprocessor):
        """Feature names should not include 'id'."""
        assert "id" not in preprocessor.feature_names
    
    def test_excludes_target_columns(self, preprocessor):
        """Feature names should not include target columns."""
        assert "label" not in preprocessor.feature_names
        assert "attack_cat" not in preprocessor.feature_names
    
    def test_binary_labels(self, preprocessor, sample_data):
        """Binary labels should be 0 or 1."""
        y = preprocessor.get_labels(sample_data, "binary")
        assert set(np.unique(y)).issubset({0, 1})
    
    def test_multiclass_labels(self, preprocessor, sample_data):
        """Multiclass labels should be integers from 0 to 9."""
        y = preprocessor.get_labels(sample_data, "multiclass")
        assert y.min() >= 0
        assert y.max() <= 9
    
    def test_fit_transform_equals_separate(self, sample_data):
        """fit_transform should produce the same result as fit then transform."""
        p1 = NIDSPreprocessor()
        X1 = p1.fit_transform(sample_data)
        
        p2 = NIDSPreprocessor()
        p2.fit(sample_data)
        X2 = p2.transform(sample_data)
        
        np.testing.assert_array_almost_equal(X1, X2)
    
    def test_unseen_categorical_handling(self, preprocessor, sample_data):
        """Preprocessor should handle unseen categorical values."""
        df = sample_data.copy()
        df.loc[0, "proto"] = "COMPLETELY_NEW_PROTOCOL"
        
        # Should not raise an error
        X = preprocessor.transform(df)
        assert X.shape[0] == len(df)
    
    def test_save_and_load(self, preprocessor, sample_data, tmp_path):
        """Saved and loaded preprocessor should produce identical results."""
        X_before = preprocessor.transform(sample_data)
        
        save_dir = str(tmp_path / "test_preprocessor")
        preprocessor.save(save_dir)
        
        loaded = NIDSPreprocessor.load(save_dir)
        X_after = loaded.transform(sample_data)
        
        np.testing.assert_array_almost_equal(X_before, X_after)
    
    def test_unfitted_transform_raises(self):
        """Transform on unfitted preprocessor should raise RuntimeError."""
        p = NIDSPreprocessor()
        df = pd.DataFrame({"proto": ["tcp"], "service": ["-"], "state": ["FIN"]})
        with pytest.raises(RuntimeError):
            p.transform(df)


class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_valid_schema(self, preprocessor, sample_data):
        """Valid data should pass schema validation."""
        result = validate_input_schema(sample_data, preprocessor.feature_names)
        assert result["valid"]
    
    def test_missing_column_detected(self, preprocessor, sample_data):
        """Missing columns should be detected."""
        df = sample_data.drop(columns=["proto"])
        result = validate_input_schema(df, preprocessor.feature_names)
        assert not result["valid"]
        assert "proto" in result["missing_columns"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
