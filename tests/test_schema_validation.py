"""
Test Schema Validation Module
================================

Unit tests for traffic schema validation and mapping.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic.schema_mapping import (
    UNSWNB15_FEATURES,
    validate_feature_schema,
    get_feature_defaults,
    PCAP_MAPPABLE_FEATURES,
    PCAP_UNAVAILABLE_FEATURES,
)
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestSchemaMapping:
    """Tests for schema_mapping module."""

    def test_feature_count(self):
        """Should have 42 features."""
        assert len(UNSWNB15_FEATURES) == 42

    def test_feature_names_match_dataset(self):
        """Feature names should match the actual dataset columns."""
        sample_path = PROJECT_ROOT / "datasets" / "sample" / "UNSW_NB15_sample.csv"
        if not sample_path.exists():
            pytest.skip("Sample dataset not available")
        df = pd.read_csv(sample_path)
        expected_features = [c for c in df.columns if c not in ("id", "label", "attack_cat")]
        assert set(UNSWNB15_FEATURES) == set(expected_features)

    def test_validate_valid_schema(self):
        """Valid DataFrame should pass validation."""
        df = pd.DataFrame({f: [0.0] for f in UNSWNB15_FEATURES})
        result = validate_feature_schema(df)
        assert result["valid"] is True
        assert len(result["missing"]) == 0

    def test_validate_missing_columns(self):
        """Missing columns should be detected."""
        features_minus_one = UNSWNB15_FEATURES[:-1]
        df = pd.DataFrame({f: [0.0] for f in features_minus_one})
        result = validate_feature_schema(df)
        assert result["valid"] is False
        assert len(result["missing"]) == 1

    def test_validate_extra_columns_ok(self):
        """Extra columns should be allowed (not cause failure)."""
        df = pd.DataFrame({f: [0.0] for f in UNSWNB15_FEATURES})
        df["extra_col"] = 1.0
        result = validate_feature_schema(df)
        assert result["valid"] is True
        assert "extra_col" in result["extra"]

    def test_defaults_cover_all_features(self):
        """Feature defaults should provide values for all features."""
        defaults = get_feature_defaults()
        for f in UNSWNB15_FEATURES:
            assert f in defaults, f"Missing default for feature: {f}"

    def test_pcap_features_subset(self):
        """PCAP mappable features should be a subset of all features."""
        assert set(PCAP_MAPPABLE_FEATURES).issubset(set(UNSWNB15_FEATURES))

    def test_pcap_unavailable_is_complement(self):
        """Unavailable features should be the complement of mappable."""
        all_set = set(UNSWNB15_FEATURES)
        mappable_set = set(PCAP_MAPPABLE_FEATURES)
        unavailable_set = set(PCAP_UNAVAILABLE_FEATURES)
        # Every feature is either mappable or unavailable
        assert mappable_set | unavailable_set == all_set

    def test_no_overlap_mappable_unavailable(self):
        """Mappable and unavailable features should not overlap."""
        assert len(set(PCAP_MAPPABLE_FEATURES) & set(PCAP_UNAVAILABLE_FEATURES)) == 0


class TestSchemaOnSampleData:
    """Test schema validation against actual sample data."""

    def test_sample_dataset_valid(self):
        """Sample dataset should pass schema validation."""
        sample_path = PROJECT_ROOT / "datasets" / "sample" / "UNSW_NB15_sample.csv"
        if not sample_path.exists():
            pytest.skip("Sample dataset not available")
        df = pd.read_csv(sample_path)
        result = validate_feature_schema(df)
        assert result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
