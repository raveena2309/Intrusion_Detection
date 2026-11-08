"""
Preprocessing Module for Intelligent NIDS
==========================================

Handles feature preprocessing, encoding, scaling, and schema validation.
All transformations are fitted on training data only. Raw data is never modified.
"""

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_name: str = "data_config.yaml") -> dict:
    """Load a YAML configuration file."""
    config_path = get_project_root() / "config" / config_name
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class NIDSPreprocessor:
    """Preprocessing pipeline for UNSW-NB15 dataset.
    
    This preprocessor handles:
    - Categorical encoding (proto, service, state)
    - Numerical scaling
    - Missing value imputation
    - Infinity value handling
    - Feature selection (excluding identifiers and target columns)
    - Label encoding for attack categories
    
    All transformations are fitted on training data only.
    """
    
    # Categorical columns in UNSW-NB15
    CATEGORICAL_COLUMNS = ["proto", "service", "state"]
    
    # Columns to always exclude from features
    EXCLUDE_COLUMNS = ["id", "label", "attack_cat"]
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary. If None, loads from default config file.
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.is_fitted = False
        
        # Encoders for categorical features
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        
        # Scaler for numerical features
        self.scaler: Optional[object] = None
        
        # Imputer for missing values
        self.numerical_imputer: Optional[SimpleImputer] = None
        
        # Label encoder for attack categories
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Feature names and ordering
        self.feature_names: List[str] = []
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        
        # Metadata
        self.metadata: Dict = {}
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature column names, excluding identifiers and targets.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            List of feature column names.
        """
        return [col for col in df.columns if col not in self.EXCLUDE_COLUMNS]
    
    def _handle_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinity values in numerical columns.
        
        Args:
            df: Input DataFrame (will be modified in place on a copy).
        
        Returns:
            DataFrame with infinities handled.
        """
        df = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        strategy = self.config.get("preprocessing", {}).get("infinity_handling", "clip")
        
        if strategy == "clip":
            for col in numerical_cols:
                if df[col].dtype in [np.float64, np.float32]:
                    finite_vals = df[col][np.isfinite(df[col])]
                    if len(finite_vals) > 0:
                        max_val = finite_vals.max()
                        min_val = finite_vals.min()
                        df[col] = df[col].clip(lower=min_val, upper=max_val)
        elif strategy == "nan":
            df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder.
        
        Args:
            df: Input DataFrame.
            fit: Whether to fit the encoders (True for training data only).
        
        Returns:
            DataFrame with categorical columns encoded as integers.
        """
        df = df.copy()
        
        for col in self.CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                # Include a placeholder for unseen values
                unique_vals = df[col].astype(str).unique().tolist()
                unique_vals.append("__UNKNOWN__")
                le.fit(unique_vals)
                self.categorical_encoders[col] = le
            
            if col in self.categorical_encoders:
                le = self.categorical_encoders[col]
                known_classes = set(le.classes_)
                
                # Handle unseen categories safely
                col_values = df[col].astype(str).copy()
                unseen_mask = ~col_values.isin(known_classes)
                if unseen_mask.any():
                    col_values[unseen_mask] = "__UNKNOWN__"
                
                df[col] = le.transform(col_values)
        
        return df
    
    def fit(self, df: pd.DataFrame) -> 'NIDSPreprocessor':
        """Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame (raw, unmodified).
        
        Returns:
            Self, for method chaining.
        """
        config = self.config.get("preprocessing", {})
        
        # Work on a copy
        df_work = df.copy()
        
        # Identify feature columns
        feature_cols = self._get_feature_columns(df_work)
        
        # Separate categorical and numerical
        self.categorical_features = [c for c in self.CATEGORICAL_COLUMNS if c in feature_cols]
        self.numerical_features = [c for c in feature_cols if c not in self.CATEGORICAL_COLUMNS]
        self.feature_names = self.categorical_features + self.numerical_features
        
        # Handle infinities
        df_work = self._handle_infinities(df_work)
        
        # Encode categoricals
        df_work = self._encode_categoricals(df_work, fit=True)
        
        # Fit imputer on numerical features
        self.numerical_imputer = SimpleImputer(
            strategy=config.get("missing_values", {}).get("numerical_strategy", "median")
        )
        if self.numerical_features:
            self.numerical_imputer.fit(df_work[self.numerical_features])
        
        # Fit scaler
        scaling_method = config.get("scaling", {}).get("method", "standard")
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Fit scaler on all feature columns (after encoding categoricals)
        all_features_df = df_work[self.feature_names]
        if self.numerical_features:
            all_features_df = all_features_df.copy()
            all_features_df[self.numerical_features] = self.numerical_imputer.transform(
                all_features_df[self.numerical_features]
            )
        self.scaler.fit(all_features_df)
        
        # Fit label encoder for attack categories
        target_col = self.config["dataset"]["target"]["multiclass_label"]
        if target_col in df.columns:
            self.label_encoder = LabelEncoder()
            labels = df[target_col].astype(str).str.strip()
            self.label_encoder.fit(labels)
        
        # Store metadata
        self.metadata = {
            "n_features": len(self.feature_names),
            "n_numerical": len(self.numerical_features),
            "n_categorical": len(self.categorical_features),
            "feature_names": self.feature_names,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "scaling_method": scaling_method,
            "random_seed": config.get("random_seed", 42),
        }
        
        if self.label_encoder is not None:
            self.metadata["label_classes"] = self.label_encoder.classes_.tolist()
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform (raw).
        
        Returns:
            Numpy array of preprocessed features.
        
        Raises:
            RuntimeError: If preprocessor hasn't been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        # Work on a copy
        df_work = df.copy()
        
        # Handle infinities
        df_work = self._handle_infinities(df_work)
        
        # Encode categoricals
        df_work = self._encode_categoricals(df_work, fit=False)
        
        # Select and order features
        features_df = df_work[self.feature_names].copy()
        
        # Impute missing numerical values
        if self.numerical_features:
            features_df[self.numerical_features] = self.numerical_imputer.transform(
                features_df[self.numerical_features]
            )
        
        # Scale
        X = self.scaler.transform(features_df)
        
        return X
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            df: Training DataFrame.
        
        Returns:
            Numpy array of preprocessed features.
        """
        self.fit(df)
        return self.transform(df)
    
    def get_labels(self, df: pd.DataFrame, label_type: str = "multiclass") -> np.ndarray:
        """Extract and encode labels from a DataFrame.
        
        Args:
            df: DataFrame containing label columns.
            label_type: Either 'binary' or 'multiclass'.
        
        Returns:
            Numpy array of encoded labels.
        """
        config = self.config["dataset"]["target"]
        
        if label_type == "binary":
            col = config["binary_label"]
            return df[col].values.astype(int)
        elif label_type == "multiclass":
            col = config["multiclass_label"]
            labels = df[col].astype(str).str.strip()
            if self.label_encoder is not None:
                return self.label_encoder.transform(labels)
            else:
                raise RuntimeError("Label encoder not fitted. Call fit() first.")
        else:
            raise ValueError(f"label_type must be 'binary' or 'multiclass', got '{label_type}'")
    
    def save(self, directory: str):
        """Save the fitted preprocessor to disk.
        
        Args:
            directory: Directory to save preprocessor artifacts.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            "categorical_encoders": self.categorical_encoders,
            "scaler": self.scaler,
            "numerical_imputer": self.numerical_imputer,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "metadata": self.metadata,
            "config": self.config,
        }
        
        joblib.dump(artifacts, dir_path / "preprocessor.joblib")
        
        # Also save metadata as YAML for readability
        with open(dir_path / "preprocessor_metadata.yaml", "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
    
    @classmethod
    def load(cls, directory: str) -> 'NIDSPreprocessor':
        """Load a fitted preprocessor from disk.
        
        Args:
            directory: Directory containing preprocessor artifacts.
        
        Returns:
            Fitted NIDSPreprocessor instance.
        """
        dir_path = Path(directory)
        artifacts = joblib.load(dir_path / "preprocessor.joblib")
        
        preprocessor = cls(config=artifacts["config"])
        preprocessor.categorical_encoders = artifacts["categorical_encoders"]
        preprocessor.scaler = artifacts["scaler"]
        preprocessor.numerical_imputer = artifacts["numerical_imputer"]
        preprocessor.label_encoder = artifacts["label_encoder"]
        preprocessor.feature_names = artifacts["feature_names"]
        preprocessor.numerical_features = artifacts["numerical_features"]
        preprocessor.categorical_features = artifacts["categorical_features"]
        preprocessor.metadata = artifacts["metadata"]
        preprocessor.is_fitted = True
        
        return preprocessor


def validate_input_schema(df: pd.DataFrame, expected_features: List[str]) -> Dict:
    """Validate that input data has the expected feature schema.
    
    Args:
        df: Input DataFrame to validate.
        expected_features: List of expected feature column names.
    
    Returns:
        Validation result dictionary.
    """
    result = {
        "valid": True,
        "missing_columns": [],
        "extra_columns": [],
        "issues": [],
    }
    
    available_cols = set(df.columns)
    expected_cols = set(expected_features)
    
    result["missing_columns"] = sorted(expected_cols - available_cols)
    result["extra_columns"] = sorted(available_cols - expected_cols)
    
    if result["missing_columns"]:
        result["valid"] = False
        result["issues"].append(f"Missing columns: {result['missing_columns']}")
    
    # Check for NaN in critical columns
    for col in expected_features:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                result["issues"].append(f"Column '{col}' has {nan_count} NaN values")
    
    return result
