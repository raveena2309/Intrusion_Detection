"""
Data Loader Module for Intelligent NIDS
========================================

Handles loading, validation, and initial inspection of the UNSW-NB15 dataset.
Keeps raw data unchanged; all transformations happen on copies.
"""

import os
import hashlib
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_name: str = "data_config.yaml") -> dict:
    """Load a YAML configuration file.
    
    Args:
        config_name: Name of the config file in the config/ directory.
    
    Returns:
        Dictionary with configuration values.
    """
    config_path = get_project_root() / "config" / config_name
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification.
    
    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm to use (default: sha256).
    
    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_file_info(filepath: str) -> Dict:
    """Get file metadata including size and hash.
    
    Args:
        filepath: Path to the file.
    
    Returns:
        Dictionary with file information.
    """
    path = Path(filepath)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": compute_file_hash(str(path)),
    }


def load_raw_dataset(split: str = "training") -> pd.DataFrame:
    """Load a raw dataset CSV without any modifications.
    
    Args:
        split: Either 'training' or 'testing'.
    
    Returns:
        Raw DataFrame as read from the CSV.
    
    Raises:
        FileNotFoundError: If the dataset file is not found.
        ValueError: If split is not 'training' or 'testing'.
    """
    if split not in ("training", "testing"):
        raise ValueError(f"split must be 'training' or 'testing', got '{split}'")
    
    config = load_config()
    raw_dir = get_project_root() / config["dataset"]["paths"]["raw_dir"]
    filename = config["dataset"]["expected_files"][split]
    filepath = raw_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            f"Please download the UNSW-NB15 {split} set from:\n"
            f"  {config['dataset']['source_url']}\n"
            f"and place it in: {raw_dir}"
        )
    
    df = pd.read_csv(filepath)
    return df


def validate_dataset_schema(df: pd.DataFrame, split: str) -> Dict:
    """Validate the dataset schema against expected structure.
    
    Args:
        df: DataFrame to validate.
        split: Dataset split name for reporting.
    
    Returns:
        Dictionary with validation results.
    """
    config = load_config()
    expected_target_binary = config["dataset"]["target"]["binary_label"]
    expected_target_multi = config["dataset"]["target"]["multiclass_label"]
    expected_attacks = config["dataset"]["attack_categories"]
    
    results = {
        "split": split,
        "shape": df.shape,
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "has_binary_label": expected_target_binary in df.columns,
        "has_multiclass_label": expected_target_multi in df.columns,
        "issues": [],
    }
    
    # Check for target columns
    if not results["has_binary_label"]:
        results["issues"].append(f"Missing binary label column: '{expected_target_binary}'")
    if not results["has_multiclass_label"]:
        results["issues"].append(f"Missing multiclass label column: '{expected_target_multi}'")
    
    # Check attack categories if multiclass label exists
    if results["has_multiclass_label"]:
        actual_categories = df[expected_target_multi].dropna().unique()
        results["actual_categories"] = sorted([str(c).strip() for c in actual_categories])
        missing_cats = set(expected_attacks) - set(results["actual_categories"])
        extra_cats = set(results["actual_categories"]) - set(expected_attacks)
        if missing_cats:
            results["issues"].append(f"Missing expected categories: {missing_cats}")
        if extra_cats:
            results["issues"].append(f"Unexpected categories found: {extra_cats}")
    
    return results


def analyze_data_quality(df: pd.DataFrame, split: str) -> Dict:
    """Perform comprehensive data quality analysis.
    
    Args:
        df: DataFrame to analyze.
        split: Dataset split name for reporting.
    
    Returns:
        Dictionary with data quality metrics.
    """
    quality = {
        "split": split,
        "shape": df.shape,
        "missing_values": {},
        "infinity_values": {},
        "duplicate_rows": 0,
        "constant_columns": [],
        "column_types": {"numerical": [], "categorical": [], "other": []},
    }
    
    # Missing values per column
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    quality["total_missing_cells"] = int(missing.sum())
    quality["missing_values"] = {
        col: {"count": int(missing[col]), "percentage": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }
    
    # Infinity values in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            quality["infinity_values"][col] = inf_count
    
    # Duplicate rows
    quality["duplicate_rows"] = int(df.duplicated().sum())
    quality["duplicate_percentage"] = round(quality["duplicate_rows"] / len(df) * 100, 2)
    
    # Constant columns (only one unique value)
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            quality["constant_columns"].append(col)
    
    # Column type classification
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            quality["column_types"]["numerical"].append(col)
        elif df[col].dtype == object or str(df[col].dtype) == "category":
            quality["column_types"]["categorical"].append(col)
        else:
            quality["column_types"]["other"].append(col)
    
    # Basic statistics for numerical columns
    quality["numerical_stats"] = {}
    for col in quality["column_types"]["numerical"]:
        series = df[col].replace([np.inf, -np.inf], np.nan)
        quality["numerical_stats"][col] = {
            "min": float(series.min()) if not series.isna().all() else None,
            "max": float(series.max()) if not series.isna().all() else None,
            "mean": float(series.mean()) if not series.isna().all() else None,
            "std": float(series.std()) if not series.isna().all() else None,
            "median": float(series.median()) if not series.isna().all() else None,
            "nunique": int(series.nunique()),
        }
    
    # Categorical column details
    quality["categorical_stats"] = {}
    for col in quality["column_types"]["categorical"]:
        quality["categorical_stats"][col] = {
            "nunique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(10).to_dict(),
        }
    
    return quality


def analyze_class_distribution(df: pd.DataFrame, split: str) -> Dict:
    """Analyze label and attack category distributions.
    
    Args:
        df: DataFrame to analyze.
        split: Dataset split name for reporting.
    
    Returns:
        Dictionary with class distribution information.
    """
    config = load_config()
    binary_col = config["dataset"]["target"]["binary_label"]
    multi_col = config["dataset"]["target"]["multiclass_label"]
    
    distribution = {"split": split}
    
    # Binary label distribution
    if binary_col in df.columns:
        binary_counts = df[binary_col].value_counts().sort_index()
        distribution["binary_distribution"] = {
            str(k): int(v) for k, v in binary_counts.items()
        }
        distribution["binary_percentage"] = {
            str(k): round(v / len(df) * 100, 2) for k, v in binary_counts.items()
        }
    
    # Multiclass distribution
    if multi_col in df.columns:
        # Strip whitespace from attack_cat
        attack_cats = df[multi_col].str.strip() if df[multi_col].dtype == object else df[multi_col]
        multi_counts = attack_cats.value_counts().sort_values(ascending=False)
        distribution["multiclass_distribution"] = {
            str(k): int(v) for k, v in multi_counts.items()
        }
        distribution["multiclass_percentage"] = {
            str(k): round(v / len(df) * 100, 2) for k, v in multi_counts.items()
        }
        distribution["n_classes"] = int(multi_counts.nunique())
        
        # Imbalance ratio
        if len(multi_counts) > 0:
            distribution["imbalance_ratio"] = round(
                float(multi_counts.max()) / float(multi_counts.min()), 2
            )
    
    return distribution


def check_train_test_overlap(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict:
    """Check for overlapping records between train and test sets.
    
    Excludes target columns and identifiers when checking for duplicates.
    
    Args:
        df_train: Training DataFrame.
        df_test: Testing DataFrame.
    
    Returns:
        Dictionary with overlap analysis results.
    """
    config = load_config()
    exclude_cols = config["dataset"]["exclude_columns"]
    
    # Get feature columns (exclude targets and identifiers)
    feature_cols = [c for c in df_train.columns if c not in exclude_cols and c in df_test.columns]
    
    # Create string representation for comparison
    train_records = df_train[feature_cols].astype(str).agg(",".join, axis=1)
    test_records = df_test[feature_cols].astype(str).agg(",".join, axis=1)
    
    overlap_count = test_records.isin(train_records).sum()
    
    return {
        "feature_columns_compared": len(feature_cols),
        "train_records": len(df_train),
        "test_records": len(df_test),
        "overlapping_records": int(overlap_count),
        "overlap_percentage_of_test": round(overlap_count / len(df_test) * 100, 2),
        "feature_columns_used": feature_cols,
    }


def identify_leakage_risks(df: pd.DataFrame) -> List[Dict]:
    """Identify potential data leakage risks in the dataset.
    
    Args:
        df: DataFrame to analyze.
    
    Returns:
        List of dictionaries describing potential leakage risks.
    """
    config = load_config()
    binary_col = config["dataset"]["target"]["binary_label"]
    multi_col = config["dataset"]["target"]["multiclass_label"]
    
    risks = []
    
    # Check for ID columns
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ("id", "index", "row_id"):
            risks.append({
                "column": col,
                "risk": "Identifier column - should be excluded from features",
                "severity": "HIGH",
                "recommendation": f"Exclude '{col}' from feature set",
            })
    
    # Check for perfect correlation with target
    if binary_col in df.columns:
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in (binary_col, multi_col):
                continue
            try:
                corr = abs(df[col].corr(df[binary_col].astype(float)))
                if corr > 0.99:
                    risks.append({
                        "column": col,
                        "risk": f"Near-perfect correlation ({corr:.4f}) with binary label",
                        "severity": "CRITICAL",
                        "recommendation": f"Investigate '{col}' for potential target leakage",
                    })
            except (ValueError, TypeError):
                pass
    
    # Check for columns with label-derived names
    suspect_names = ["label", "target", "class", "attack", "category"]
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in (binary_col.lower(), multi_col.lower()):
            continue
        for suspect in suspect_names:
            if suspect in col_lower:
                risks.append({
                    "column": col,
                    "risk": f"Column name suggests potential target derivation (contains '{suspect}')",
                    "severity": "MEDIUM",
                    "recommendation": f"Verify '{col}' is a legitimate feature, not derived from the target",
                })
                break
    
    return risks


def generate_dataset_report(output_dir: Optional[str] = None) -> Dict:
    """Generate a comprehensive dataset report for Phase 1.
    
    Args:
        output_dir: Directory to save the report. Defaults to results/metrics/.
    
    Returns:
        Complete dataset analysis report as a dictionary.
    """
    config = load_config()
    project_root = get_project_root()
    
    if output_dir is None:
        output_dir = project_root / "results" / "metrics"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "report_date": datetime.now().isoformat(),
        "dataset_name": config["dataset"]["name"],
        "source_url": config["dataset"]["source_url"],
    }
    
    # File information
    raw_dir = project_root / config["dataset"]["paths"]["raw_dir"]
    train_file = raw_dir / config["dataset"]["expected_files"]["training"]
    test_file = raw_dir / config["dataset"]["expected_files"]["testing"]
    
    report["files"] = {
        "training": get_file_info(str(train_file)),
        "testing": get_file_info(str(test_file)),
    }
    
    # Check if files exist
    if not train_file.exists() or not test_file.exists():
        report["status"] = "INCOMPLETE"
        report["message"] = "One or more dataset files are missing."
        return report
    
    print("Loading training set...")
    df_train = load_raw_dataset("training")
    print(f"  Training set shape: {df_train.shape}")
    
    print("Loading testing set...")
    df_test = load_raw_dataset("testing")
    print(f"  Testing set shape: {df_test.shape}")
    
    # Schema validation
    print("Validating schemas...")
    report["schema"] = {
        "training": validate_dataset_schema(df_train, "training"),
        "testing": validate_dataset_schema(df_test, "testing"),
    }
    
    # Data quality
    print("Analyzing data quality (training set)...")
    report["quality"] = {
        "training": analyze_data_quality(df_train, "training"),
    }
    print("Analyzing data quality (testing set)...")
    report["quality"]["testing"] = analyze_data_quality(df_test, "testing")
    
    # Class distribution
    print("Analyzing class distribution...")
    report["class_distribution"] = {
        "training": analyze_class_distribution(df_train, "training"),
        "testing": analyze_class_distribution(df_test, "testing"),
    }
    
    # Train/test overlap
    print("Checking train/test overlap...")
    report["overlap"] = check_train_test_overlap(df_train, df_test)
    
    # Leakage risks
    print("Identifying leakage risks...")
    report["leakage_risks"] = identify_leakage_risks(df_train)
    
    # Save report as YAML
    report_path = output_dir / "dataset_report.yaml"
    
    # Convert to serializable format
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        return obj
    
    serializable_report = make_serializable(report)
    with open(report_path, "w") as f:
        yaml.dump(serializable_report, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nReport saved to: {report_path}")
    report["status"] = "COMPLETE"
    
    return report


def print_dataset_summary(report: Dict):
    """Print a human-readable summary of the dataset report.
    
    Args:
        report: Dataset report dictionary from generate_dataset_report().
    """
    print("\n" + "=" * 70)
    print("UNSW-NB15 DATASET SUMMARY REPORT")
    print("=" * 70)
    
    # File info
    print("\n--- File Information ---")
    for split in ("training", "testing"):
        info = report["files"][split]
        if info["exists"]:
            print(f"  {split.capitalize()}: {info['filename']}")
            print(f"    Size: {info['size_mb']} MB ({info['size_bytes']} bytes)")
            print(f"    SHA-256: {info['sha256'][:16]}...")
        else:
            print(f"  {split.capitalize()}: NOT FOUND")
    
    # Schema
    print("\n--- Schema ---")
    for split in ("training", "testing"):
        schema = report["schema"][split]
        print(f"  {split.capitalize()}: {schema['n_rows']} rows × {schema['n_columns']} columns")
        if schema["issues"]:
            for issue in schema["issues"]:
                print(f"    ⚠ {issue}")
    
    # Class distribution
    print("\n--- Class Distribution ---")
    for split in ("training", "testing"):
        dist = report["class_distribution"][split]
        print(f"\n  {split.capitalize()} set:")
        
        if "binary_distribution" in dist:
            print(f"    Binary: {dist['binary_distribution']}")
        
        if "multiclass_distribution" in dist:
            print(f"    Attack categories ({dist.get('n_classes', 'N/A')} classes):")
            for cat, count in sorted(dist["multiclass_distribution"].items(), 
                                    key=lambda x: -x[1]):
                pct = dist["multiclass_percentage"][cat]
                print(f"      {cat:20s}: {count:>7,d} ({pct:5.1f}%)")
            
            if "imbalance_ratio" in dist:
                print(f"    Imbalance ratio: {dist['imbalance_ratio']}")
    
    # Data quality
    print("\n--- Data Quality ---")
    for split in ("training", "testing"):
        q = report["quality"][split]
        print(f"\n  {split.capitalize()} set:")
        print(f"    Total missing cells: {q['total_missing_cells']}")
        if q["missing_values"]:
            print(f"    Columns with missing values:")
            for col, info in q["missing_values"].items():
                print(f"      {col}: {info['count']} ({info['percentage']}%)")
        else:
            print(f"    No missing values found.")
        
        print(f"    Infinity values: {len(q['infinity_values'])} columns")
        if q["infinity_values"]:
            for col, count in q["infinity_values"].items():
                print(f"      {col}: {count}")
        
        print(f"    Duplicate rows: {q['duplicate_rows']} ({q['duplicate_percentage']}%)")
        print(f"    Constant columns: {q['constant_columns'] if q['constant_columns'] else 'None'}")
        print(f"    Numerical features: {len(q['column_types']['numerical'])}")
        print(f"    Categorical features: {len(q['column_types']['categorical'])}")
    
    # Overlap
    print("\n--- Train/Test Overlap ---")
    overlap = report["overlap"]
    print(f"  Feature columns compared: {overlap['feature_columns_compared']}")
    print(f"  Overlapping records: {overlap['overlapping_records']} ({overlap['overlap_percentage_of_test']}% of test set)")
    
    # Leakage risks
    print("\n--- Potential Leakage Risks ---")
    if report["leakage_risks"]:
        for risk in report["leakage_risks"]:
            print(f"  [{risk['severity']}] {risk['column']}: {risk['risk']}")
            print(f"    -> {risk['recommendation']}")
    else:
        print("  No leakage risks identified.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("UNSW-NB15 Dataset Exploration - Phase 1")
    print("=" * 70)
    
    report = generate_dataset_report()
    print_dataset_summary(report)
