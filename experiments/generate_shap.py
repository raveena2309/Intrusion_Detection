"""
experiments/generate_shap.py
==============================

Phase 5: SHAP explainability for the best classifier (XGBoost).
"""

from __future__ import annotations
import sys
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explainability import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_feature_importance,
    save_shap_artifacts,
)
from preprocessing import NIDSPreprocessor

LABEL_NAMES = [
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms",
]


def main():
    print("=" * 70)
    print("Phase 5 - SHAP Explainability Experiment")
    print("=" * 70)

    xgb_path = PROJECT_ROOT / "models" / "classifier" / "xgboost.joblib"
    if not xgb_path.exists():
        print("[PENDING] XGBoost model not found. Run Phase 2 first.")
        sys.exit(0)

    model = joblib.load(xgb_path)
    print(f"[generate_shap] XGBoost model loaded")

    preprocessor = NIDSPreprocessor.load(str(PROJECT_ROOT / "models" / "preprocessing"))
    print("[generate_shap] Preprocessor loaded")

    test_csv = PROJECT_ROOT / "datasets" / "raw" / "UNSW_NB15_testing-set.csv"
    df_test = pd.read_csv(test_csv)
    X_test = preprocessor.transform(df_test)
    feature_names = preprocessor.feature_names
    print(f"[generate_shap] Test data shape: {X_test.shape}")

    print("\n--- Computing SHAP values (max_samples=500) ---")
    shap_dict = compute_shap_values(
        model=model, X=X_test, feature_names=feature_names, max_samples=500,
    )
    print(f"[generate_shap] SHAP computed on {shap_dict['n_samples']} samples")

    shap_collapsed = shap_dict["shap_values"]
    mean_abs = np.abs(shap_collapsed).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("\nTop-10 most important features (mean |SHAP|):")
    print(f"{'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}")
    print("-" * 50)
    for rank, i in enumerate(order[:10], 1):
        print(f"{rank:<5} {feature_names[i]:<30} {mean_abs[i]:.6f}")

    figures_dir = PROJECT_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_shap_summary(
        shap_dict=shap_dict, X=X_test, feature_names=feature_names,
        save_path=str(figures_dir / "shap_summary.png"), max_display=20,
    )

    plot_shap_feature_importance(
        shap_dict=shap_dict, feature_names=feature_names,
        save_path=str(figures_dir / "shap_importance.png"), top_n=20,
    )

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_shap_artifacts(shap_dict, str(metrics_dir))

    print("\n[generate_shap] All SHAP artifacts saved.")
    print("=" * 70)
    print("Phase 5 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
