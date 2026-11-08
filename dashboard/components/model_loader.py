"""
dashboard/components/model_loader.py
======================================
Cached model loading for the Streamlit dashboard.
"""

import sys
import joblib
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@st.cache_resource(show_spinner="Loading models...")
def load_all_models():
    """Load all trained models. Returns None values if models not found."""
    from preprocessing import NIDSPreprocessor
    from inference import NIDSInferenceEngine

    models = {
        "preprocessor": None,
        "classifier": None,
        "anomaly_model": None,
        "fusion": None,
        "label_names": ["Analysis","Backdoor","DoS","Exploits","Fuzzers",
                         "Generic","Normal","Reconnaissance","Shellcode","Worms"],
        "ready": False,
        "missing": [],
    }

    try:
        prep_dir = PROJECT_ROOT / "models" / "preprocessing"
        if (prep_dir / "preprocessor.joblib").exists():
            models["preprocessor"] = NIDSPreprocessor.load(str(prep_dir))
        else:
            models["missing"].append("preprocessor")
    except Exception as e:
        models["missing"].append(f"preprocessor ({e})")

    try:
        clf_dir = PROJECT_ROOT / "models" / "classifier"
        for name in ("xgboost.joblib", "random_forest.joblib"):
            p = clf_dir / name
            if p.exists():
                models["classifier"] = joblib.load(p)
                models["classifier_name"] = name.replace(".joblib", "")
                break
        if models["classifier"] is None:
            models["missing"].append("classifier")
    except Exception as e:
        models["missing"].append(f"classifier ({e})")

    try:
        anom_dir = PROJECT_ROOT / "models" / "anomaly"
        anom_path = anom_dir / "isolation_forest.joblib"
        if anom_path.exists():
            models["anomaly_model"] = joblib.load(anom_path)
        else:
            models["missing"].append("anomaly_model")
    except Exception as e:
        models["missing"].append(f"anomaly_model ({e})")

    try:
        from decision_fusion import DecisionFusion
        calib_dir = PROJECT_ROOT / "models" / "calibration"
        fusion_path = calib_dir / "decision_fusion.joblib"
        if fusion_path.exists():
            models["fusion"] = DecisionFusion.load(str(calib_dir))
        else:
            # Use default thresholds
            models["fusion"] = DecisionFusion(confidence_threshold=0.6, anomaly_threshold=0.005)
    except Exception as e:
        models["missing"].append(f"fusion ({e})")

    if models["preprocessor"] and models["classifier"] and models["anomaly_model"]:
        try:
            models["engine"] = NIDSInferenceEngine(
                preprocessor=models["preprocessor"],
                classifier=models["classifier"],
                anomaly_model=models["anomaly_model"],
                fusion=models["fusion"],
                label_encoder=models["preprocessor"].label_encoder,
            )
            models["ready"] = True
        except Exception as e:
            models["missing"].append(f"inference engine ({e})")

    return models


def load_results():
    """Load saved experiment results."""
    import yaml
    results = {}
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    for fname in ("classifier_results.yaml", "anomaly_results.yaml",
                  "fusion_results.yaml", "dataset_report.yaml"):
        p = metrics_dir / fname
        if p.exists():
            with open(p) as f:
                results[fname.replace(".yaml", "")] = yaml.safe_load(f)

    # Normalize anomaly_results if nested under full_test_set
    if "anomaly_results" in results and isinstance(results["anomaly_results"], dict):
        anom = results["anomaly_results"]
        full_set = anom.get("full_test_set", {})
        if isinstance(full_set, dict):
            for k, v in full_set.items():
                if k not in anom:
                    anom[k] = v
            if "per_class_detection_rate" in full_set and "per_class_detection_rates" not in anom:
                anom["per_class_detection_rates"] = full_set["per_class_detection_rate"]

    return results


def load_shap_artifacts():
    """Load SHAP artifacts if available."""
    import pickle
    p = PROJECT_ROOT / "results" / "metrics" / "shap_artifacts.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None
