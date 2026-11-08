"""
Feature Engineering Module for Intelligent NIDS
=================================================

Optional feature transformations for the UNSW-NB15 dataset.
All transformations are applied after basic preprocessing.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio-based derived features.
    
    These ratios capture relationships between existing flow features
    that may be informative for classification.
    
    Args:
        df: DataFrame with raw UNSW-NB15 features.
    
    Returns:
        DataFrame with additional ratio features appended.
    """
    df = df.copy()
    
    # Bytes ratio (source vs dest)
    total_bytes = df["sbytes"] + df["dbytes"]
    df["bytes_ratio"] = np.where(total_bytes > 0, df["sbytes"] / total_bytes, 0.5)
    
    # Packets ratio
    total_pkts = df["spkts"] + df["dpkts"]
    df["pkts_ratio"] = np.where(total_pkts > 0, df["spkts"] / total_pkts, 0.5)
    
    # Mean packet size ratio
    total_mean = df["smean"] + df["dmean"]
    df["mean_pkt_ratio"] = np.where(total_mean > 0, df["smean"] / total_mean, 0.5)
    
    # Loss ratio
    total_loss = df["sloss"] + df["dloss"]
    df["loss_ratio"] = np.where(total_loss > 0, df["sloss"] / total_loss, 0.5)
    
    # Load ratio
    total_load = df["sload"] + df["dload"]
    df["load_ratio"] = np.where(total_load > 0, df["sload"] / total_load, 0.5)
    
    # Jitter ratio
    total_jit = df["sjit"] + df["djit"]
    df["jit_ratio"] = np.where(total_jit > 0, df["sjit"] / total_jit, 0.5)
    
    return df


def add_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add aggregated features combining multiple columns.
    
    Args:
        df: DataFrame with raw UNSW-NB15 features.
    
    Returns:
        DataFrame with additional aggregated features.
    """
    df = df.copy()
    
    # Total bytes and packets
    df["total_bytes"] = df["sbytes"] + df["dbytes"]
    df["total_pkts"] = df["spkts"] + df["dpkts"]
    
    # Bytes per packet
    df["bytes_per_pkt_src"] = np.where(df["spkts"] > 0, df["sbytes"] / df["spkts"], 0)
    df["bytes_per_pkt_dst"] = np.where(df["dpkts"] > 0, df["dbytes"] / df["dpkts"], 0)
    
    # TCP handshake features
    df["tcp_setup_time"] = df["synack"] + df["ackdat"]
    
    # Connection density (sum of ct_* features)
    ct_cols = [c for c in df.columns if c.startswith("ct_")]
    if ct_cols:
        df["ct_sum"] = df[ct_cols].sum(axis=1)
    
    return df


def add_log_features(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Add log-transformed versions of skewed numerical features.
    
    Args:
        df: DataFrame with raw features.
        columns: Columns to log-transform. If None, uses default set.
    
    Returns:
        DataFrame with log-transformed features appended.
    """
    if columns is None:
        columns = ["sbytes", "dbytes", "sload", "dload", "rate",
                    "sinpkt", "dinpkt", "sjit", "djit", "response_body_len"]
    
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    
    return df


def get_feature_engineering_config() -> dict:
    """Return the feature engineering configuration.
    
    Returns:
        Dictionary describing all engineered features.
    """
    return {
        "ratio_features": [
            "bytes_ratio", "pkts_ratio", "mean_pkt_ratio",
            "loss_ratio", "load_ratio", "jit_ratio",
        ],
        "aggregated_features": [
            "total_bytes", "total_pkts", "bytes_per_pkt_src",
            "bytes_per_pkt_dst", "tcp_setup_time", "ct_sum",
        ],
        "log_features": [
            "sbytes_log", "dbytes_log", "sload_log", "dload_log",
            "rate_log", "sinpkt_log", "dinpkt_log", "sjit_log",
            "djit_log", "response_body_len_log",
        ],
        "description": (
            "Feature engineering adds ratio, aggregated, and log-transformed "
            "features. These are optional and must be evaluated against the "
            "base 42-feature model to verify they improve performance (E7)."
        ),
    }
