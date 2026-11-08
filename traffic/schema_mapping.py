"""Schema mapping between PCAP-extracted features and UNSW-NB15 model features.

This module defines the canonical feature set used by the NIDS model, provides
validation utilities, and documents which features can be extracted from raw PCAP
versus which require specialised tools (like NetMate or Argus).
"""

from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Canonical UNSW-NB15 feature list (42 features, excluding id/label/attack_cat)
# ---------------------------------------------------------------------------

UNSWNB15_FEATURES: List[str] = [
    # Categorical (3)
    "proto",
    "service",
    "state",
    # Flow duration / volume (8)
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    # Load / loss (4)
    "sload",
    "dload",
    "sloss",
    "dloss",
    # Inter-packet timing (4)
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    # TCP window / sequence (5)
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    # TCP handshake timing (3)
    "tcprtt",
    "synack",
    "ackdat",
    # Mean payload sizes (2)
    "smean",
    "dmean",
    # HTTP / FTP counts (2)
    "trans_depth",
    "response_body_len",
    # Connection features (12)
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
]

assert len(UNSWNB15_FEATURES) == 42, "UNSWNB15_FEATURES must contain exactly 42 features."


# ---------------------------------------------------------------------------
# PCAP-mappable vs unavailable features
# ---------------------------------------------------------------------------

PCAP_MAPPABLE_FEATURES: List[str] = [
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "swin",
    "dwin",
]
"""Subset of UNSWNB15_FEATURES that can be extracted directly from raw PCAP using Scapy.

These are basic flow-level statistics available without specialised network
measurement tools.
"""

PCAP_UNAVAILABLE_FEATURES: List[str] = [
    f for f in UNSWNB15_FEATURES if f not in PCAP_MAPPABLE_FEATURES
]
"""Features in UNSWNB15_FEATURES that are NOT derivable from raw PCAP without
specialised tools such as Argus, NetMate, or the UNSW tool-chain.

Includes computed statistics like jitter, load, connection-table counts, and
FTP/HTTP application-layer counters.  These will be set to NaN when processing
live/PCAP traffic and filled with safe defaults before inference.
"""


# ---------------------------------------------------------------------------
# Safe defaults for missing features
# ---------------------------------------------------------------------------

def get_feature_defaults() -> Dict[str, float]:
    """Return safe default values for features that cannot be extracted from PCAP.

    Defaults are chosen to be neutral (zero or typical small values) so that
    the model is not steered by imputed data.  The preprocessor's fitted imputer
    will override these anyway during the transform step.

    Returns:
        Dictionary mapping feature name → default numeric value.
    """
    defaults: Dict[str, float] = {
        # Categorical defaults (will be encoded as strings elsewhere)
        "proto": 0.0,
        "service": 0.0,
        "state": 0.0,
        # Timing / volume
        "dur": 0.0,
        "spkts": 0,
        "dpkts": 0,
        "sbytes": 0,
        "dbytes": 0,
        "rate": 0.0,
        "sttl": 64.0,
        "dttl": 64.0,
        "sload": 0.0,
        "dload": 0.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 0.0,
        "dinpkt": 0.0,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 0,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 0,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 0.0,
        "dmean": 0.0,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 0,
        "ct_state_ttl": 0,
        "ct_dst_ltm": 0,
        "ct_src_dport_ltm": 0,
        "ct_dst_sport_ltm": 0,
        "ct_dst_src_ltm": 0,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 0,
        "ct_srv_dst": 0,
        "is_sm_ips_ports": 0,
    }
    return defaults


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_feature_schema(df: pd.DataFrame) -> Dict:
    """Validate a DataFrame against the expected UNSW-NB15 feature schema.

    Args:
        df: DataFrame to validate.  Columns are compared against
            :data:`UNSWNB15_FEATURES`.

    Returns:
        Dictionary with keys:

        - ``valid`` (bool): True if all required features are present.
        - ``missing`` (list[str]): Features in schema but absent from *df*.
        - ``extra`` (list[str]): Columns in *df* not in schema (informational).
        - ``issues`` (list[str]): Human-readable problem descriptions.
    """
    available = set(df.columns)
    expected = set(UNSWNB15_FEATURES)

    missing = sorted(expected - available)
    extra = sorted(available - expected)
    issues: List[str] = []

    if missing:
        issues.append(f"{len(missing)} required feature(s) missing: {missing}")

    # Warn on NaN-heavy columns
    for col in UNSWNB15_FEATURES:
        if col in df.columns:
            nan_frac = df[col].isna().mean()
            if nan_frac > 0.5:
                issues.append(
                    f"Column '{col}' is {nan_frac:.0%} NaN — likely not extracted from PCAP."
                )

    return {
        "valid": len(missing) == 0,
        "missing": missing,
        "extra": extra,
        "issues": issues,
    }
