"""Flow extraction from PCAP files using Scapy.

Each bidirectional flow is identified by the 5-tuple
(src_ip, dst_ip, src_port, dst_port, protocol).  Only the subset of UNSW-NB15
features that can be derived from raw packets is extracted; all other features
are left as NaN.

Requirements
------------
- Scapy must be installed: ``pip install scapy``
- Reading PCAP files does NOT require elevated privileges.
"""

import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from traffic.schema_mapping import (
    PCAP_MAPPABLE_FEATURES,
    PCAP_UNAVAILABLE_FEATURES,
    UNSWNB15_FEATURES,
    get_feature_defaults,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FlowKey:
    """Five-tuple that uniquely identifies a bidirectional network flow.

    The key is *canonical*: the lower (src_ip, src_port) pair is always stored
    as the source so that forward and reverse packets map to the same key.
    """

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # IP protocol number (6=TCP, 17=UDP, 1=ICMP, …)

    def __hash__(self) -> int:  # noqa: D105
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))

    def __eq__(self, other: object) -> bool:  # noqa: D105
        if not isinstance(other, FlowKey):
            return NotImplemented
        return (
            self.src_ip == other.src_ip
            and self.dst_ip == other.dst_ip
            and self.src_port == other.src_port
            and self.dst_port == other.dst_port
            and self.protocol == other.protocol
        )


@dataclass
class FlowRecord:
    """Extracted flow record containing UNSW-NB15 compatible fields.

    Fields that cannot be derived from raw PCAP are set to ``None`` /
    ``float('nan')`` and will be replaced by safe defaults before inference.
    """

    # --- identifiers (not model features) ---
    flow_key: Optional[FlowKey] = None
    start_time: float = 0.0  # epoch seconds of first packet

    # --- PCAP-extractable (PCAP_MAPPABLE_FEATURES) ---
    proto: str = "other"
    service: str = "-"
    state: str = "CON"
    dur: float = 0.0
    spkts: int = 0
    dpkts: int = 0
    sbytes: int = 0
    dbytes: int = 0
    sttl: float = 0.0
    dttl: float = 0.0
    swin: int = 0
    dwin: int = 0

    # --- NOT derivable from raw PCAP (will be NaN) ---
    rate: float = float("nan")
    sload: float = float("nan")
    dload: float = float("nan")
    sloss: float = float("nan")
    dloss: float = float("nan")
    sinpkt: float = float("nan")
    dinpkt: float = float("nan")
    sjit: float = float("nan")
    djit: float = float("nan")
    stcpb: float = float("nan")
    dtcpb: float = float("nan")
    tcprtt: float = float("nan")
    synack: float = float("nan")
    ackdat: float = float("nan")
    smean: float = float("nan")
    dmean: float = float("nan")
    trans_depth: float = float("nan")
    response_body_len: float = float("nan")
    ct_srv_src: float = float("nan")
    ct_state_ttl: float = float("nan")
    ct_dst_ltm: float = float("nan")
    ct_src_dport_ltm: float = float("nan")
    ct_dst_sport_ltm: float = float("nan")
    ct_dst_src_ltm: float = float("nan")
    is_ftp_login: float = float("nan")
    ct_ftp_cmd: float = float("nan")
    ct_flw_http_mthd: float = float("nan")
    ct_src_ltm: float = float("nan")
    ct_srv_dst: float = float("nan")
    is_sm_ips_ports: float = float("nan")


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------

_PROTO_NAMES: Dict[int, str] = {
    1: "icmp",
    6: "tcp",
    17: "udp",
    58: "ipv6-icmp",
    132: "sctp",
}

_SERVICE_PORT_MAP: Dict[int, str] = {
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "https",
    3306: "mysql",
    3389: "rdp",
    8080: "http",
}


def _proto_name(proto_num: int) -> str:
    return _PROTO_NAMES.get(proto_num, str(proto_num))


def _service_from_ports(src_port: int, dst_port: int) -> str:
    svc = _SERVICE_PORT_MAP.get(dst_port) or _SERVICE_PORT_MAP.get(src_port)
    return svc if svc else "-"


# ---------------------------------------------------------------------------
# Flow builder (internal)
# ---------------------------------------------------------------------------

class _FlowBuilder:
    """Accumulates packets belonging to a single bidirectional flow."""

    def __init__(self, key: FlowKey, first_packet_time: float) -> None:
        self.key = key
        self.start_time = first_packet_time
        self.last_time = first_packet_time

        self._src = (key.src_ip, key.src_port)

        # Per-direction packet lists: (time, length, ttl, tcp_window)
        self._fwd_pkts: List[Tuple[float, int, int, int]] = []
        self._bwd_pkts: List[Tuple[float, int, int, int]] = []

    def add_packet(
        self,
        pkt_time: float,
        src_ip: str,
        src_port: int,
        pkt_len: int,
        ttl: int,
        tcp_window: int,
    ) -> None:
        """Register one packet."""
        self.last_time = max(self.last_time, pkt_time)
        entry = (pkt_time, pkt_len, ttl, tcp_window)
        if (src_ip, src_port) == self._src:
            self._fwd_pkts.append(entry)
        else:
            self._bwd_pkts.append(entry)

    def build(self) -> FlowRecord:
        """Produce a :class:`FlowRecord` from accumulated packets."""
        rec = FlowRecord()
        rec.flow_key = self.key
        rec.start_time = self.start_time

        dur = self.last_time - self.start_time
        rec.dur = max(dur, 0.0)
        rec.proto = _proto_name(self.key.protocol)
        rec.service = _service_from_ports(self.key.src_port, self.key.dst_port)
        rec.state = "CON"  # simplified; full state-machine would need TCP flags

        rec.spkts = len(self._fwd_pkts)
        rec.dpkts = len(self._bwd_pkts)
        rec.sbytes = sum(p[1] for p in self._fwd_pkts)
        rec.dbytes = sum(p[1] for p in self._bwd_pkts)

        rec.sttl = (
            float(np.mean([p[2] for p in self._fwd_pkts])) if self._fwd_pkts else 0.0
        )
        rec.dttl = (
            float(np.mean([p[2] for p in self._bwd_pkts])) if self._bwd_pkts else 0.0
        )
        rec.swin = self._fwd_pkts[-1][3] if self._fwd_pkts else 0
        rec.dwin = self._bwd_pkts[-1][3] if self._bwd_pkts else 0

        return rec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_flows_from_pcap(
    pcap_path: str,
    timeout: float = 120.0,
) -> List[FlowRecord]:
    """Extract bidirectional network flows from a PCAP file using Scapy.

    Only the features listed in :data:`~traffic.schema_mapping.PCAP_MAPPABLE_FEATURES`
    are populated.  All other UNSW-NB15 features remain as NaN and must be
    filled with defaults before inference.

    Args:
        pcap_path: Absolute or relative path to the ``.pcap`` / ``.pcapng`` file.
        timeout: Maximum seconds to spend parsing.  Parsing stops early if the
            wall-clock limit is exceeded (unused in offline mode but kept for
            API symmetry with live capture).

    Returns:
        List of :class:`FlowRecord` instances, one per bidirectional flow.

    Raises:
        FileNotFoundError: If *pcap_path* does not exist.
        ImportError: If Scapy is not installed.
        RuntimeError: If the PCAP file cannot be parsed.
    """
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    try:
        from scapy.utils import rdpcap
        from scapy.layers.inet import IP, TCP, UDP
        import scapy.packet  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Scapy is required for PCAP processing. "
            "Install it with: pip install scapy"
        ) from exc

    logger.info("Reading PCAP file: %s", pcap_path)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            packets = rdpcap(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PCAP file '{pcap_path}': {exc}") from exc

    logger.info("Loaded %d packets from PCAP.", len(packets))

    flows: Dict[FlowKey, _FlowBuilder] = {}

    for pkt in packets:
        try:
            if not pkt.haslayer(IP):
                continue

            ip_layer = pkt[IP]
            src_ip: str = ip_layer.src
            dst_ip: str = ip_layer.dst
            proto_num: int = ip_layer.proto
            ttl: int = ip_layer.ttl
            pkt_len: int = len(pkt)
            pkt_time: float = float(pkt.time)
            tcp_window: int = 0
            src_port: int = 0
            dst_port: int = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                tcp_window = int(tcp.window)
            elif pkt.haslayer(UDP):
                udp = pkt["UDP"]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)

            # Canonical key: lower (IP, port) tuple is always "src"
            if (src_ip, src_port) > (dst_ip, dst_port):
                src_ip, dst_ip = dst_ip, src_ip
                src_port, dst_port = dst_port, src_port

            key = FlowKey(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=proto_num,
            )

            if key not in flows:
                flows[key] = _FlowBuilder(key, pkt_time)

            flows[key].add_packet(
                pkt_time=pkt_time,
                src_ip=src_ip,
                src_port=src_port,
                pkt_len=pkt_len,
                ttl=ttl,
                tcp_window=tcp_window,
            )

        except Exception as pkt_exc:  # noqa: BLE001
            logger.debug("Skipped malformed packet: %s", pkt_exc)
            continue

    records = [builder.build() for builder in flows.values()]
    logger.info("Extracted %d flows from PCAP.", len(records))
    return records


def flows_to_dataframe(flows: List[FlowRecord]) -> pd.DataFrame:
    """Convert a list of :class:`FlowRecord` instances to a pandas DataFrame.

    The returned DataFrame has exactly the columns listed in
    :data:`~traffic.schema_mapping.UNSWNB15_FEATURES`.  Features that could not
    be extracted from PCAP are ``NaN``.

    Args:
        flows: List of flow records (output of :func:`extract_flows_from_pcap`).

    Returns:
        DataFrame with one row per flow and UNSW-NB15 column names.
    """
    if not flows:
        return pd.DataFrame(columns=UNSWNB15_FEATURES)

    rows = []
    defaults = get_feature_defaults()

    for rec in flows:
        row: Dict = {}
        for feat in UNSWNB15_FEATURES:
            val = getattr(rec, feat, None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                row[feat] = defaults.get(feat, float("nan"))
            else:
                row[feat] = val
        rows.append(row)

    df = pd.DataFrame(rows, columns=UNSWNB15_FEATURES)
    return df


def get_flow_extraction_warnings(df: pd.DataFrame) -> List[str]:
    """Return human-readable warnings about features that could not be extracted.

    Args:
        df: DataFrame produced by :func:`flows_to_dataframe`.

    Returns:
        List of warning strings, one per unavailable feature group.
    """
    warnings_out: List[str] = []

    if df.empty:
        warnings_out.append("No flows extracted — the PCAP may be empty or contain no IP packets.")
        return warnings_out

    unavailable_present = [f for f in PCAP_UNAVAILABLE_FEATURES if f in df.columns]
    if unavailable_present:
        warnings_out.append(
            f"{len(unavailable_present)} features could not be extracted from PCAP and were "
            f"filled with defaults: {unavailable_present[:5]}{'...' if len(unavailable_present) > 5 else ''}."
        )

    # Flag columns that are entirely zero (may indicate extraction issues)
    all_zero = [
        col
        for col in PCAP_MAPPABLE_FEATURES
        if col in df.columns and df[col].dtype != object and (df[col] == 0).all()
    ]
    if all_zero:
        warnings_out.append(
            f"The following extracted features are all-zero (possible extraction issue): {all_zero}."
        )

    return warnings_out
