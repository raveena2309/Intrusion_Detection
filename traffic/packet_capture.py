"""Authorized local packet capture using Scapy.

.. warning::
    Live packet capture requires **administrator / root privileges** on all
    major operating systems.  On Windows, run as Administrator.  On Linux/macOS,
    run as root or with ``CAP_NET_RAW`` capability.

    This module captures packets *only* on the local machine's network
    interfaces.  It is intended for authorized security monitoring only.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface listing
# ---------------------------------------------------------------------------


def list_interfaces() -> List[str]:
    """Return available network interface names on this host.

    Uses Scapy's interface database.  Returns an empty list if Scapy is not
    available or the interface list cannot be retrieved.

    Returns:
        List of interface name strings (e.g. ``['eth0', 'lo']`` on Linux,
        ``['\\\\Device\\\\NPF_{...}']`` on Windows).
    """
    try:
        from scapy.arch import get_if_list  # type: ignore
        return get_if_list()
    except ImportError:
        logger.error("Scapy is not installed.  Install with: pip install scapy")
        return []
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not enumerate interfaces: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Packet capture
# ---------------------------------------------------------------------------


class PacketCapture:
    """Non-blocking live packet capture using Scapy's AsyncSniffer.

    .. warning::
        Requires administrator / root privileges.

    Example usage::

        capture = PacketCapture()
        capture.start(interface="eth0", callback=my_callback)
        # ... do other work ...
        capture.stop()

    The *callback* is invoked in a background thread for every captured
    packet.  It must be thread-safe.
    """

    def __init__(self) -> None:
        self._sniffer = None
        self._start_time: Optional[float] = None
        self._packet_count: int = 0
        self._lock = threading.Lock()
        self._errors: List[str] = []
        self._running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        interface: Optional[str] = None,
        callback: Optional[Callable] = None,
        bpf_filter: str = "",
    ) -> None:
        """Start non-blocking packet capture.

        Args:
            interface: Network interface name to sniff on.  Passes ``None``
                to Scapy to capture on the default/all interfaces.
            callback: Optional callable invoked with each captured Scapy
                packet.  If ``None``, packets are counted but discarded.
            bpf_filter: Optional BPF filter string (e.g. ``'tcp port 80'``).

        Raises:
            ImportError: If Scapy is not installed.
            RuntimeError: If capture is already running or Scapy raises an
                error during sniffer start (e.g. insufficient privileges).
        """
        if self._running:
            raise RuntimeError("Capture is already running. Call stop() first.")

        try:
            from scapy.sendrecv import AsyncSniffer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Scapy is required for live capture.  Install with: pip install scapy"
            ) from exc

        def _internal_callback(pkt):
            with self._lock:
                self._packet_count += 1
            if callback is not None:
                try:
                    callback(pkt)
                except Exception as cb_exc:  # noqa: BLE001
                    with self._lock:
                        self._errors.append(f"Callback error: {cb_exc}")

        kwargs: Dict = {"prn": _internal_callback, "store": False}
        if interface:
            kwargs["iface"] = interface
        if bpf_filter:
            kwargs["filter"] = bpf_filter

        try:
            self._sniffer = AsyncSniffer(**kwargs)
            self._sniffer.start()
            self._start_time = time.time()
            self._running = True
            logger.info(
                "Packet capture started on interface=%s filter='%s'.",
                interface or "all",
                bpf_filter,
            )
        except Exception as exc:
            self._errors.append(str(exc))
            self._running = False
            raise RuntimeError(
                f"Failed to start packet capture: {exc}. "
                "Ensure you have administrator/root privileges."
            ) from exc

    def stop(self) -> None:
        """Stop the running packet capture gracefully.

        Safe to call even if capture is not running (no-op in that case).
        All errors encountered during the capture session are logged.
        """
        if not self._running or self._sniffer is None:
            logger.debug("stop() called but capture is not running.")
            return

        try:
            self._sniffer.stop()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._errors.append(f"Stop error: {exc}")
            logger.warning("Error stopping sniffer: %s", exc)
        finally:
            self._running = False
            logger.info(
                "Packet capture stopped.  Total packets captured: %d",
                self._packet_count,
            )

        if self._errors:
            logger.warning(
                "Capture session encountered %d error(s): %s",
                len(self._errors),
                self._errors,
            )

    def get_status(self) -> Dict:
        """Return current capture status.

        Returns:
            Dictionary with keys:

            - ``running`` (bool): Whether capture is active.
            - ``packet_count`` (int): Number of packets captured so far.
            - ``duration_seconds`` (float): Elapsed time since capture started.
            - ``interface`` (str | None): Interface being sniffed.
            - ``errors`` (list[str]): Any errors that occurred.
        """
        with self._lock:
            duration = (
                time.time() - self._start_time if self._start_time and self._running else 0.0
            )
            iface: Optional[str] = None
            if self._sniffer is not None:
                iface = getattr(self._sniffer, "iface", None)

            return {
                "running": self._running,
                "packet_count": self._packet_count,
                "duration_seconds": round(duration, 2),
                "interface": iface,
                "errors": list(self._errors),
            }
