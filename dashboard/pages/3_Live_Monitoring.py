"""dashboard/pages/3_Live_Monitoring.py"""
import streamlit as st
st.set_page_config(page_title="Live Monitoring | NIDS", page_icon="shield", layout="wide")
st.markdown("""<style>.stApp{background-color:#0e1117}h1,h2,h3{color:#00d4aa}</style>""",
            unsafe_allow_html=True)
st.title("Live Monitoring")
st.warning("Live capture requires administrator/root privileges and is intended for authorized test networks only.")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

try:
    from traffic.packet_capture import PacketCapture, list_interfaces
    if hasattr(PacketCapture, "list_interfaces"):
        interfaces = PacketCapture.list_interfaces()
    else:
        interfaces = list_interfaces()
except Exception:
    interfaces = ["Wi-Fi", "Ethernet", "Loopback"]

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### Capture Settings")
    if interfaces:
        selected_iface = st.selectbox("Network Interface", interfaces)
    else:
        selected_iface = st.text_input("Interface Name", placeholder="e.g. eth0, Wi-Fi")

    col_a, col_b = st.columns(2)
    with col_a:
        start_btn = st.button("Start Capture", type="primary", disabled=True)
    with col_b:
        stop_btn = st.button("Stop Capture", disabled=True)

    st.info("Live capture is disabled in this version. Upload a PCAP or CSV on the Data Input page.")

with col2:
    st.markdown("### Status")
    st.markdown("""
    | Field | Value |
    |-------|-------|
    | Status | Inactive |
    | Mode | Dataset Batch (not live) |
    | Packets Captured | 0 |
    | Flows Extracted | 0 |
    """)
    st.markdown("---")
    st.markdown("**Note:** Live monitoring would display real-time flow predictions here.")
    st.caption("Do not use live capture on unauthorized networks. Requires Scapy and admin rights.")
