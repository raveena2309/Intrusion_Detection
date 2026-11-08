# Dataset Documentation

## UNSW-NB15 Dataset

### Source
- **Official source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **Download mirror used:** GitHub (raw.githubusercontent.com/Nir-J/ML-Projects)
- **Download date:** 2026-11-08

### Files

| File | Rows | Columns | Size | SHA-256 (first 16 chars) |
|------|------|---------|------|--------------------------|
| UNSW_NB15_training-set.csv | 175,341 | 45 | 30.80 MB | bec7dd5ec88dc2a0 |
| UNSW_NB15_testing-set.csv | 82,332 | 45 | 14.67 MB | 734fe6642edf758f |

### Column Schema (45 columns)

| # | Column | Type | Description | Role |
|---|--------|------|-------------|------|
| 0 | id | int64 | Row identifier | **EXCLUDE** (leakage risk) |
| 1 | dur | float64 | Record total duration | Feature |
| 2 | proto | string | Transaction protocol | Feature (categorical) |
| 3 | service | string | Network service (http, ftp, etc.) | Feature (categorical) |
| 4 | state | string | Protocol state (FIN, CON, etc.) | Feature (categorical) |
| 5 | spkts | int64 | Source-to-dest packet count | Feature |
| 6 | dpkts | int64 | Dest-to-source packet count | Feature |
| 7 | sbytes | int64 | Source-to-dest bytes | Feature |
| 8 | dbytes | int64 | Dest-to-source bytes | Feature |
| 9 | rate | float64 | Packets per second | Feature |
| 10 | sttl | int64 | Source TTL value | Feature |
| 11 | dttl | int64 | Dest TTL value | Feature |
| 12 | sload | float64 | Source bits per second | Feature |
| 13 | dload | float64 | Dest bits per second | Feature |
| 14 | sloss | int64 | Source packets retransmitted/dropped | Feature |
| 15 | dloss | int64 | Dest packets retransmitted/dropped | Feature |
| 16 | sinpkt | float64 | Source inter-packet arrival time | Feature |
| 17 | dinpkt | float64 | Dest inter-packet arrival time | Feature |
| 18 | sjit | float64 | Source jitter | Feature |
| 19 | djit | float64 | Dest jitter | Feature |
| 20 | swin | int64 | Source TCP window size | Feature |
| 21 | stcpb | int64 | Source TCP base seq number | Feature |
| 22 | dtcpb | int64 | Dest TCP base seq number | Feature |
| 23 | dwin | int64 | Dest TCP window size | Feature |
| 24 | tcprtt | float64 | TCP round trip time | Feature |
| 25 | synack | float64 | SYN-ACK time | Feature |
| 26 | ackdat | float64 | ACK data time | Feature |
| 27 | smean | int64 | Mean source packet size | Feature |
| 28 | dmean | int64 | Mean dest packet size | Feature |
| 29 | trans_depth | int64 | HTTP transaction depth | Feature |
| 30 | response_body_len | int64 | HTTP response body length | Feature |
| 31 | ct_srv_src | int64 | Connection count from same src to same service | Feature |
| 32 | ct_state_ttl | int64 | Connection count per state and TTL | Feature |
| 33 | ct_dst_ltm | int64 | Connection count to same dest in last time window | Feature |
| 34 | ct_src_dport_ltm | int64 | Connection count from same src port in last time window | Feature |
| 35 | ct_dst_sport_ltm | int64 | Connection count to same dest port in last time window | Feature |
| 36 | ct_dst_src_ltm | int64 | Connection count between src-dest pair in last time window | Feature |
| 37 | is_ftp_login | int64 | FTP session with login | Feature |
| 38 | ct_ftp_cmd | int64 | Count of FTP commands in session | Feature |
| 39 | ct_flw_http_mthd | int64 | Count of HTTP methods in flow | Feature |
| 40 | ct_src_ltm | int64 | Connection count from same src in last time window | Feature |
| 41 | ct_srv_dst | int64 | Connection count to same service from dest | Feature |
| 42 | is_sm_ips_ports | int64 | Same src/dest IPs and ports | Feature |
| 43 | attack_cat | string | Attack category label | **TARGET** (multiclass) |
| 44 | label | int64 | Binary label (0=normal, 1=attack) | **TARGET** (binary) |

### Class Distribution

#### Training Set (175,341 records)

| Category | Count | Percentage |
|----------|-------|------------|
| Normal | 56,000 | 31.9% |
| Generic | 40,000 | 22.8% |
| Exploits | 33,393 | 19.0% |
| Fuzzers | 18,184 | 10.4% |
| DoS | 12,264 | 7.0% |
| Reconnaissance | 10,491 | 6.0% |
| Analysis | 2,000 | 1.1% |
| Backdoor | 1,746 | 1.0% |
| Shellcode | 1,133 | 0.7% |
| Worms | 130 | 0.1% |

#### Testing Set (82,332 records)

| Category | Count | Percentage |
|----------|-------|------------|
| Normal | 37,000 | 44.9% |
| Generic | 18,871 | 22.9% |
| Exploits | 11,132 | 13.5% |
| Fuzzers | 6,062 | 7.4% |
| DoS | 4,089 | 5.0% |
| Reconnaissance | 3,496 | 4.2% |
| Analysis | 677 | 0.8% |
| Backdoor | 583 | 0.7% |
| Shellcode | 378 | 0.5% |
| Worms | 44 | 0.1% |

### Data Quality Summary

- **Missing values:** None in either split
- **Infinity values:** None detected
- **Duplicate rows:** None in either split
- **Constant columns:** None
- **Train/test overlap:** 8,541 records (10.37% of test set)

### Categorical Feature Details

| Feature | Train Unique | Test Unique | Values |
|---------|-------------|-------------|--------|
| proto | 133 | 133 | tcp, udp, arp, ospf, icmp, igmp, ... |
| service | 13 | 13 | -, dns, http, smtp, ftp-data, ftp, ssh, pop3, dhcp, snmp, ssl, irc, radius |
| state | 9 | 8 | INT, FIN, CON, REQ, RST, ECO, PAR, URN, no |

### Leakage Risks Identified

1. **`id` column (HIGH):** Correlation 0.7272 with binary label. The ID is assigned sequentially and the dataset appears ordered such that normal records tend to have lower IDs. **Must be excluded from features.**

2. **`label` and `attack_cat`:** These are target columns and must not be used as input features.

### Notes

- The dataset uses the official pre-partitioned training/testing split
- Imbalance ratio (max/min class) is ~431:1 in training, ~841:1 in testing
- `is_ftp_login` has values {0, 1, 2, 4} — not strictly binary despite the name
- `ct_ftp_cmd` has identical distribution to `is_ftp_login`
- Service value `-` represents "no service identified" (most common: 53.7% of training)
- The `state` column has rare values (PAR, URN, no: 1 each in training)
