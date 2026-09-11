"""MITRE ATT&CK Stage Taxonomy and Labeling Rules for NetForecast.

Maps network anomalies and flow behaviors into the 5 core attack phases
defined by the NTRO PS 26153 specification:
  1. Reconnaissance
  2. Initial Access
  3. Lateral Movement
  4. Command & Control
  5. Exfiltration
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np

STAGE_NAMES = {
    0: "Benign",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command & Control",
    5: "Exfiltration"
}

STAGE_COLORS = {
    0: "#10b981",  # Emerald green
    1: "#38bdf8",  # Sky blue
    2: "#f59e0b",  # Amber
    3: "#8b5cf6",  # Purple
    4: "#ec4899",  # Pink
    5: "#ef4444"   # Red
}

STAGE_METADATA = {
    1: {
        "name": "Reconnaissance",
        "tactic_id": "TA0043",
        "technique": "T1046: Network Service Discovery",
        "description": "Adversary probing IP addresses and destination ports to map accessible services.",
        "soc_action": "Rate-limit SYN probing, isolate source IP on edge firewall, trigger port-scan alerts."
    },
    2: {
        "name": "Initial Access",
        "tactic_id": "TA0001",
        "technique": "T1190 / T1566: Exploit Public-Facing App / Phishing",
        "description": "Adversary executing exploit payloads or user opening malicious web links.",
        "soc_action": "Terminate active session, inspect endpoint browser/process logs, trigger EDR quarantine."
    },
    3: {
        "name": "Lateral Movement",
        "tactic_id": "TA0008",
        "technique": "T1021.002: SMB / Windows Admin Shares",
        "description": "Compromised host scanning and accessing internal workstations/servers via SMB/RDP.",
        "soc_action": "Segment VLAN, block port 445/3389 laterally, invalidate compromised domain credentials."
    },
    4: {
        "name": "Command & Control",
        "tactic_id": "TA0011",
        "technique": "T1071: Application Layer Protocol (Web / DNS / IRC Beacons)",
        "description": "Malware establishing periodic heartbeat connections with remote attacker C2 infrastructure.",
        "soc_action": "Sinkhole C2 domain, deploy SSL/TLS decryption inspection, inspect beacon regularity."
    },
    5: {
        "name": "Exfiltration",
        "tactic_id": "TA0010",
        "technique": "T1041: Exfiltration Over C2 Channel",
        "description": "Sustained high-volume outbound data transfer to external destination.",
        "soc_action": "Enforce outbound data throttling, immediately sever external route, invoke incident response."
    }
}


def label_flow_stage(flow_record: Dict[str, Any]) -> int:
    """Classifies an individual flow record into an ATT&CK stage (1..5) or 0 (Benign)."""
    # 1. CTU-13 / CIC Label string checks
    raw_label = str(flow_record.get("Label", flow_record.get("label", ""))).lower()
    
    if "cc" in raw_label or "c2" in raw_label or "botnet" in raw_label and "irc" in raw_label:
        return 4  # Command & Control
    if "spam" in raw_label or "exfil" in raw_label or "upload" in raw_label:
        return 5  # Exfiltration
    if "download" in raw_label or "exploit" in raw_label or "binary" in raw_label:
        return 2  # Initial Access
    if "scan" in raw_label or "probe" in raw_label or "attempt" in raw_label:
        return 1  # Reconnaissance

    # 2. Heuristic rule-based fallback
    dport = int(flow_record.get("dport", flow_record.get("Dst Port", 0)))
    tot_bytes = float(flow_record.get("tot_bytes", flow_record.get("TotLen Fwd Pkts", 0)))
    tot_pkts = float(flow_record.get("tot_pkts", flow_record.get("Tot Fwd Pkts", 0)))
    is_malicious = int(flow_record.get("is_malicious", 0))

    if not is_malicious and "botnet" not in raw_label and "attack" not in raw_label:
        return 0

    if dport in (445, 139, 3389):
        return 3  # Lateral Movement
    if tot_bytes > 200_000 and tot_pkts > 50:
        return 5  # Exfiltration
    if tot_pkts > 100 and tot_bytes / max(tot_pkts, 1) < 80:
        return 1  # Reconnaissance
    if dport in (80, 443, 8080, 6667):
        return 4  # C2
    return 1  # Default to Reconnaissance if malicious


def determine_dominant_stage(stage_counts: Dict[int, int]) -> int:
    """Returns the dominant malicious stage in a 60-second window, or 0 if completely benign."""
    malicious_counts = {s: count for s, count in stage_counts.items() if s > 0}
    if not malicious_counts:
        return 0
    return max(malicious_counts.items(), key=lambda kv: kv[1])[0]
