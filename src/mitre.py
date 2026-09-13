"""MITRE ATT&CK stage vocabulary, taxonomy, and rules for Threatora.

Maps network anomalies and flow behaviors into the 5 core attack phases
defined by the NTRO PS 26153 specification:
  0. Benign
  1. Reconnaissance
  2. Initial Access
  3. Lateral Movement
  4. Command & Control
  5. Exfiltration

Stage supervision comes from two sources, in order of authority:
  1. `stage_from_label` - the CTU-13 dataset's own annotation. A flow marked CC16
     is ground-truth command-and-control.
  2. `derive_stage` - behavioral rules used where labels are silent or traffic arrives unannotated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Optional

from .config import C2_PORTS, INTERNAL_PREFIXES, LATERAL_PORTS


# --------------------------------------------------------------------------
# Stage vocabulary
# --------------------------------------------------------------------------

BENIGN = 0
RECONNAISSANCE = 1
INITIAL_ACCESS = 2
LATERAL_MOVEMENT = 3
COMMAND_AND_CONTROL = 4
EXFILTRATION = 5

class StageNames(tuple):
    """Tuple subclass that also supports dict-like .get(key, default) access."""
    def get(self, key: Any, default: str = "Benign") -> str:
        try:
            k = int(key)
            if 0 <= k < len(self):
                return self[k]
            return default
        except (ValueError, TypeError):
            return default


STAGE_NAMES: StageNames = StageNames((
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
))

N_STAGES = len(STAGE_NAMES)

# Kill-chain ordering used by dashboard to decide progression vs persistence
STAGE_ORDER = {
    BENIGN: -1,
    RECONNAISSANCE: 0,
    INITIAL_ACCESS: 1,
    LATERAL_MOVEMENT: 2,
    COMMAND_AND_CONTROL: 3,
    EXFILTRATION: 4,
}

STAGE_COLORS = {
    0: "#10b981",  # Emerald green
    1: "#38bdf8",  # Sky blue
    2: "#f59e0b",  # Amber
    3: "#8b5cf6",  # Purple
    4: "#ec4899",  # Pink
    5: "#ef4444",  # Red
}


@dataclass(frozen=True)
class TechniqueRef:
    """A MITRE ATT&CK tactic/technique pair shown alongside a predicted stage."""

    tactic_id: str
    tactic: str
    technique_id: str
    technique: str


STAGE_TECHNIQUES: dict[int, TechniqueRef] = {
    BENIGN: TechniqueRef("-", "No adversary activity", "-", "-"),
    RECONNAISSANCE: TechniqueRef(
        "TA0043", "Reconnaissance", "T1046", "Network Service Discovery"
    ),
    INITIAL_ACCESS: TechniqueRef(
        "TA0001", "Initial Access", "T1190", "Exploit Public-Facing Application"
    ),
    LATERAL_MOVEMENT: TechniqueRef(
        "TA0008", "Lateral Movement", "T1021", "Remote Services"
    ),
    COMMAND_AND_CONTROL: TechniqueRef(
        "TA0011", "Command and Control", "T1071", "Application Layer Protocol"
    ),
    EXFILTRATION: TechniqueRef(
        "TA0010", "Exfiltration", "T1041", "Exfiltration Over C2 Channel"
    ),
}

STAGE_METADATA: dict[int, dict[str, str]] = {
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


def is_internal(addr: str) -> bool:
    """True when an address sits inside monitored enterprise prefixes."""
    return any(addr.startswith(p) for p in INTERNAL_PREFIXES)


# --------------------------------------------------------------------------
# Stage from dataset flow labels
# --------------------------------------------------------------------------

_LABEL_STAGE_RULES: tuple[tuple[str, int], ...] = (
    # Explicit command-and-control channels, numbered by capture authors
    (r"-cc\d+", COMMAND_AND_CONTROL),
    (r"irc", COMMAND_AND_CONTROL),
    (r"custom-encryption", COMMAND_AND_CONTROL),

    # Outbound spam / mail relay
    (r"spam", EXFILTRATION),
    (r"smtp", EXFILTRATION),

    # Payload retrieval
    (r"binary-download", INITIAL_ACCESS),

    # Click fraud / tasking over application layer
    (r"http-ad", COMMAND_AND_CONTROL),
    (r"web-established", COMMAND_AND_CONTROL),

    # Unanswered probes and discovery
    (r"attempt", RECONNAISSANCE),
    (r"dns", RECONNAISSANCE),
    (r"icmp", RECONNAISSANCE),

    # Generic established botnet traffic
    (r"established", INITIAL_ACCESS),
)

_LABEL_STAGE_COMPILED = tuple(
    (re.compile(pattern), stage) for pattern, stage in _LABEL_STAGE_RULES
)


def stage_from_label(label: str) -> int | None:
    """Maps a CTU-13 flow label onto an ATT&CK stage."""
    if not label:
        return None
    text = str(label).lower()
    if "botnet" not in text:
        return None
    for pattern, stage in _LABEL_STAGE_COMPILED:
        if pattern.search(text):
            return stage
    return None


def label_stage_patterns() -> list[dict]:
    """Returns mapping table for frontend and architecture inspection."""
    return [
        {"pattern": p, "stage": STAGE_NAMES[s], "stage_id": s}
        for p, s in _LABEL_STAGE_RULES
    ]


# --------------------------------------------------------------------------
# Behavioral Stage Derivation Fallback
# --------------------------------------------------------------------------

def derive_stage(
    *,
    n_flows: int = 0,
    n_unique_dst_ips: int = 0,
    n_unique_dst_ports: int = 0,
    mean_duration: float = 0.0,
    src_bytes: float = 0.0,
    total_bytes: float = 0.0,
    frac_internal_dst: float = 0.0,
    frac_lateral_ports: float = 0.0,
    frac_c2_ports: float = 0.0,
    frac_syn_only: float = 0.0,
    beacon_regularity: float = 0.0,
    **kwargs: Any
) -> int:
    """Maps one host-window of malicious flow behaviour onto a kill-chain stage."""
    fan_out = max(n_unique_dst_ips, n_unique_dst_ports)
    egress_ratio = src_bytes / total_bytes if total_bytes > 0 else 0.0

    # 1. Reconnaissance: fan-out of short probes
    if fan_out >= 15 and mean_duration < 2.0 and (frac_syn_only > 0.5 or n_flows >= 25):
        return RECONNAISSANCE

    # 2. Exfiltration: large outbound transfer leaving enterprise
    if (
        egress_ratio > 0.75
        and src_bytes > 50_000
        and frac_internal_dst < 0.5
        and n_unique_dst_ips <= 5
    ):
        return EXFILTRATION

    # 3. Lateral movement: internal traffic on admin ports
    if frac_internal_dst > 0.6 and frac_lateral_ports > 0.3:
        return LATERAL_MOVEMENT

    # 4. Command & Control: beaconing regularity or c2 port contact
    if (
        beacon_regularity > 0.6
        and n_unique_dst_ips <= 3
        and n_flows >= 4
        and total_bytes < 200_000
    ) or (frac_c2_ports > 0.5 and n_unique_dst_ips <= 3 and beacon_regularity > 0.4):
        return COMMAND_AND_CONTROL

    # 5. Initial Access fallback
    return INITIAL_ACCESS


def label_flow_stage(flow_record: Dict[str, Any]) -> int:
    """Compatibility wrapper: classifies individual flow record into stage (0..5)."""
    raw_label = str(flow_record.get("Label", flow_record.get("label", "")))
    lbl_stage = stage_from_label(raw_label)
    if lbl_stage is not None:
        return lbl_stage

    is_malicious = int(flow_record.get("is_malicious", 0))
    if not is_malicious and "botnet" not in raw_label.lower() and "attack" not in raw_label.lower():
        return BENIGN

    dport = int(flow_record.get("dport", flow_record.get("Dst Port", 0)))
    tot_bytes = float(flow_record.get("tot_bytes", flow_record.get("TotLen Fwd Pkts", 0)))
    tot_pkts = float(flow_record.get("tot_pkts", flow_record.get("Tot Fwd Pkts", 0)))

    if dport in LATERAL_PORTS or dport in (445, 139, 3389):
        return LATERAL_MOVEMENT
    if tot_bytes > 200_000 and tot_pkts > 50:
        return EXFILTRATION
    if dport in C2_PORTS or dport in (80, 443, 8080, 6667):
        return COMMAND_AND_CONTROL
    if tot_pkts > 100 and tot_bytes / max(tot_pkts, 1) < 80:
        return RECONNAISSANCE
    return INITIAL_ACCESS


def determine_dominant_stage(stage_counts: Dict[int, int]) -> int:
    """Returns dominant malicious stage in a window, or 0 if benign."""
    malicious_counts = {s: count for s, count in stage_counts.items() if s > 0}
    if not malicious_counts:
        return BENIGN
    return max(malicious_counts.items(), key=lambda kv: kv[1])[0]


def stage_name(stage: int) -> str:
    if 0 <= stage < len(STAGE_NAMES):
        return STAGE_NAMES[stage]
    return "Unknown"


def is_progression(from_stage: int, to_stage: int) -> bool:
    """True when `to_stage` sits later in the kill chain than `from_stage`."""
    return STAGE_ORDER.get(to_stage, -1) > STAGE_ORDER.get(from_stage, -1)


def describe(stage: int) -> dict:
    """Serialisable stage description for API layer."""
    ref = STAGE_TECHNIQUES.get(stage, STAGE_TECHNIQUES[BENIGN])
    return {
        "id": stage,
        "name": stage_name(stage),
        "tactic_id": ref.tactic_id,
        "tactic": ref.tactic,
        "technique_id": ref.technique_id,
        "technique": ref.technique,
        "order": STAGE_ORDER.get(stage, -1),
    }


def all_stages() -> Sequence[dict]:
    return [describe(s) for s in range(N_STAGES)]
