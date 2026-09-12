"""Decision Layer and MITRE ATT&CK Mapping for Threatora.

Translates fused attack probabilities and feature signals into binary decisions
(NORMAL vs. ATTACK) and granular MITRE ATT&CK tactical mappings with SOC recommendations.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np


MITRE_ATTACK_TACTICS: Dict[str, Dict[str, Any]] = {
    "Reconnaissance": {
        "tactic_id": "TA0043",
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery / Port Scan",
        "stage_id": 1,
        "color": "#38bdf8",
        "severity": "MEDIUM",
        "description": "Adversary probing IP addresses and destination ports to map active services.",
        "soc_action": "Rate-limit SYN probing on perimeter firewall, isolate source IP, and inspect scan distribution.",
    },
    "Initial Access": {
        "tactic_id": "TA0001",
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "stage_id": 2,
        "color": "#f59e0b",
        "severity": "HIGH",
        "description": "Adversary executing exploit payloads or transmitting malicious binary content.",
        "soc_action": "Terminate active connection session, inspect endpoint process trees, and trigger EDR host isolation.",
    },
    "Lateral Movement": {
        "tactic_id": "TA0008",
        "technique_id": "T1021.002",
        "technique_name": "SMB / Windows Admin Shares",
        "stage_id": 3,
        "color": "#8b5cf6",
        "severity": "HIGH",
        "description": "Compromised host scanning and attempting unauthorized lateral access via SMB/RDP.",
        "soc_action": "Segment VLAN boundary, block ports 445/3389 laterally, and rotate compromised domain credentials.",
    },
    "Command & Control": {
        "tactic_id": "TA0011",
        "technique_id": "T1071",
        "technique_name": "Application Layer Protocol (Web / DNS / IRC Beacons)",
        "stage_id": 4,
        "color": "#ec4899",
        "severity": "CRITICAL",
        "description": "Malware maintaining periodic heartbeat C2 communications with external attacker infrastructure.",
        "soc_action": "Sinkhole C2 domain, inspect beacon inter-arrival regularity, and trigger deep packet TLS inspection.",
    },
    "Exfiltration": {
        "tactic_id": "TA0010",
        "technique_id": "T1041",
        "technique_name": "Exfiltration Over C2 Channel",
        "stage_id": 5,
        "color": "#ef4444",
        "severity": "CRITICAL",
        "description": "Sustained high-volume outbound data transfer to unauthorized external destinations.",
        "soc_action": "Enforce immediate outbound bandwidth throttling, sever external routing path, and initiate DFIR triage.",
    },
    "Impact / DoS": {
        "tactic_id": "TA0040",
        "technique_id": "T1498",
        "technique_name": "Network Denial of Service (SYN / UDP / ICMP Flood)",
        "stage_id": 6,
        "color": "#f97316",
        "severity": "CRITICAL",
        "description": "High-rate resource exhaustion flooding designed to degrade system availability.",
        "soc_action": "Activate upstream DDoS mitigation scrubbing center, deploy SYN cookies, and rate-limit ingress.",
    },
}


class DecisionLayer:
    """Decision Layer for Network Security Classification & Threat Mapping."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def decide(
        self,
        p_attack: float,
        feature_vector: Optional[np.ndarray] = None,
        context_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluates fused attack probability to produce final operational decision.

        Args:
            p_attack: Fused probability P_attack(t) in [0, 1]
            feature_vector: Optional 12-dim flow feature vector for signature inference
            context_meta: Optional metadata dict (e.g. raw labels, IPs, ports)

        Returns:
            Dictionary with classification (NORMAL vs ATTACK), MITRE ATT&CK mapping,
            confidence score, and recommended SOC actions.
        """
        is_attack = float(p_attack) >= self.threshold
        confidence = float(p_attack) if is_attack else (1.0 - float(p_attack))

        if not is_attack:
            return {
                "decision": "NORMAL",
                "is_attack": False,
                "confidence": round(confidence, 4),
                "risk_score": round(float(p_attack), 4),
                "tactic": None,
                "technique": None,
                "stage": {
                    "id": 0,
                    "name": "Benign",
                    "color": "#10b981",
                    "severity": "LOW",
                    "description": "Traffic conforms to normal network baseline distributions.",
                    "soc_action": "Routine continuous monitoring.",
                },
            }

        # Attack classification & MITRE ATT&CK stage mapping
        tactic_key = self._infer_tactic(feature_vector, context_meta)
        tactic_info = MITRE_ATTACK_TACTICS[tactic_key]

        return {
            "decision": "ATTACK",
            "is_attack": True,
            "confidence": round(confidence, 4),
            "risk_score": round(float(p_attack), 4),
            "tactic": tactic_key,
            "technique": f"{tactic_info['technique_id']}: {tactic_info['technique_name']}",
            "stage": {
                "id": tactic_info["stage_id"],
                "name": tactic_key,
                "color": tactic_info["color"],
                "severity": tactic_info["severity"],
                "tactic_id": tactic_info["tactic_id"],
                "technique_id": tactic_info["technique_id"],
                "description": tactic_info["description"],
                "soc_action": tactic_info["soc_action"],
            },
        }

    def _infer_tactic(
        self,
        features: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Infers MITRE ATT&CK tactic from feature heuristics or metadata."""
        if meta and "label" in meta:
            lbl = str(meta["label"]).lower()
            if "ddos" in lbl or "dos" in lbl or "flood" in lbl:
                return "Impact / DoS"
            if "recon" in lbl or "scan" in lbl or "port" in lbl:
                return "Reconnaissance"
            if "c2" in lbl or "cc" in lbl or "botnet" in lbl or "irc" in lbl:
                return "Command & Control"
            if "exfil" in lbl or "upload" in lbl:
                return "Exfiltration"
            if "exploit" in lbl or "infect" in lbl or "binary" in lbl:
                return "Initial Access"

        if features is not None and len(features) >= 12:
            # Features: [flow_duration(0), Duration(1), Rate(2), Srate(3), Drate(4),
            #            fin(5), syn(6), rst(7), ack(8), Tot_size(9), IAT(10), Number(11)]
            rate = float(features[2])
            syn_flag = float(features[6])
            rst_flag = float(features[7])
            tot_size = float(features[9])
            iat = float(features[10])

            # DoS / Flooding signature (high packet rate / flood)
            if rate > 100.0 or syn_flag > 0.8:
                return "Impact / DoS"
            # Reconnaissance (SYN/RST probe with small payload size)
            if (syn_flag > 0.0 or rst_flag > 0.0) and tot_size < 100.0:
                return "Reconnaissance"
            # High volume exfiltration
            if tot_size > 50000.0:
                return "Exfiltration"
            # Periodic C2 beaconing
            if iat > 1e7:
                return "Command & Control"

        return "Command & Control"
