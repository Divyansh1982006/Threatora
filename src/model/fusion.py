"""Multi-Modal Fusion Layer for Threatora Network World Models.

Combines Flow-level predictions (P_flow) and Packet-level predictions (P_packet)
into a unified Attack Probability P_attack(t) with cross-modal Agreement scoring.
Supports single-modality (CSV-only or PCAP-only) and dual-modality inputs.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Union
import numpy as np


class FusionLayer:
    """Multi-Modal Fusion Engine combining Flow and Packet State-Space Estimations."""

    def __init__(
        self,
        flow_weight: float = 0.5,
        conflict_threshold: float = 0.40,
    ):
        """
        Args:
            flow_weight: Prior weight alpha for flow model (packet weight is 1 - alpha).
            conflict_threshold: Absolute discrepancy delta triggering a cross-modal conflict flag.
        """
        self.flow_weight = flow_weight
        self.packet_weight = 1.0 - flow_weight
        self.conflict_threshold = conflict_threshold

    def fuse(
        self,
        p_flow: Optional[Union[float, np.ndarray, List[float]]] = None,
        p_packet: Optional[Union[float, np.ndarray, List[float]]] = None,
        flow_confidence: float = 1.0,
        packet_confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Fuses flow and packet probability predictions.

        Args:
            p_flow: Attack probability from Flow LSTM World Model [0, 1]
            p_packet: Attack probability from Packet LSTM World Model [0, 1]
            flow_confidence: Optional confidence scalar for flow stream
            packet_confidence: Optional confidence scalar for packet stream

        Returns:
            Dictionary containing:
                - p_attack: Final fused score P_attack(t)
                - p_flow: Raw flow probability
                - p_packet: Raw packet probability
                - agreement_score: Concordance metric in [0, 1]
                - is_conflict: True if cross-modal disagreement exceeds threshold
                - active_modalities: List of active input modalities ('flow', 'packet')
                - fusion_mode: 'flow_only', 'packet_only', or 'dual_modality'
        """
        has_flow = p_flow is not None
        has_packet = p_packet is not None

        if not has_flow and not has_packet:
            return {
                "p_attack": 0.0,
                "p_flow": None,
                "p_packet": None,
                "agreement_score": 1.0,
                "is_conflict": False,
                "active_modalities": [],
                "fusion_mode": "none",
            }

        # Single Modality: Flow Only
        if has_flow and not has_packet:
            score = float(np.mean(p_flow)) if isinstance(p_flow, (list, np.ndarray)) else float(p_flow)
            return {
                "p_attack": round(score, 4),
                "p_flow": round(score, 4),
                "p_packet": None,
                "agreement_score": 1.0,
                "is_conflict": False,
                "active_modalities": ["flow"],
                "fusion_mode": "flow_only",
            }

        # Single Modality: Packet Only
        if has_packet and not has_flow:
            score = float(np.mean(p_packet)) if isinstance(p_packet, (list, np.ndarray)) else float(p_packet)
            return {
                "p_attack": round(score, 4),
                "p_flow": None,
                "p_packet": round(score, 4),
                "agreement_score": 1.0,
                "is_conflict": False,
                "active_modalities": ["packet"],
                "fusion_mode": "packet_only",
            }

        # Dual Modality: Both Flow and Packet present
        pf = float(np.mean(p_flow)) if isinstance(p_flow, (list, np.ndarray)) else float(p_flow)
        pp = float(np.mean(p_packet)) if isinstance(p_packet, (list, np.ndarray)) else float(p_packet)

        # Weighted fusion with confidence modulation
        w_flow = self.flow_weight * flow_confidence
        w_packet = self.packet_weight * packet_confidence
        w_total = w_flow + w_packet

        if w_total > 0:
            norm_w_flow = w_flow / w_total
            norm_w_packet = w_packet / w_total
        else:
            norm_w_flow, norm_w_packet = 0.5, 0.5

        fused_score = norm_w_flow * pf + norm_w_packet * pp

        discrepancy = abs(pf - pp)
        agreement = max(0.0, 1.0 - discrepancy)
        is_conflict = discrepancy > self.conflict_threshold

        return {
            "p_attack": round(float(fused_score), 4),
            "p_flow": round(pf, 4),
            "p_packet": round(pp, 4),
            "agreement_score": round(float(agreement), 4),
            "is_conflict": is_conflict,
            "discrepancy": round(float(discrepancy), 4),
            "active_modalities": ["flow", "packet"],
            "fusion_mode": "dual_modality",
        }
