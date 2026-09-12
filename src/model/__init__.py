"""Model subpackage for Threatora Dual-Branch Attack Forecasting & World Models."""

from .world_model import NetworkWorldModel
from .flow_world_model import FlowLSTMWorldModel, FLOW_FEATURE_NAMES
from .packet_world_model import PacketLSTMWorldModel, PACKET_FEATURE_NAMES, PACKET_MITRE_STAGES
from .fusion import FusionLayer
from .decision import DecisionLayer, MITRE_ATTACK_TACTICS
from .baseline import LogisticRegressionBaseline, PersistenceBaseline

__all__ = [
    "NetworkWorldModel",
    "FlowLSTMWorldModel",
    "FLOW_FEATURE_NAMES",
    "PacketLSTMWorldModel",
    "PACKET_FEATURE_NAMES",
    "PACKET_MITRE_STAGES",
    "FusionLayer",
    "DecisionLayer",
    "MITRE_ATTACK_TACTICS",
    "LogisticRegressionBaseline",
    "PersistenceBaseline",
]
