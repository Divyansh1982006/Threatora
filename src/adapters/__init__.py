"""Dynamic Canonical Feature Adapter Layer for Heterogeneous Network Flow Telemetry.

SIH Problem Statement 26153 (AI-based Network Attack Forecasting).
Supports:
  - UNSW-NB15
  - CSE-CIC-IDS2018
  - CTU-13
Maps heterogeneous schemas into 12 canonical continuous slots with zero data leakage.
"""

from .dataset_adapter import (
    CANONICAL_SLOTS,
    CanonicalFeatureExtractor,
    load_schema_registry,
    parse_port_value,
)

__all__ = [
    "CANONICAL_SLOTS",
    "CanonicalFeatureExtractor",
    "load_schema_registry",
    "parse_port_value",
]
