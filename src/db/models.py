"""SQLAlchemy ORM Models for Threatora State Store (Zero-Trust Architecture).

Stores:
  - Asset Inventory: Track known endpoints, subnet locations, and criticality.
  - Incident Logs: Record inference anomaly alerts, MITRE stages, and SHAP drivers.
  - Mitigation Playbooks: Actionable containment strategies and audit logs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()


def utc_now() -> datetime:
    """Returns current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Asset(Base):
    """Network Host / Endpoint Asset Inventory Record."""

    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ip_address = Column(String(45), unique=True, nullable=False, index=True)
    hostname = Column(String(255), nullable=False, default="unknown-host")
    mac_address = Column(String(17), nullable=True)
    subnet = Column(String(64), nullable=True, default="192.168.1.0/24")
    criticality = Column(String(32), nullable=False, default="MEDIUM")  # LOW, MEDIUM, HIGH, MISSION_CRITICAL
    status = Column(String(32), nullable=False, default="HEALTHY")     # HEALTHY, SUSPICIOUS, COMPROMISED, ISOLATED
    operating_system = Column(String(128), nullable=True, default="Linux / Ubuntu")
    services = Column(Text, nullable=True, default="[]")                # JSON string of listening ports/services
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    incidents = relationship("Incident", back_populates="asset", cascade="all, delete-orphan")
    playbooks = relationship("MitigationPlaybook", back_populates="asset", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        try:
            parsed_services = json.loads(self.services) if self.services else []
        except Exception:
            parsed_services = []

        return {
            "id": self.id,
            "ip_address": self.ip_address,
            "hostname": self.hostname,
            "mac_address": self.mac_address,
            "subnet": self.subnet,
            "criticality": self.criticality,
            "status": self.status,
            "operating_system": self.operating_system,
            "services": parsed_services,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Incident(Base):
    """Detected Security Incident from World Model & Anomaly Pipeline."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_uid = Column(String(64), unique=True, nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="SET NULL"), nullable=True)
    target_ip = Column(String(45), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    stage_id = Column(Integer, nullable=False, default=0)
    stage_name = Column(String(64), nullable=False, default="Benign")
    technique = Column(String(128), nullable=True)
    tactic_id = Column(String(32), nullable=True)
    forecast_timeline_json = Column(Text, nullable=True)  # JSON-serialized K-step rollout
    shap_drivers_json = Column(Text, nullable=True)       # JSON-serialized SHAP feature attributions
    is_contained = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)

    # Relationships
    asset = relationship("Asset", back_populates="incidents")
    playbooks = relationship("MitigationPlaybook", back_populates="incident", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_incident_target_created", "target_ip", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        try:
            forecast_data = json.loads(self.forecast_timeline_json) if self.forecast_timeline_json else []
        except Exception:
            forecast_data = []

        try:
            shap_data = json.loads(self.shap_drivers_json) if self.shap_drivers_json else {}
        except Exception:
            shap_data = {}

        return {
            "id": self.id,
            "incident_uid": self.incident_uid,
            "asset_id": self.asset_id,
            "target_ip": self.target_ip,
            "risk_score": self.risk_score,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "technique": self.technique,
            "tactic_id": self.tactic_id,
            "forecast_timeline": forecast_data,
            "shap_drivers": shap_data,
            "is_contained": self.is_contained,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MitigationPlaybook(Base):
    """Dynamic Playbook compiled by Mitigation Engine for an Incident."""

    __tablename__ = "mitigation_playbooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    playbook_uid = Column(String(64), unique=True, nullable=False, index=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="SET NULL"), nullable=True)
    target_ip = Column(String(45), nullable=False, index=True)
    kill_chain_stage = Column(String(64), nullable=False)
    damage_assessment = Column(Text, nullable=False)
    containment_strategy = Column(Text, nullable=False)
    containment_commands_json = Column(Text, nullable=False)  # JSON-serialized list of mitigation shell scripts
    status = Column(String(32), nullable=False, default="PENDING")  # PENDING, APPROVED, EXECUTED, FAILED
    execution_log = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    executed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    incident = relationship("Incident", back_populates="playbooks")
    asset = relationship("Asset", back_populates="playbooks")

    def to_dict(self) -> Dict[str, Any]:
        try:
            commands = json.loads(self.containment_commands_json) if self.containment_commands_json else []
        except Exception:
            commands = []

        return {
            "id": self.id,
            "playbook_uid": self.playbook_uid,
            "incident_id": self.incident_id,
            "asset_id": self.asset_id,
            "target_ip": self.target_ip,
            "kill_chain_stage": self.kill_chain_stage,
            "damage_assessment": self.damage_assessment,
            "containment_strategy": self.containment_strategy,
            "containment_commands": commands,
            "status": self.status,
            "execution_log": self.execution_log,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


class User(Base):
    """Authenticated Operator / Analyst Identity Record (Zero-Trust Access)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(128), nullable=False, default="Alex Kelly")
    role = Column(String(32), nullable=False, default="CHIEF_CISO_ADMIN")
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)

    def set_password(self, password: str) -> None:
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Validate password against hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }
