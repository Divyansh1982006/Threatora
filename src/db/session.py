"""Database Session Management and Initial Seed Routine.

Zero-Trust Principles:
  - Strict connection pooling with pre-ping validation.
  - Fail-safe fallback to local SQLite persistence when PostgreSQL environment is absent.
  - Deterministic auto-seeding of baseline critical infrastructure assets.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, Asset, User
from ..config import ROOT_DIR, ARTIFACTS_DIR

# Determine database URL from environment or fallback to SQLite
DB_DIR = ARTIFACTS_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SQLITE_URL = f"sqlite:///{DB_DIR / 'threatora.db'}"

raw_db_url = os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)
# Render compatibility for postgres:// scheme
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

# SQLite concurrency configuration vs PostgreSQL connection pooling
connect_args = {"check_same_thread": False} if raw_db_url.startswith("sqlite") else {}

engine = create_engine(
    raw_db_url,
    echo=False,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Creates database schema if not present and seeds initial asset inventory and default users."""
    Base.metadata.create_all(bind=engine)
    seed_asset_inventory()
    seed_default_users()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """Dependency generator for Flask / FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_asset_inventory() -> None:
    """Pre-populates baseline asset inventory if empty."""
    with get_db_context() as db:
        if db.query(Asset).first() is not None:
            return  # Already populated

        initial_assets = [
            Asset(
                ip_address="192.168.1.5",
                hostname="SRV-DATABASE-01",
                mac_address="00:50:56:A1:22:B4",
                subnet="192.168.1.0/24",
                criticality="MISSION_CRITICAL",
                status="HEALTHY",
                operating_system="Ubuntu 22.04 LTS",
                services=json.dumps(["postgresql:5432", "ssh:22", "node_exporter:9100"]),
            ),
            Asset(
                ip_address="192.168.1.10",
                hostname="DC-CORP-AD01",
                mac_address="00:50:56:A1:33:C5",
                subnet="192.168.1.0/24",
                criticality="MISSION_CRITICAL",
                status="HEALTHY",
                operating_system="Windows Server 2022",
                services=json.dumps(["kerberos:88", "ldap:389", "smb:445", "dns:53"]),
            ),
            Asset(
                ip_address="192.168.1.15",
                hostname="INGRESS-NGINX-PROXY",
                mac_address="00:50:56:B2:44:D6",
                subnet="192.168.1.0/24",
                criticality="HIGH",
                status="HEALTHY",
                operating_system="Alpine Linux",
                services=json.dumps(["http:80", "https:443", "ssh:2222"]),
            ),
            Asset(
                ip_address="192.168.1.105",
                hostname="DEV-WORKSTATION-05",
                mac_address="3C:52:82:11:FE:09",
                subnet="192.168.1.0/24",
                criticality="MEDIUM",
                status="HEALTHY",
                operating_system="Debian 12 Bookworm",
                services=json.dumps(["ssh:22", "docker:2375"]),
            ),
            Asset(
                ip_address="147.32.84.165",
                hostname="EDGE-GATEWAY-EXT",
                mac_address="52:54:00:12:34:56",
                subnet="147.32.84.0/24",
                criticality="HIGH",
                status="HEALTHY",
                operating_system="VyOS Router",
                services=json.dumps(["bgp:179", "ipsec:500", "snmp:161"]),
            ),
        ]
        db.add_all(initial_assets)
        print(f"[+] Successfully seeded {len(initial_assets)} baseline assets into inventory.")


def seed_default_users() -> None:
    """Pre-populates default operator account if empty."""
    with get_db_context() as db:
        if db.query(User).first() is not None:
            return  # Already seeded

        admin_user = User(
            username=os.environ.get("THREATORA_ADMIN_USER", "admin"),
            email="ciso-soc@threatora.internal",
            full_name="Alex Kelly",
            role="CHIEF_CISO_ADMIN",
            is_active=True,
        )
        admin_user.set_password(os.environ.get("THREATORA_ADMIN_PASSWORD", "Threatora@2026"))
        db.add(admin_user)
        print("[+] Seeded default administrative operator into user identity store: admin / Threatora@2026")
