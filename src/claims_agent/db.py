"""SQLite access helpers for the synthetic claims database."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id            TEXT PRIMARY KEY,
    patient_name        TEXT NOT NULL,
    dob                 TEXT NOT NULL,
    member_id           TEXT NOT NULL,
    ssn                 TEXT NOT NULL,
    phone               TEXT NOT NULL,
    payer               TEXT NOT NULL,
    rendering_npi       TEXT NOT NULL,
    cpt_code            TEXT NOT NULL,
    modifier            TEXT NOT NULL,
    icd10_code          TEXT NOT NULL,
    billed_amount       REAL NOT NULL,
    service_date        TEXT NOT NULL,
    submitted_date      TEXT NOT NULL,
    denial_code         TEXT NOT NULL,
    denial_date         TEXT NOT NULL,
    has_clinical_notes  INTEGER NOT NULL,
    timely_filing_proof INTEGER NOT NULL,
    notes               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prior_auths (
    auth_id        TEXT PRIMARY KEY,
    claim_id       TEXT NOT NULL REFERENCES claims(claim_id),
    cpt_code       TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('approved','pending','denied')),
    auth_number    TEXT,
    decision_date  TEXT
);
CREATE TABLE IF NOT EXISTS denial_policies (
    denial_code             TEXT NOT NULL,
    payer                   TEXT NOT NULL,      -- '*' = default for all payers
    category                TEXT NOT NULL,
    description             TEXT NOT NULL,
    guidance                TEXT NOT NULL,
    appeal_window_days      INTEGER NOT NULL,
    retro_auth_window_days  INTEGER NOT NULL,
    policy_doc_ids          TEXT NOT NULL,      -- comma-separated
    PRIMARY KEY (denial_code, payer)
);
CREATE INDEX IF NOT EXISTS idx_prior_auths_claim ON prior_auths(claim_id);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a read-friendly connection with dict-like rows."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_database(db_path: Path | str) -> Path:
    """Create and seed the database if it does not exist yet."""
    path = Path(db_path)
    if not path.exists():
        from claims_agent.data.seed import build_database

        logger.info("Database %s not found; seeding synthetic data", path)
        build_database(path)
    return path
