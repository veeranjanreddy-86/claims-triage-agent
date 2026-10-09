"""Deterministic generator for the SYNTHETIC claims database.

Every value here is invented. Patient identifiers are deliberately fake:
SSNs use the never-issued 9xx area range, phone numbers use the 555-01xx
fictional range, and payer names are fictional.

Usage:  python -m claims_agent.data.seed --db var/claims.db
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from claims_agent.db import SCHEMA, connect

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
PAYERS = ["Bluefield Mutual", "Cedar Valley Care", "Northwind Health Plan", "Summit Coast Health"]
FIRST = ["Avery", "Jordan", "Riley", "Morgan", "Quinn", "Harper", "Rowan", "Emerson", "Sage"]
LAST = ["Lindqvist", "Okafor", "Marchetti", "Tanaka", "Castellano", "Brennan", "Iyer", "Novak"]

# fmt: off
# (code, payer, category, description, guidance, appeal_days, retro_days, doc_ids)
DENIAL_POLICIES: list[tuple[str, str, str, str, str, int, int, str]] = [
    ("CO-16", "*", "coding", "Claim is missing information or contains a billing error.",
     "Fix the missing or invalid data element and send a corrected claim.", 90, 0, "POL-COD-001"),
    ("CO-4", "*", "coding", "Procedure code and modifier do not agree, or a modifier is missing.",
     "Add or correct the modifier and send a corrected claim.", 90, 0, "POL-COD-001,POL-COD-002"),
    ("CO-11", "*", "coding", "Diagnosis does not support the billed procedure.",
     "Review documentation, correct diagnosis pointer/code, send a corrected claim.", 90, 0,
     "POL-COD-001,POL-COD-003"),
    ("CO-197", "*", "authorization", "Required prior authorization was not on the claim.",
     "Verify whether an authorization exists; attach it or request retro-authorization.", 180, 30,
     "POL-AUTH-001"),
    ("CO-197", "Cedar Valley Care", "authorization",
     "Required prior authorization was not on the claim.",
     "Cedar Valley Care allows retro-authorization only within 14 days of service.", 180, 14,
     "POL-AUTH-001,POL-CV-AUTH-002"),
    ("CO-50", "*", "medical_necessity", "Payer judged the service not medically necessary.",
     "Appeal with clinical documentation if it supports necessity.", 180, 0, "POL-MN-001"),
    ("CO-29", "*", "timely_filing", "Claim was received after the filing limit.",
     "Appeal only with proof of timely original submission.", 90, 0, "POL-TF-001"),
    ("CO-29", "Northwind Health Plan", "timely_filing",
     "Claim was received after the filing limit.",
     "Appeal only with proof of timely original submission.", 60, 0, "POL-TF-001,POL-NW-TF-002"),
    ("CO-18", "*", "duplicate", "Claim duplicates a previously adjudicated claim.",
     "Confirm the original was paid; close the duplicate balance.", 90, 0, "POL-DUP-001"),
]
# fmt: on


@dataclass(frozen=True)
class AnchorClaim:
    claim_id: str
    payer: str
    cpt: str
    modifier: str
    icd10: str
    service: str
    submitted: str
    denial_code: str
    denial: str
    clinical_notes: bool
    tf_proof: bool
    npi: str
    notes: str
    auth: tuple[str, str | None, str | None] | None = None  # (status, auth_number, decision_date)


# fmt: off
# Hand-designed claims that the evaluation scenarios refer to.
ANCHORS: list[AnchorClaim] = [
    AnchorClaim("CLM-1001", PAYERS[0], "99214", "", "E11.9", "2026-05-04", "2026-05-10", "CO-16",
                "2026-05-25", False, False, "",
                "Rendering NPI missing on line 1. Left voicemail for patient at {phone}."),
    AnchorClaim("CLM-1002", PAYERS[1], "20610", "", "M17.11", "2026-05-12", "2026-05-15", "CO-4",
                "2026-06-02", True, False, "1932456781",
                "Laterality modifier not reported for large-joint injection."),
    AnchorClaim("CLM-1003", PAYERS[3], "97110", "GP", "Z00.00", "2026-04-22", "2026-04-28",
                "CO-11", "2026-05-19", True, False, "1588123409",
                "Primary diagnosis is a wellness code; therapy note documents low back pain."),
    AnchorClaim("CLM-1004", PAYERS[0], "72148", "", "M54.16", "2026-05-20", "2026-05-22",
                "CO-197", "2026-06-10", True, False, "1477201938",
                "Front desk states auth was obtained before the MRI.",
                ("approved", "AUTH-BM-55021", "2026-05-14")),
    AnchorClaim("CLM-1005", PAYERS[2], "70553", "", "G43.909", "2026-06-12", "2026-06-15",
                "CO-197", "2026-06-24", True, False, "1366098124",
                "Urgent brain MRI scheduled same day; no auth request found."),
    AnchorClaim("CLM-1006", PAYERS[1], "29881", "RT", "S83.241A", "2026-03-02", "2026-03-06",
                "CO-197", "2026-03-30", True, False, "1255876301",
                "Arthroscopy performed; authorization never requested."),
    AnchorClaim("CLM-1007", PAYERS[3], "64483", "", "M54.17", "2026-04-30", "2026-05-02",
                "CO-50", "2026-05-28", True, False, "1144765290",
                "Epidural injection after 8 weeks of failed conservative therapy."),
    AnchorClaim("CLM-1008", PAYERS[0], "97140", "GP", "M62.81", "2026-05-06", "2026-05-09",
                "CO-50", "2026-06-03", False, False, "1033654189",
                "Manual therapy; no progress notes on file."),
    AnchorClaim("CLM-1009", PAYERS[0], "99213", "", "J06.9", "2025-12-10", "2026-01-05",
                "CO-29", "2026-06-01", True, True, "1922543078",
                "Clearinghouse acceptance report for original submission is on file."),
    AnchorClaim("CLM-1010", PAYERS[3], "99204", "", "R10.9", "2025-08-14", "2026-04-20",
                "CO-29", "2026-05-12", True, False, "1811432967",
                "Claim held in billing queue; first submission was April 2026."),
    AnchorClaim("CLM-1011", PAYERS[1], "99213", "", "I10", "2026-05-11", "2026-05-27",
                "CO-18", "2026-06-08", True, False, "1700321856",
                "Second submission of a claim already paid on 2026-05-30."),
    AnchorClaim("CLM-1012", PAYERS[2], "97162", "GP", "M25.561", "2025-10-20", "2025-10-24",
                "CO-50", "2025-11-15", True, False, "1699210745",
                "PT evaluation; denial letter misplaced and found during cleanup."),
    AnchorClaim("CLM-1013", PAYERS[3], "73721", "", "M23.205", "2026-06-05", "2026-06-08",
                "CO-197", "2026-06-20", True, False, "1588109634",
                "Auth request submitted, awaiting payer decision.",
                ("pending", None, None)),
    AnchorClaim("CLM-1014", PAYERS[2], "63030", "", "M51.26", "2026-04-15", "2026-04-18",
                "CO-197", "2026-05-05", True, False, "1477098523",
                "Payer denied auth; surgeon has imaging and conservative-care records.",
                ("denied", "AUTH-NW-77310", "2026-04-10")),
    AnchorClaim("CLM-1015", PAYERS[2], "99215", "", "E78.5", "2026-05-26", "2026-05-28",
                "CO-16", "2026-06-11", False, False, "1366987412",
                "Subscriber ID format rejected by payer; member ID was keyed as {member_id}."),
    AnchorClaim("CLM-1016", PAYERS[2], "99214", "25", "J45.40", "2025-11-03", "2025-12-01",
                "CO-29", "2026-06-15", True, True, "1255876309",
                "Payer acknowledgment (277CA) for original submission is on file."),
    AnchorClaim("CLM-1017", PAYERS[1], "97530", "GP", "M62.81", "2026-05-01", "2026-05-04",
                "CO-50", "2026-06-05", False, False, "1144765298",
                "Therapeutic activities; documentation status not updated in billing system."),
    AnchorClaim("CLM-1018", PAYERS[0], "99203", "", "Z23", "2026-05-18", "2026-05-20", "CO-11",
                "2026-06-09", True, False, "1033654187",
                "New-patient visit billed with immunization diagnosis only."),
]
# fmt: on


def _fake_identity(rng: random.Random) -> dict[str, str]:
    dob = date(1940, 1, 1) + timedelta(days=rng.randint(0, 365 * 65))
    return {
        "patient_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        "dob": dob.isoformat(),
        "member_id": f"MBR{rng.randint(10_000_000, 99_999_999)}",
        "ssn": f"9{rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}",
        "phone": f"555-01{rng.randint(10, 99)}",
    }


def _insert_claim(conn, row: dict) -> None:  # type: ignore[no-untyped-def]
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO claims ({cols}) VALUES ({marks})", list(row.values()))


def build_database(db_path: Path | str, seed: int = DEFAULT_SEED, n_filler: int = 180) -> Path:
    """(Re)build the synthetic database. Same seed -> byte-identical content."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    conn = connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO denial_policies VALUES (?, ?, ?, ?, ?, ?, ?, ?)", DENIAL_POLICIES
        )
        for a in ANCHORS:
            ident = _fake_identity(rng)
            _insert_claim(
                conn,
                {
                    "claim_id": a.claim_id,
                    **ident,
                    "payer": a.payer,
                    "rendering_npi": a.npi,
                    "cpt_code": a.cpt,
                    "modifier": a.modifier,
                    "icd10_code": a.icd10,
                    "billed_amount": round(rng.uniform(120, 4800), 2),
                    "service_date": a.service,
                    "submitted_date": a.submitted,
                    "denial_code": a.denial_code,
                    "denial_date": a.denial,
                    "has_clinical_notes": int(a.clinical_notes),
                    "timely_filing_proof": int(a.tf_proof),
                    "notes": a.notes.format(phone=ident["phone"], member_id=ident["member_id"]),
                },
            )
            if a.auth:
                status, number, decided = a.auth
                conn.execute(
                    "INSERT INTO prior_auths VALUES (?, ?, ?, ?, ?, ?)",
                    (f"PA-{a.claim_id[4:]}", a.claim_id, a.cpt, status, number, decided),
                )
        codes = sorted({p[0] for p in DENIAL_POLICIES})
        cpts = ["99213", "99214", "97110", "97140", "72148", "73721", "20610", "93000"]
        for i in range(n_filler):
            claim_id = f"CLM-{2001 + i}"
            service = date(2025, 9, 1) + timedelta(days=rng.randint(0, 270))
            submitted = service + timedelta(days=rng.randint(2, 40))
            denied = submitted + timedelta(days=rng.randint(10, 35))
            code = rng.choice(codes)
            cpt = rng.choice(cpts)
            _insert_claim(
                conn,
                {
                    "claim_id": claim_id,
                    **_fake_identity(rng),
                    "payer": rng.choice(PAYERS),
                    "rendering_npi": str(rng.randint(1_000_000_000, 1_999_999_999)),
                    "cpt_code": cpt,
                    "modifier": "",
                    "icd10_code": "R69",
                    "billed_amount": round(rng.uniform(80, 3000), 2),
                    "service_date": service.isoformat(),
                    "submitted_date": submitted.isoformat(),
                    "denial_code": code,
                    "denial_date": denied.isoformat(),
                    "has_clinical_notes": int(rng.random() < 0.6),
                    "timely_filing_proof": int(rng.random() < 0.3),
                    "notes": "",
                },
            )
            if code == "CO-197" and rng.random() < 0.5:
                status = rng.choice(["approved", "pending", "denied"])
                conn.execute(
                    "INSERT INTO prior_auths VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        f"PA-{2001 + i}",
                        claim_id,
                        cpt,
                        status,
                        None if status == "pending" else f"AUTH-{rng.randint(10000, 99999)}",
                        None if status == "pending" else service.isoformat(),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    logger.info("Seeded %s anchor + %s filler claims into %s", len(ANCHORS), n_filler, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic claims database.")
    parser.add_argument("--db", default="var/claims.db")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_database(args.db, seed=args.seed)


if __name__ == "__main__":
    main()
