"""Provenance certificate (stretch) — a "Verified Human" credential.

A creator can *earn* a credential by completing an extra verification step:
submitting a short, original writing sample. The sample is run through the same
detection pipeline; if it reads convincingly human (combined confidence at or
below the human threshold) and meets a minimum length, an HMAC-signed
certificate is issued and stored.

How it's displayed: once a creator is verified, every ``/submit`` response and
transparency label for that creator carries a ``✓ Verified Human creator`` badge
(see ``labels.make_label`` / ``app.py``). The badge is *independent* of the
per-submission analysis — it vouches for the creator, not for a single post — and
is shown alongside, never overriding, the per-post attribution.

This is a lightweight, demonstrable design (HMAC over creator_id + issue time),
not a production identity system; see README → Known limitations.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import sqlite3
import threading
from typing import Any, Optional

import config
import pipeline
from audit import utc_now_iso

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS certificates (
    creator_id     TEXT PRIMARY KEY,
    token          TEXT NOT NULL,
    issued_at      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'verified',
    basis_confidence REAL,
    basis_words    INTEGER
);
"""


class CertificateError(Exception):
    """Raised when a certificate cannot be issued."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the certificates table if needed (idempotent)."""
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(_SCHEMA)
    logger.info("Certificate store ready.")


def _sign(creator_id: str, issued_at: str) -> str:
    msg = f"{creator_id}|{issued_at}".encode("utf-8")
    return hmac.new(config.CERT_SECRET.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def request_certificate(creator_id: str, sample_text: str) -> dict[str, Any]:
    """Issue a certificate if the writing sample passes verification.

    Raises:
        CertificateError: 400 on bad input or if the sample fails verification.
    """
    if not isinstance(creator_id, str) or not creator_id.strip():
        raise CertificateError("`creator_id` is required.", 400)
    if not isinstance(sample_text, str) or not sample_text.strip():
        raise CertificateError("`sample_text` is required.", 400)

    creator_id = creator_id.strip()
    result = pipeline.classify(sample_text)
    confidence = result.score.confidence
    word_count = sum(1 for _ in sample_text.split())

    if word_count < config.CERT_MIN_WORDS:
        raise CertificateError(
            f"Writing sample too short: need at least {config.CERT_MIN_WORDS} words, got {word_count}.",
            400,
        )
    if confidence > config.CERT_MAX_AI_CONFIDENCE:
        raise CertificateError(
            "Writing sample did not read as confidently human "
            f"(AI-likelihood {round(confidence * 100)}% exceeds the "
            f"{round(config.CERT_MAX_AI_CONFIDENCE * 100)}% ceiling). "
            "Submit a longer, original sample in your own voice.",
            400,
        )

    issued_at = utc_now_iso()
    token = _sign(creator_id, issued_at)
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            "INSERT INTO certificates "
            "(creator_id, token, issued_at, status, basis_confidence, basis_words) "
            "VALUES (?, ?, ?, 'verified', ?, ?) "
            "ON CONFLICT(creator_id) DO UPDATE SET "
            "token=excluded.token, issued_at=excluded.issued_at, status='verified', "
            "basis_confidence=excluded.basis_confidence, basis_words=excluded.basis_words",
            (creator_id, token, issued_at, confidence, word_count),
        )
    logger.info("Issued provenance certificate for creator_id=%s", creator_id)
    return {
        "creator_id": creator_id,
        "verified_human": True,
        "badge": "✓ Verified Human creator",
        "token": token,
        "issued_at": issued_at,
        "basis": {
            "sample_confidence": confidence,
            "sample_words": word_count,
            "explanation": (
                "Issued because the submitted writing sample read as confidently "
                f"human ({round(confidence * 100)}% AI-likelihood, at or below the "
                f"{round(config.CERT_MAX_AI_CONFIDENCE * 100)}% ceiling)."
            ),
        },
    }


def get_certificate(creator_id: str) -> Optional[dict[str, Any]]:
    """Return a creator's certificate as a dict, or ``None`` if not verified."""
    if not creator_id:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM certificates WHERE creator_id = ? AND status = 'verified'",
            (creator_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["verified_human"] = True
    d["badge"] = "✓ Verified Human creator"
    return d


def is_verified_human(creator_id: str) -> bool:
    """Return whether ``creator_id`` currently holds a valid certificate."""
    cert = get_certificate(creator_id)
    if not cert:
        return False
    # Re-derive the signature to confirm the stored token is authentic.
    expected = _sign(cert["creator_id"], cert["issued_at"])
    return hmac.compare_digest(expected, cert["token"])
