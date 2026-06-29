"""Structured audit log (SQLite).

Every attribution decision is persisted as one row so the system has a durable,
queryable record — never ``print()`` to a console. The ``content_id`` written
here is the same id returned by ``/submit`` and is the key ``/appeal`` looks up.

Each row captures: timestamp, content_id, creator_id, content_type, attribution,
combined confidence, **both/all individual signal scores**, the full signal
detail (JSON), status, and any appeal info.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

# Serialise writes from Flask's threaded dev server; SQLite handles the rest.
_WRITE_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    content_id        TEXT    NOT NULL UNIQUE,
    creator_id        TEXT    NOT NULL,
    timestamp         TEXT    NOT NULL,
    content_type      TEXT    NOT NULL DEFAULT 'text',
    text_excerpt      TEXT,
    word_count        INTEGER,
    attribution       TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    llm_score         REAL,
    stylometric_score REAL,
    repetition_score  REAL,
    signal_detail     TEXT,
    fallback_used     INTEGER NOT NULL DEFAULT 0,
    status            TEXT    NOT NULL DEFAULT 'classified',
    appeal_reasoning  TEXT,
    appeal_timestamp  TEXT
);
"""

_COLUMNS = (
    "content_id", "creator_id", "timestamp", "content_type", "text_excerpt",
    "word_count", "attribution", "confidence", "llm_score", "stylometric_score",
    "repetition_score", "signal_detail", "fallback_used", "status",
    "appeal_reasoning", "appeal_timestamp",
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the audit table if it does not yet exist (idempotent)."""
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(_SCHEMA)
    logger.info("Audit log ready at %s", config.DB_PATH)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (millisecond precision, 'Z')."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def record_submission(entry: dict[str, Any]) -> None:
    """Insert one classification decision.

    ``entry`` must contain at least ``content_id``, ``creator_id``,
    ``attribution`` and ``confidence``; everything else is optional and defaulted.
    ``signal_detail`` may be a dict (it is JSON-encoded here).
    """
    row = {
        "content_id": entry["content_id"],
        "creator_id": entry["creator_id"],
        "timestamp": entry.get("timestamp") or utc_now_iso(),
        "content_type": entry.get("content_type", config.CONTENT_TYPE_TEXT),
        "text_excerpt": entry.get("text_excerpt"),
        "word_count": entry.get("word_count"),
        "attribution": entry["attribution"],
        "confidence": float(entry["confidence"]),
        "llm_score": entry.get("llm_score"),
        "stylometric_score": entry.get("stylometric_score"),
        "repetition_score": entry.get("repetition_score"),
        "signal_detail": json.dumps(entry.get("signal_detail", {}), ensure_ascii=False),
        "fallback_used": 1 if entry.get("fallback_used") else 0,
        "status": entry.get("status", "classified"),
        "appeal_reasoning": entry.get("appeal_reasoning"),
        "appeal_timestamp": entry.get("appeal_timestamp"),
    }
    placeholders = ", ".join(f":{c}" for c in _COLUMNS)
    columns = ", ".join(_COLUMNS)
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(f"INSERT INTO audit_log ({columns}) VALUES ({placeholders})", row)
    logger.info(
        "Audit row written: content_id=%s attribution=%s confidence=%.3f",
        row["content_id"], row["attribution"], row["confidence"],
    )


def record_appeal(content_id: str, reasoning: str) -> Optional[dict[str, Any]]:
    """Attach an appeal to an existing decision and set status to under_review.

    Returns the updated row as a dict, or ``None`` if ``content_id`` is unknown.
    The original decision is preserved; only the appeal fields + status change.
    """
    ts = utc_now_iso()
    with _WRITE_LOCK, _connect() as conn:
        cur = conn.execute(
            "UPDATE audit_log SET status = 'under_review', "
            "appeal_reasoning = ?, appeal_timestamp = ? WHERE content_id = ?",
            (reasoning, ts, content_id),
        )
        if cur.rowcount == 0:
            logger.warning("Appeal for unknown content_id=%s", content_id)
            return None
        updated = conn.execute(
            "SELECT * FROM audit_log WHERE content_id = ?", (content_id,)
        ).fetchone()
    logger.info("Appeal recorded for content_id=%s (status=under_review)", content_id)
    return _row_to_dict(updated) if updated else None


def get_by_content_id(content_id: str) -> Optional[dict[str, Any]]:
    """Return a single decision by ``content_id`` (or ``None``)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM audit_log WHERE content_id = ?", (content_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_recent(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` decisions, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_under_review() -> list[dict[str, Any]]:
    """Return the reviewer queue: all decisions awaiting human review."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE status = 'under_review' ORDER BY appeal_timestamp DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_all() -> list[dict[str, Any]]:
    """Return every decision (used by analytics)."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id ASC").fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a DB row to a JSON-friendly dict (decoding ``signal_detail``)."""
    d = dict(row)
    d["fallback_used"] = bool(d.get("fallback_used"))
    raw = d.get("signal_detail")
    if raw:
        try:
            d["signal_detail"] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            pass
    return d
