"""Appeals workflow.

A creator who believes a classification is wrong can contest it. An appeal does
**not** trigger automated re-classification (by design — the human reviewer is
the safety valve). It:

1. looks up the original decision by ``content_id``,
2. stores the creator's reasoning beside that decision,
3. flips the content's status to ``under_review``,
4. returns a confirmation the creator can see.

The reviewer queue (status == ``under_review``) is exposed via
``audit.get_under_review`` and surfaced on the analytics dashboard.
"""

from __future__ import annotations

import logging
from typing import Any

import audit

logger = logging.getLogger(__name__)


class AppealError(Exception):
    """Raised when an appeal cannot be accepted (bad input or unknown content)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def submit_appeal(content_id: str, creator_reasoning: str) -> dict[str, Any]:
    """Record an appeal and return a confirmation payload.

    Raises:
        AppealError: 400 if input is malformed, 404 if ``content_id`` is unknown.
    """
    if not isinstance(content_id, str) or not content_id.strip():
        raise AppealError("`content_id` is required.", 400)
    if not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        raise AppealError("`creator_reasoning` is required.", 400)

    reasoning = creator_reasoning.strip()
    updated = audit.record_appeal(content_id.strip(), reasoning)
    if updated is None:
        raise AppealError(
            f"No submission found for content_id '{content_id}'.", 404
        )

    logger.info("Appeal accepted for content_id=%s", content_id)
    return {
        "content_id": updated["content_id"],
        "status": updated["status"],  # "under_review"
        "appeal_reasoning": updated["appeal_reasoning"],
        "appeal_timestamp": updated["appeal_timestamp"],
        "original_attribution": updated["attribution"],
        "original_confidence": updated["confidence"],
        "message": (
            "Your appeal has been received and logged. This submission is now "
            "marked 'under_review' and a human will look at it. No automated "
            "re-classification is performed."
        ),
    }
