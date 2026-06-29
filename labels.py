"""Transparency labels — turn a confidence score into reader-facing text.

The label is the UX surface of the whole system: the one thing a non-technical
reader actually sees. There are exactly three variants, selected by the
*attribution* band from :mod:`scoring` (never a raw threshold re-check here, so
the label can never disagree with the audit log's attribution).

The template strings below are the **canonical wording** — ``planning.md`` and
``README.md`` quote them verbatim, and ``tests/`` assert that every variant is
reachable and unchanged. ``{pct}`` is replaced with the integer AI-likelihood
percentage (``round(confidence * 100)``).
"""

from __future__ import annotations

import config
from scoring import ScoreResult

# --------------------------------------------------------------------------- #
# Canonical label templates  (quoted verbatim in planning.md / README.md)
# --------------------------------------------------------------------------- #
LABEL_LIKELY_AI = (
    "🤖 Likely AI-generated — Our automated check estimates a {pct}% likelihood "
    "that this text was produced mainly by an AI tool, which is above our "
    "high-confidence threshold. We're surfacing this for transparency. It is an "
    "estimate, not proof — if you wrote this yourself, you can appeal and a human "
    "will review it."
)

LABEL_LIKELY_HUMAN = (
    "✍️ Likely human-written — Our automated check found writing patterns "
    "consistent with a human author (estimated {pct}% likelihood of AI "
    "generation — low). No strong AI-generation signals were detected. This is an "
    "automated estimate, not a guarantee of authorship."
)

LABEL_UNCERTAIN = (
    "❓ Uncertain — We can't confidently tell whether this was written by a person "
    "or by an AI (estimated {pct}% likelihood of AI generation, inside our "
    "'unsure' range). Our signals were weak or in conflict, so we are deliberately "
    "not labeling it either way. Treat it as unverified; the creator can add "
    "context through an appeal."
)

# Short chips for compact UI surfaces.
SHORT_LABEL = {
    config.ATTRIBUTION_LIKELY_AI: "Likely AI-generated",
    config.ATTRIBUTION_LIKELY_HUMAN: "Likely human-written",
    config.ATTRIBUTION_UNCERTAIN: "Uncertain attribution",
}

_TEMPLATE = {
    config.ATTRIBUTION_LIKELY_AI: LABEL_LIKELY_AI,
    config.ATTRIBUTION_LIKELY_HUMAN: LABEL_LIKELY_HUMAN,
    config.ATTRIBUTION_UNCERTAIN: LABEL_UNCERTAIN,
}

# Stretch: provenance certificate badge appended to the label.
VERIFIED_HUMAN_BADGE = "✓ Verified Human creator"
_VERIFIED_SUFFIX = (
    " · {badge}: this creator completed human-verification, which is independent "
    "of the per-submission analysis above."
)


def make_label(score: ScoreResult, verified_human: bool = False) -> dict[str, object]:
    """Build the reader-facing label payload for a scored submission.

    Args:
        score: The combined :class:`ScoreResult`.
        verified_human: Whether the creator holds a provenance certificate
            (stretch); appends a "Verified Human" badge if so.

    Returns:
        ``{attribution, short, text, confidence_pct, badge?}``.
    """
    pct = round(score.confidence * 100)
    attribution = score.attribution
    text = _TEMPLATE[attribution].format(pct=pct)

    payload: dict[str, object] = {
        "attribution": attribution,
        "short": SHORT_LABEL[attribution],
        "confidence_pct": pct,
        "text": text,
    }
    if verified_human:
        payload["badge"] = VERIFIED_HUMAN_BADGE
        payload["text"] = text + _VERIFIED_SUFFIX.format(badge=VERIFIED_HUMAN_BADGE)
    return payload
