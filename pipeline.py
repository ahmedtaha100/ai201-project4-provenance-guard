"""Detection pipeline orchestration.

The one place that runs the signals and combines them, so ``/submit`` and the
provenance-certificate flow share identical classification behaviour. Keeping
this here (rather than in ``app.py``) means the scoring path is import-safe and
unit-testable without Flask.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import config
import scoring
from scoring import ScoreResult
from signals import SignalResult, ensemble, llm_signal, stylometric

logger = logging.getLogger(__name__)


@dataclass
class ClassifyResult:
    """Everything ``/submit`` needs to build a response + audit row."""

    score: ScoreResult
    signals: list[SignalResult]

    def by_name(self, name: str) -> Optional[SignalResult]:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def signal_score(self, name: str) -> Optional[float]:
        """Return a signal's score, or ``None`` if it did not run/contribute.

        Signals that reported unavailable (e.g. the repetition signal on short
        text) are dropped from scoring, so they are recorded as ``None`` in both
        the API response and the audit log rather than a misleading neutral 0.5.
        """
        s = self.by_name(name)
        if s is None or not s.available:
            return None
        return s.ai_probability

    @property
    def fallback_used(self) -> bool:
        return any(s.fallback_used for s in self.signals)

    def signal_detail(self) -> dict[str, dict]:
        return {s.name: s.detail for s in self.signals}


def classify(text: str, content_type: str = config.CONTENT_TYPE_TEXT) -> ClassifyResult:
    """Run all configured signals over ``text`` and combine them.

    Args:
        text: Raw content (already validated by the caller).
        content_type: ``"text"`` or ``"image_metadata"`` (changes the LLM prompt).

    Returns:
        A :class:`ClassifyResult` bundling the combined score and every signal.
    """
    signals: list[SignalResult] = [
        llm_signal.assess(text, content_type=content_type),
        stylometric.assess(text),
    ]
    if config.USE_ENSEMBLE:
        signals.append(ensemble.assess(text))

    score = scoring.combine(signals)
    logger.info(
        "Pipeline classify: type=%s attribution=%s confidence=%.3f signals=%s",
        content_type, score.attribution, score.confidence,
        {s.name: round(s.ai_probability, 3) for s in signals},
    )
    return ClassifyResult(score=score, signals=signals)
