"""Confidence scoring — combine detection signals into one calibrated score.

The combined ``confidence`` is the system's probability that the content is
**AI-generated**, in ``[0, 1]``:

* ``0.0`` = certainly human, ``1.0`` = certainly AI.
* ``0.5`` = maximal honest uncertainty — the signals are split or weak and the
  system declines to attribute the work either way.

Pipeline (all constants live in :mod:`config`):

1. **Weighted blend.** ``raw = Σ wᵢ·pᵢ`` over the signals that ran, with weights
   renormalised over the present signals (so a missing/unreliable signal does not
   silently bias the result).
2. **Disagreement → uncertainty.** ``raw`` is pulled toward 0.5 in proportion to
   how much the signals disagree (``spread`` between min and max). Conflicting
   evidence must read as *uncertain*, never as a confident verdict.
3. **Asymmetric banding.** The score maps to one of three attributions using
   thresholds that are deliberately harder to clear on the AI side
   (``AI_THRESHOLD`` 0.70) than the human side (``HUMAN_THRESHOLD`` 0.40). This
   is where false-positive aversion lives: borderline text lands in the wide
   "uncertain" band rather than being asserted as AI.

A 0.51 and a 0.95 therefore produce different attributions/labels — there is no
binary flip at 0.5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import config
from signals import SignalResult, clamp01

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """The combined verdict for one submission."""

    confidence: float          # combined P(AI), 0..1 — the reported "confidence"
    attribution: str           # likely_ai | uncertain | likely_human
    raw_weighted: float        # weighted blend before the disagreement pull
    disagreement: float        # spread between the most human/AI signals, 0..1
    components: dict[str, float] = field(default_factory=dict)   # signal -> p_ai
    weights: dict[str, float] = field(default_factory=dict)      # signal -> weight used
    notes: list[str] = field(default_factory=list)


def attribution_for(confidence: float) -> str:
    """Map a combined confidence to one of the three attributions (asymmetric)."""
    if confidence >= config.AI_THRESHOLD:
        return config.ATTRIBUTION_LIKELY_AI
    if confidence <= config.HUMAN_THRESHOLD:
        return config.ATTRIBUTION_LIKELY_HUMAN
    return config.ATTRIBUTION_UNCERTAIN


def combine(signals: list[SignalResult], use_ensemble: bool | None = None) -> ScoreResult:
    """Combine signal results into a calibrated :class:`ScoreResult`.

    Args:
        signals: One :class:`SignalResult` per signal that ran.
        use_ensemble: Which weight table to use. ``None`` -> pick automatically:
            ensemble weights if a repetition signal is present, else base weights.
    """
    if not signals:
        raise ValueError("combine() requires at least one signal result")

    if use_ensemble is None:
        use_ensemble = any(s.name == config.SIGNAL_REPETITION for s in signals)
    weight_table = config.WEIGHTS_ENSEMBLE if use_ensemble else config.WEIGHTS_BASE

    notes: list[str] = []

    # --- 1. Weighted blend over AVAILABLE signals --------------------------- #
    # Signals that report unavailable (e.g. the repetition signal on short text,
    # where its reading would be confidently wrong) are dropped entirely and the
    # remaining weights are renormalised — never blended in at a fake 0.5, which
    # would silently bias every short submission toward "uncertain".
    effective: list[tuple[SignalResult, float]] = []
    for s in signals:
        w = weight_table.get(s.name, 0.0)
        if w <= 0:
            notes.append(f"signal '{s.name}' has no weight in the active table; ignored")
            continue
        if not s.available:
            notes.append(f"signal '{s.name}' unavailable (e.g. text too short); dropped")
            continue
        effective.append((s, w))

    if not effective:
        # Degenerate fallback: nothing was usable. Re-include any weighted signal
        # so we still return a (low-trust) verdict rather than crashing.
        effective = [(s, weight_table.get(s.name, 0.0) or 1.0) for s in signals]
        notes.append("no signal reported available; using all signals at low trust")

    total_w = sum(w for _, w in effective)
    components = {s.name: round(s.ai_probability, 4) for s, _ in effective}
    weights_used = {s.name: round(w / total_w, 4) for s, w in effective}
    raw = sum(s.ai_probability * w for s, w in effective) / total_w

    # --- 2. Disagreement -> pull toward 0.5 (uncertainty) ------------------- #
    probs = [s.ai_probability for s, _ in effective]
    spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
    pull = config.MAX_DISAGREEMENT_PULL * spread          # up to MAX_PULL
    confidence = (1.0 - pull) * raw + pull * 0.5
    confidence = clamp01(confidence)
    if spread >= 0.34:
        notes.append(
            f"signals disagree (spread={spread:.2f}); pulled toward uncertain by {pull:.2f}"
        )

    # --- 3. Asymmetric banding --------------------------------------------- #
    attribution = attribution_for(confidence)

    logger.info(
        "Scoring: raw=%.3f spread=%.3f -> confidence=%.3f attribution=%s",
        raw, spread, confidence, attribution,
    )
    return ScoreResult(
        confidence=round(confidence, 4),
        attribution=attribution,
        raw_weighted=round(raw, 4),
        disagreement=round(spread, 4),
        components=components,
        weights=weights_used,
        notes=notes,
    )
