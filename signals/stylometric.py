"""Signal 2 — Stylometric heuristics (pure Python, structural).

**What it measures.** The *statistical shape* of the writing, independent of
meaning. Three sub-metrics are combined into one AI-likeness score:

1. ``burstiness``  — coefficient of variation of sentence lengths. Humans write
   in bursts (a long sentence, then a short punchy one); AI text is metronomic.
   Low variance  -> more AI-like.
2. ``informality`` — density of contractions, casual punctuation (!, ?, ...),
   lowercase sentence starts and slang. Casual human writing is rich in these;
   default AI prose is clean and formal. High informality -> more human.
3. ``diversity``   — moving-average type-token ratio (MATTR, window 25), a
   length-robust vocabulary-diversity measure. Very low lexical diversity
   (repetitive word choice) nudges toward AI.

**Why it differs from the LLM signal.** This signal never reads for *meaning* —
it only counts shapes. That makes it a genuinely independent cross-check: it can
catch machine text the LLM vouches for, and (importantly) it *fails differently*,
which is what makes the combination informative.

**Blind spots (documented, and handled downstream).**
* Formal human writing (academic, legal, monetary-policy prose) is uniform and
  complex, so it scores high — a false positive. Mitigated by the LLM signal and
  the disagreement-to-uncertain rule in scoring.py.
* A repetitive, simple-vocabulary poem looks AI-uniform to this signal.
* Short text (< MIN_RELIABLE_WORDS) yields unstable statistics; the score is
  blended toward 0.5 proportionally so it cannot dominate.
"""

from __future__ import annotations

import logging
import re
from statistics import mean, pstdev

import config
from signals import SignalResult, clamp01
from signals.text_utils import sentence_lengths, sentences, word_count, words

logger = logging.getLogger(__name__)

# Sub-metric weights (sum to 1.0). Burstiness is the most reliable separator, so
# it carries the most weight; lexical diversity is the noisiest, so the least.
_W_BURST = 0.40
_W_INFORMAL = 0.35
_W_DIVERSITY = 0.25

_CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
_SLANG = (
    "lol", "lmao", "ngl", "tbh", "idk", "imo", "btw", "kinda", "gonna",
    "wanna", "yeah", "nah", "tho", "honestly", "pretty much", "way too",
)


def assess(text: str) -> SignalResult:
    """Compute the stylometric AI-likeness score for ``text``."""
    sent_lengths = sentence_lengths(text)
    n_words = word_count(text)
    toks = words(text)

    if not sent_lengths or n_words == 0:
        # Degenerate input — no usable structure.
        return SignalResult(
            name=config.SIGNAL_STYLOMETRIC,
            ai_probability=0.5,
            detail={"note": "empty_or_unstructured_text"},
            available=False,
        )

    burst_ai = _burstiness_ai_score(sent_lengths)
    informal_ai, informal_raw = _informality_ai_score(text, sentences(text), toks)
    diversity_ai, mattr = _diversity_ai_score(toks)

    raw_score = _W_BURST * burst_ai + _W_INFORMAL * informal_ai + _W_DIVERSITY * diversity_ai

    # Reliability shrink for short text: blend toward 0.5.
    reliable = n_words >= config.MIN_RELIABLE_WORDS
    if not reliable:
        shrink = n_words / config.MIN_RELIABLE_WORDS
        score = 0.5 + (raw_score - 0.5) * shrink
    else:
        score = raw_score

    prob = clamp01(score)
    # Stylometry always *contributes* for any structured text (the short-text
    # shrink toward 0.5 already limits its influence); only genuinely degenerate
    # input is dropped. This guarantees the pipeline keeps >=2 signals (LLM +
    # stylometric) for any real submission.
    detail = {
        "source": "stylometric",
        "ai_probability": prob,
        "reliable": reliable,
        "word_count": n_words,
        "sentence_count": len(sent_lengths),
        "avg_sentence_length": round(mean(sent_lengths), 2),
        "sentence_length_cv": round(_cv(sent_lengths), 3),
        "metrics": {
            "burstiness_ai": round(burst_ai, 3),
            "informality_ai": round(informal_ai, 3),
            "informality_raw": round(informal_raw, 3),
            "diversity_ai": round(diversity_ai, 3),
            "mattr": round(mattr, 3),
        },
        "weights": {"burstiness": _W_BURST, "informality": _W_INFORMAL, "diversity": _W_DIVERSITY},
    }
    logger.info(
        "Stylometric signal: p_ai=%.3f (burst=%.2f informal=%.2f div=%.2f reliable=%s)",
        prob, burst_ai, informal_ai, diversity_ai, reliable,
    )
    return SignalResult(
        name=config.SIGNAL_STYLOMETRIC,
        ai_probability=prob,
        detail=detail,
        available=True,  # structured text always contributes (shrunk if short)
    )


# --------------------------------------------------------------------------- #
# Sub-metrics
# --------------------------------------------------------------------------- #
def _cv(values: list[int]) -> float:
    """Coefficient of variation (population std / mean) of ``values``."""
    m = mean(values)
    if m == 0:
        return 0.0
    return pstdev(values) / m


def _burstiness_ai_score(sent_lengths: list[int]) -> float:
    """Map sentence-length variability to an AI-likeness score in [0, 1].

    Human prose typically has CV ~0.5-0.9; uniform AI prose ~0.2-0.4. We map
    CV >= 0.70 -> 0.0 (very human) and CV <= 0.25 -> 1.0 (very AI), linear between.
    A single sentence has no variance, so it is treated as neutral (0.5).
    """
    if len(sent_lengths) < 2:
        return 0.5
    cv = _cv(sent_lengths)
    hi, lo = 0.70, 0.25
    return clamp01((hi - cv) / (hi - lo))


def _informality_ai_score(text: str, sents: list[str], toks: list[str]) -> tuple[float, float]:
    """Density of casual-writing markers -> (AI-likeness, raw-informality).

    Returns AI-likeness = 1 - informality (more casual => more human => lower AI).
    """
    n_words = max(1, len(toks))
    n_sents = max(1, len(sents))

    contractions = len(_CONTRACTION_RE.findall(text))
    exclaims = text.count("!")
    questions = text.count("?")
    ellipses = text.count("...")
    slang = sum(text.lower().count(s) for s in _SLANG)
    lowercase_starts = sum(1 for s in sents if s[:1].islower())

    # Per-unit densities, each capped so one feature cannot dominate.
    informality = (
        min(contractions / n_words * 6.0, 0.5)
        + min((exclaims + questions + ellipses) / n_sents, 0.3)
        + min(slang / n_words * 12.0, 0.4)
        + min(lowercase_starts / n_sents, 0.3)
    )
    informality_raw = clamp01(informality)
    return 1.0 - informality_raw, informality_raw


def _diversity_ai_score(toks: list[str]) -> tuple[float, float]:
    """Moving-average type-token ratio (MATTR) -> (AI-likeness, mattr).

    Low lexical diversity (repetitive word choice) nudges toward AI. We map
    MATTR <= 0.55 -> 1.0 and MATTR >= 0.80 -> 0.0, linear between. This is the
    weakest sub-signal (hence the lowest weight) because diversity is genuinely
    ambiguous between human and AI text.
    """
    mattr = _mattr(toks, window=25)
    hi, lo = 0.80, 0.55
    ai = clamp01((hi - mattr) / (hi - lo))
    return ai, mattr


def _mattr(toks: list[str], window: int = 25) -> float:
    """Moving-average type-token ratio over ``window``-token windows."""
    if not toks:
        return 0.7  # neutral-ish default
    if len(toks) <= window:
        return len(set(toks)) / len(toks)
    ratios: list[float] = []
    for i in range(len(toks) - window + 1):
        chunk = toks[i : i + window]
        ratios.append(len(set(chunk)) / window)
    return mean(ratios) if ratios else len(set(toks)) / len(toks)
