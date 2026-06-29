"""Signal 3 — Repetition & predictability (stretch: ensemble).

**What it measures.** How *self-repetitive and predictable* the text is at the
token level — a cheap, dependency-free proxy for the low "perplexity" that
characterises machine-generated text:

1. ``ngram_repetition`` — fraction of bigrams/trigrams that are reused. AI text
   recycles phrasing and scaffolding more than human text.
2. ``predictability``   — share of tokens drawn from a small set of very common
   English words, plus a low-content-word-diversity term. AI text leans on safe,
   high-frequency vocabulary, so it is more "predictable".

**Why it differs from the other two signals.** The LLM judges meaning; the
stylometric signal judges *sentence-shape* variance. This signal judges
*token-sequence* repetition/predictability — a third, orthogonal axis. It is
what turns the base pair into a genuine 3-signal ensemble.

**Blind spot.** Short texts have too few n-grams to estimate repetition; poems
and refrains repeat by design; templated human writing (forms, recipes) can look
predictable. It is the lightest-weighted ensemble member for these reasons.
"""

from __future__ import annotations

import logging

import config
from signals import SignalResult, clamp01
from signals.text_utils import words

logger = logging.getLogger(__name__)

# Sub-metric weights within this signal (sum to 1.0).
_W_NGRAM = 0.6
_W_PREDICT = 0.4

# ~120 of the most common English function/filler words. Membership is a coarse
# predictability proxy; the exact list need only be "common words".
_COMMON_WORDS = frozenset(
    """the of and a to in is was he for it with as his on be at by i this had not
    are but from or have an they which one you were her all she there would their
    we him been has when who will more no if out so said what up its about into
    than them can only other new some could time these two may then do first any
    my now such like our over man me even most made after also did many before
    must through back years where much your way well down should because each just
    those people mr how too little state good very make world still own see men
    work long get here between both life being under never day same another know
    while last might us great old year off come since against go came right used
    take three states himself few house use during without again place american
    around however home small found thought went say part once general high upon
    school every don does got united left number course war until always away
    something fact though water less public put think almost hand enough far took
    head yet government system better set told nothing night end why called
    didn eyes find going look asked later knew""".split()
)


def assess(text: str) -> SignalResult:
    """Compute the repetition/predictability AI-likeness score for ``text``."""
    toks = words(text)
    n = len(toks)
    if n < config.MIN_REP_WORDS:
        # Too few tokens for trustworthy n-gram statistics. Report unavailable so
        # scoring drops this signal rather than letting a confidently-wrong
        # reading on short, dense text inflate disagreement.
        return SignalResult(
            name=config.SIGNAL_REPETITION,
            ai_probability=0.5,
            detail={"note": "too_short_for_repetition_stats", "word_count": n,
                    "min_words": config.MIN_REP_WORDS},
            available=False,
        )

    ngram_ai, bi_rep, tri_rep = _ngram_repetition_score(toks)
    predict_ai, common_share, content_diversity = _predictability_score(toks)

    raw = _W_NGRAM * ngram_ai + _W_PREDICT * predict_ai
    prob = clamp01(raw)
    detail = {
        "source": "repetition_predictability",
        "ai_probability": prob,
        "word_count": n,
        "metrics": {
            "bigram_repetition": round(bi_rep, 3),
            "trigram_repetition": round(tri_rep, 3),
            "ngram_ai": round(ngram_ai, 3),
            "common_word_share": round(common_share, 3),
            "content_word_diversity": round(content_diversity, 3),
            "predictability_ai": round(predict_ai, 3),
        },
        "weights": {"ngram": _W_NGRAM, "predictability": _W_PREDICT},
    }
    logger.info("Repetition signal: p_ai=%.3f (ngram=%.2f predict=%.2f)", prob, ngram_ai, predict_ai)
    return SignalResult(
        name=config.SIGNAL_REPETITION,
        ai_probability=prob,
        detail=detail,
        available=True,
    )


# --------------------------------------------------------------------------- #
# Sub-metrics
# --------------------------------------------------------------------------- #
def _ngram_repetition_score(toks: list[str]) -> tuple[float, float, float]:
    """Reuse rate of bi/tri-grams -> (AI-likeness, bigram_rep, trigram_rep).

    repetition = 1 - unique/total. Human text typically reuses few n-grams;
    higher reuse -> more AI-like. Mapped so rep >= 0.45 -> ~1.0.
    """
    bi_rep = _repetition(toks, 2)
    tri_rep = _repetition(toks, 3)
    combined = 0.5 * bi_rep + 0.5 * tri_rep
    ai = clamp01(combined / 0.45)
    return ai, bi_rep, tri_rep


def _repetition(toks: list[str], k: int) -> float:
    """Fraction of ``k``-grams that are repeats (1 - unique/total)."""
    if len(toks) < k + 1:
        return 0.0
    grams = [tuple(toks[i : i + k]) for i in range(len(toks) - k + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def _predictability_score(toks: list[str]) -> tuple[float, float, float]:
    """Common-word reliance + low content diversity -> (AI-likeness, ...).

    High share of very common words and low diversity among *content* (non-common)
    words both signal predictable, low-perplexity text.
    """
    n = len(toks)
    common = sum(1 for t in toks if t in _COMMON_WORDS)
    common_share = common / n

    content = [t for t in toks if t not in _COMMON_WORDS]
    content_diversity = (len(set(content)) / len(content)) if content else 1.0

    # Common share: 0.40 -> 0.0, 0.65 -> 1.0 (more common words => more predictable).
    share_ai = clamp01((common_share - 0.40) / (0.65 - 0.40))
    # Content diversity: 0.85 -> 0.0, 0.55 -> 1.0 (less diverse => more predictable).
    div_ai = clamp01((0.85 - content_diversity) / (0.85 - 0.55))
    ai = clamp01(0.6 * share_ai + 0.4 * div_ai)
    return ai, common_share, content_diversity
