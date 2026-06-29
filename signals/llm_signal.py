"""Signal 1 — LLM semantic classification (Groq).

**What it measures.** Whether the text *reads* as human- or AI-written when
judged holistically: tone, idea flow, hedging, generic "AI voice" phrasing,
and overall semantic coherence. This is the only signal that understands
*meaning* rather than surface statistics.

**Why it differs from the structural signals.** Stylometry and repetition look
at the *shape* of the text (sentence-length variance, n-gram reuse). The LLM
looks at the *substance* — it can flag a paragraph that is statistically varied
but still obviously machine-templated, or vouch for formal human prose that the
structural signals would wrongly flag.

**Blind spot.** It is non-deterministic, can be confidently wrong, and is the
easiest signal to fool with lightly human-edited AI text. It also depends on a
live API + key; when that is unavailable we fall back to a deterministic lexical
heuristic so the pipeline still runs end-to-end (``fallback_used=True``).
"""

from __future__ import annotations

import json
import logging
import re

import config
from signals import SignalResult, clamp01
from signals.text_utils import sentences, word_count, words

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a forensic text-attribution analyst for a creative-sharing "
    "platform. Estimate the probability (0.0-1.0) that the SUBMITTED CONTENT was "
    "generated mainly by an AI language model rather than written by a human.\n\n"
    "Calibration rubric — anchor your number to these bands:\n"
    "  0.85-1.00  GENERIC, content-free AI scaffolding: vague universal claims "
    "('transformative paradigm shift', 'stakeholders across various sectors', "
    "'responsible deployment'), mechanical both-sides hedging, text that could be "
    "pasted into an essay on almost any topic.\n"
    "  0.60-0.84  Polished and formulaic, several AI tells, thin substance.\n"
    "  0.40-0.59  Genuinely ambiguous: formal human writing that carries real "
    "substance, OR lightly human-edited AI.\n"
    "  0.15-0.39  Mostly human: a specific argument, concrete domain detail, or "
    "varied personal rhythm.\n"
    "  0.00-0.14  Obviously human: casual voice, slang, contractions, "
    "idiosyncrasy, typos, personal/sensory specifics.\n\n"
    "CRUCIAL: a formal or academic register is NOT by itself evidence of AI — much "
    "human writing is formal and impersonal. Weigh SUBSTANCE and SPECIFICITY: "
    "concrete, domain-grounded, non-interchangeable content points to a human even "
    "without any personal voice; generic claims that would fit any topic point to "
    "AI. On a writing platform wrongly accusing a human is costly, so when a formal "
    "text carries real specific substance do not exceed ~0.55, and do not push "
    "above 0.70 without genuinely generic AI phrasing. Judge holistically, not by "
    "length; be decisive at the extremes. Respond with STRICT JSON only, no prose, "
    "matching exactly:\n"
    '{"ai_probability": <float 0..1>, "confidence": "low|medium|high", '
    '"indicators": [<short strings>], "reasoning": "<one sentence>"}'
)

_USER_TEMPLATE_TEXT = (
    "Analyze the following submitted text and return the JSON verdict.\n\n"
    "--- BEGIN TEXT ---\n{content}\n--- END TEXT ---"
)

_USER_TEMPLATE_IMAGE_META = (
    "The following is an image *description / caption / metadata* a creator "
    "submitted alongside an image. Estimate the probability the DESCRIPTION text "
    "was AI-generated (e.g. produced by an image-captioning or chatbot model) "
    "rather than written by the human creator. Return the JSON verdict.\n\n"
    "--- BEGIN DESCRIPTION ---\n{content}\n--- END DESCRIPTION ---"
)


def assess(text: str, content_type: str = config.CONTENT_TYPE_TEXT) -> SignalResult:
    """Run the LLM semantic signal, falling back to a deterministic heuristic.

    Args:
        text: The raw content to classify.
        content_type: ``"text"`` or ``"image_metadata"`` (changes the prompt).

    Returns:
        A :class:`SignalResult` named ``config.SIGNAL_LLM``.
    """
    if not config.GROQ_API_KEY:
        logger.info("GROQ_API_KEY not set — using deterministic LLM fallback.")
        return _fallback_assess(text, reason="no_api_key")

    try:
        return _groq_assess(text, content_type)
    except Exception as exc:  # noqa: BLE001 — any failure must degrade, not crash.
        logger.warning("Groq signal failed (%s); using deterministic fallback.", exc)
        return _fallback_assess(text, reason=f"groq_error:{type(exc).__name__}")


# --------------------------------------------------------------------------- #
# Live Groq path
# --------------------------------------------------------------------------- #
def _groq_assess(text: str, content_type: str) -> SignalResult:
    """Call Groq and parse a calibrated AI-probability. Raises on any failure."""
    from groq import Groq  # imported lazily so the package is optional.

    client = Groq(api_key=config.GROQ_API_KEY, timeout=config.GROQ_TIMEOUT_SECONDS)
    template = (
        _USER_TEMPLATE_IMAGE_META
        if content_type == config.CONTENT_TYPE_IMAGE_META
        else _USER_TEMPLATE_TEXT
    )
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": template.format(content=text)},
        ],
        temperature=0.0,  # deterministic-ish: we want a stable verdict.
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or "{}"
    data = json.loads(raw)

    prob = clamp01(float(data.get("ai_probability", 0.5)))
    detail = {
        "source": "groq",
        "model": config.GROQ_MODEL,
        "ai_probability": prob,
        "model_confidence": str(data.get("confidence", "unknown")),
        "indicators": data.get("indicators", [])[:8],
        "reasoning": str(data.get("reasoning", ""))[:400],
    }
    logger.info("Groq signal: p_ai=%.3f model=%s", prob, config.GROQ_MODEL)
    return SignalResult(
        name=config.SIGNAL_LLM,
        ai_probability=prob,
        detail=detail,
        available=True,
        fallback_used=False,
    )


# --------------------------------------------------------------------------- #
# Deterministic offline fallback
# --------------------------------------------------------------------------- #
# Lexical "AI-voice" tells. None is conclusive alone; density is what matters.
_AI_TELL_PHRASES = (
    "paradigm shift", "it is important to note", "it is worth noting",
    "it is essential to", "furthermore", "moreover", "in conclusion",
    "in today's fast-paced", "delve into", "delving into", "tapestry",
    "navigate the complexities", "plays a crucial role", "plays a vital role",
    "plays a significant role", "stakeholders", "leverage", "underscore",
    "multifaceted", "ever-evolving", "ever-changing", "seamless", "seamlessly",
    "foster", "holistic", "nuanced", "realm of", "testament to", "boasts",
    "comprehensive", "responsible deployment", "ethical implications",
    "transformative", "robust framework", "various sectors", "it is crucial",
    "collaborate to ensure", "by leveraging", "in the realm", "cutting-edge",
    "unlock the potential", "a myriad of", "at the forefront",
)

# Markers of casual human writing — push the estimate toward "human".
_HUMAN_INFORMAL_MARKERS = (
    "honestly", "ngl", "tbh", "lol", "lmao", "kinda", "gonna", "wanna",
    "yeah", "ok so", "okay so", "i mean", "idk", "imo", "btw", "u ",
    "way too", "pretty much", "for like", "or something", "i guess",
    "probably", "anyway", "tho",
)

_CONTRACTION_RE = re.compile(r"\b\w+'(?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)


def _fallback_assess(text: str, reason: str) -> SignalResult:
    """Deterministic lexical estimate used when Groq is unavailable.

    Approximates the semantic signal with surface AI-voice cues so the pipeline
    still returns a meaningful, repeatable verdict (and clearly flags itself as a
    degraded path via ``fallback_used=True``).
    """
    lower = text.lower()
    n_words = max(1, word_count(text))
    n_sents = max(1, len(sentences(text)))

    tell_hits = sum(lower.count(p) for p in _AI_TELL_PHRASES)
    informal_hits = sum(lower.count(m) for m in _HUMAN_INFORMAL_MARKERS)
    contractions = len(_CONTRACTION_RE.findall(text))

    # Normalise by sentence count so long and short texts are comparable.
    tell_density = tell_hits / n_sents
    informal_density = (informal_hits + 0.5 * contractions) / n_sents

    # Start near "weakly human" (absence of AI tells is mild evidence of human),
    # add for AI tells, subtract for informal/contraction markers.
    score = 0.38 + 0.16 * tell_density - 0.14 * informal_density

    # Very short text: pull toward 0.5 (little lexical evidence either way).
    if n_words < config.MIN_RELIABLE_WORDS:
        shrink = n_words / config.MIN_RELIABLE_WORDS
        score = 0.5 + (score - 0.5) * shrink

    prob = clamp01(score)
    detail = {
        "source": "fallback_heuristic",
        "reason": reason,
        "ai_probability": prob,
        "ai_tell_hits": tell_hits,
        "informal_marker_hits": informal_hits,
        "contractions": contractions,
        "note": "Deterministic offline estimate — set GROQ_API_KEY for real LLM classification.",
    }
    logger.info("Fallback LLM signal: p_ai=%.3f (tells=%d informal=%d)", prob, tell_hits, informal_hits)
    return SignalResult(
        name=config.SIGNAL_LLM,
        ai_probability=prob,
        detail=detail,
        available=True,
        fallback_used=True,
    )
