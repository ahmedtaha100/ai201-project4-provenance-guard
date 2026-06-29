"""Central configuration for Provenance Guard.

This module is the **single source of truth** for the detection-signal weights,
the score-to-band thresholds, and the asymmetry parameters. ``scoring.py`` and
``labels.py`` import these constants directly, so the running system can never
silently diverge from the numbers documented in ``planning.md`` / ``README.md``.

If you change a threshold here, the docs that quote it are describing this file.
"""

from __future__ import annotations

import os
import secrets

# --------------------------------------------------------------------------- #
# Groq / LLM signal
# --------------------------------------------------------------------------- #
# ``llama-3.3-70b-versatile`` is the recommended Groq production model as of the
# project spec. If Groq deprecates it, override with the GROQ_MODEL env var.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "20"))

# --------------------------------------------------------------------------- #
# Signal weights — base pipeline (2 distinct signals)
# --------------------------------------------------------------------------- #
# The LLM signal is weighted higher: it judges the text holistically and is the
# more reliable single detector. Stylometry is a valuable but noisier structural
# cross-check. Weights are pulled by signal name in scoring.combine().
WEIGHT_LLM: float = 0.60
WEIGHT_STYLOMETRIC: float = 0.40

# --------------------------------------------------------------------------- #
# Signal weights — ensemble (stretch: 3 distinct signals)
# --------------------------------------------------------------------------- #
# When the repetition/predictability signal is added, weights are renormalised
# over whichever signals actually ran.
ENSEMBLE_WEIGHT_LLM: float = 0.50
ENSEMBLE_WEIGHT_STYLOMETRIC: float = 0.30
ENSEMBLE_WEIGHT_REPETITION: float = 0.20

# Whether /submit runs the 3-signal ensemble (stretch) or the 2-signal base.
USE_ENSEMBLE: bool = os.getenv("USE_ENSEMBLE", "1") not in ("0", "false", "False")

# Canonical signal names (used as dict keys + weight lookup).
SIGNAL_LLM: str = "llm"
SIGNAL_STYLOMETRIC: str = "stylometric"
SIGNAL_REPETITION: str = "repetition"

WEIGHTS_BASE: dict[str, float] = {
    SIGNAL_LLM: WEIGHT_LLM,
    SIGNAL_STYLOMETRIC: WEIGHT_STYLOMETRIC,
}
WEIGHTS_ENSEMBLE: dict[str, float] = {
    SIGNAL_LLM: ENSEMBLE_WEIGHT_LLM,
    SIGNAL_STYLOMETRIC: ENSEMBLE_WEIGHT_STYLOMETRIC,
    SIGNAL_REPETITION: ENSEMBLE_WEIGHT_REPETITION,
}

# --------------------------------------------------------------------------- #
# Disagreement -> uncertainty
# --------------------------------------------------------------------------- #
# When signals disagree, the combined score is blended toward 0.5 ("honest I
# don't know") by up to this fraction of the spread. Conflicting evidence must
# read as *uncertain*, not as a confident verdict in either direction.
MAX_DISAGREEMENT_PULL: float = 0.50

# --------------------------------------------------------------------------- #
# Decision bands  (ASYMMETRIC — this is where false-positive aversion lives)
# --------------------------------------------------------------------------- #
# ``confidence`` == P(content is AI-generated), in [0, 1].
#
#   confidence >= AI_THRESHOLD      -> likely_ai     (HIGH bar)
#   confidence <= HUMAN_THRESHOLD   -> likely_human  (lower bar)
#   in between                      -> uncertain
#
# The bar to call something AI (0.70) is deliberately higher and farther from
# 0.5 than the bar to call it human (0.40). Calling a real person's writing
# "AI-generated" is the costlier mistake on a creative platform, so borderline
# cases fall into the wide "uncertain" band instead of being asserted as AI.
AI_THRESHOLD: float = 0.70
HUMAN_THRESHOLD: float = 0.40

ATTRIBUTION_LIKELY_AI: str = "likely_ai"
ATTRIBUTION_UNCERTAIN: str = "uncertain"
ATTRIBUTION_LIKELY_HUMAN: str = "likely_human"

# --------------------------------------------------------------------------- #
# Stylometry reliability floor
# --------------------------------------------------------------------------- #
# Below this many words, sentence-level statistics are too sparse to trust, so
# the stylometric score is blended toward 0.5 proportionally (see stylometric.py).
# Stylometry still *contributes* below this floor (just weakly); it is the
# repetition signal that is dropped entirely on short text.
MIN_RELIABLE_WORDS: int = 40

# The repetition/predictability signal needs enough tokens for n-gram statistics
# to mean anything. Below this it reports unavailable and is dropped from scoring
# (rather than contributing a confidently-wrong reading on short, dense text).
MIN_REP_WORDS: int = 50

# Reject input longer than this many characters (abuse / cost guard).
MAX_TEXT_CHARS: int = 50_000

# Supported content types (stretch: multi-modal).
CONTENT_TYPE_TEXT: str = "text"
CONTENT_TYPE_IMAGE_META: str = "image_metadata"
SUPPORTED_CONTENT_TYPES: tuple[str, ...] = (CONTENT_TYPE_TEXT, CONTENT_TYPE_IMAGE_META)

# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
# 10/min comfortably covers a real creator editing + resubmitting in a burst;
# 100/day caps sustained automated abuse and protects the Groq quota.
RATE_LIMITS: str = "10 per minute;100 per day"
RATELIMIT_STORAGE_URI: str = "memory://"

# --------------------------------------------------------------------------- #
# Audit log / persistence
# --------------------------------------------------------------------------- #
DB_PATH: str = os.getenv("PROVENANCE_DB", "provenance_guard.db")

# --------------------------------------------------------------------------- #
# Provenance certificate (stretch)
# --------------------------------------------------------------------------- #
# HMAC secret for signing "verified human" certificates. NEVER ship a hardcoded
# default — a known constant would let anyone forge a "Verified Human" credential.
# In production set PROVENANCE_CERT_SECRET (from a secret manager). If it is unset,
# we generate a random per-process secret: dev still works, but certificates do
# not survive a restart and cannot be forged from a published constant. app.py
# logs a warning when the ephemeral secret is in use.
_ENV_CERT_SECRET: str | None = os.getenv("PROVENANCE_CERT_SECRET")
CERT_SECRET: str = _ENV_CERT_SECRET or secrets.token_hex(32)
CERT_SECRET_EPHEMERAL: bool = _ENV_CERT_SECRET is None
# A verification writing sample must read at least this human (confidence <= X)
# and be at least this many words to earn a certificate.
CERT_MAX_AI_CONFIDENCE: float = HUMAN_THRESHOLD
CERT_MIN_WORDS: int = 30
