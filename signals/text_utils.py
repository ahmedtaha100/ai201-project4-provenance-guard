"""Pure-Python text tokenisation helpers shared by the structural signals.

No third-party dependencies — the stylometric and repetition signals must run
even when nothing but the standard library is installed.
"""

from __future__ import annotations

import re

# A "word" is a run of letters/digits, allowing internal apostrophes/hyphens so
# that contractions ("don't") and compounds ("work-life") stay intact.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

# Split on sentence-ending punctuation followed by whitespace. Deliberately
# simple: stylometry only needs approximate sentence boundaries, and a heavy
# NLP dependency would defeat the "pure Python" design goal.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> list[str]:
    """Return the lowercased word tokens in ``text``."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def word_count(text: str) -> int:
    """Return the number of word tokens in ``text``."""
    return len(_WORD_RE.findall(text))


def sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty, stripped sentence strings."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def sentence_lengths(text: str) -> list[int]:
    """Return the word count of each sentence in ``text``."""
    return [word_count(s) for s in sentences(text) if word_count(s) > 0]
