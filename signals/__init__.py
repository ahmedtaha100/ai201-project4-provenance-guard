"""Detection-signal package.

Each signal is an independent function that takes raw text and returns a
:class:`SignalResult` — a normalised AI-probability in ``[0, 1]`` plus a
signal-specific ``detail`` breakdown. Signals capture *genuinely different*
properties of the text:

* ``llm_signal``   — semantic / holistic (does it *read* as AI?)
* ``stylometric``  — structural / statistical (sentence-shape uniformity)
* ``ensemble``     — repetition / predictability (n-gram + lexical entropy)

Keeping the contract uniform lets ``scoring.combine`` treat any subset of
signals the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignalResult:
    """The output of one detection signal.

    Attributes:
        name: Canonical signal name (see ``config.SIGNAL_*``).
        ai_probability: Calibrated probability the text is AI-generated, 0..1
            (0 = certainly human, 1 = certainly AI).
        detail: Signal-specific diagnostic breakdown (safe to log / surface).
        available: ``False`` when the signal could not produce a trustworthy
            reading (e.g. text too short); such signals are down-weighted.
        fallback_used: ``True`` when a degraded/deterministic path produced this
            result (e.g. the LLM signal ran its offline heuristic).
    """

    name: str
    ai_probability: float
    detail: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    fallback_used: bool = False

    def __post_init__(self) -> None:
        # Hard clamp — every downstream consumer relies on the [0, 1] invariant.
        self.ai_probability = max(0.0, min(1.0, float(self.ai_probability)))


def clamp01(value: float) -> float:
    """Clamp a float into the closed interval ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))


__all__ = ["SignalResult", "clamp01"]
