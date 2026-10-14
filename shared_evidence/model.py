"""Exact calculations for the exchangeable Gaussian signal model.

Each input log odds is a calibrated forecaster's posterior log odds under a
50% common prior. ``overlap`` is the correlation between distinct signals
conditional on the binary outcome, not the correlation between outcomes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def logistic(value: float) -> float:
    """Return the logistic transform without overflowing for large values."""
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def logit(probability: float) -> float:
    """Return log odds for a probability strictly between zero and one."""
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("probability must be finite and strictly between 0 and 1")
    return math.log(probability) - math.log1p(-probability)


def effective_forecasters(count: int, overlap: float) -> float:
    """Return effective independent signal count under exchangeable overlap."""
    if count < 1:
        raise ValueError("count must be positive")
    if not math.isfinite(overlap) or not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be finite and between 0 and 1")
    return count / (1.0 + (count - 1) * overlap)


def aggregate_log_odds(log_odds: Sequence[float], overlap: float) -> float:
    """Return exact posterior probability for the initial shared-signal model.

    ``overlap=0`` adds independent evidence. ``overlap=1`` counts identical
    evidence once when all individual signals agree, as the model requires.
    """
    effective_forecasters(len(log_odds), overlap)
    if any(not math.isfinite(value) for value in log_odds):
        raise ValueError("all log odds must be finite")
    adjusted_log_odds = math.fsum(log_odds) / (1.0 + (len(log_odds) - 1) * overlap)
    return logistic(adjusted_log_odds)
