"""Exact pooling when independent source groups share within-group noise.

Each forecaster has the same signal strength and marginal signal variance.
Members of one group have conditional correlation ``overlap``; separate
groups are conditionally independent given the binary outcome. The common
prior is 50%. This does not cover arbitrary correlation matrices.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

from .model import effective_forecasters, logistic


def grouped_oracle_log_odds(
    log_odds: Sequence[float], group_ids: Sequence[Hashable], overlap: float
) -> float:
    """Aggregate calibrated log odds without counting group evidence repeatedly."""
    if len(log_odds) != len(group_ids) or not log_odds:
        raise ValueError("log odds and group IDs must have the same positive length")
    effective_forecasters(len(log_odds), overlap)
    grouped: dict[Hashable, list[float]] = {}
    for value, group_id in zip(log_odds, group_ids):
        if not math.isfinite(value):
            raise ValueError("all log odds must be finite")
        grouped.setdefault(group_id, []).append(value)
    return math.fsum(
        math.fsum(values) / (1.0 + (len(values) - 1) * overlap)
        for values in grouped.values()
    )


def grouped_oracle_probability(
    log_odds: Sequence[float], group_ids: Sequence[Hashable], overlap: float
) -> float:
    """Return the posterior probability under the independent-group model."""
    return logistic(grouped_oracle_log_odds(log_odds, group_ids, overlap))
