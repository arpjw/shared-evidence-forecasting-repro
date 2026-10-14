"""Proper losses and training-only aggregation for the 50% prior model.

Lower scores are better. The binary Brier score is (p-y)**2 (no factor of 2).
Arithmetic pooling is a baseline, not a claim that calibration is preserved.
The exchangeable oracle requires the assumptions in model.py. Independent
questions are the statistical units; callers must keep related questions and
future information out of training. No test records enter discount fitting.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .model import effective_forecasters, logistic


@dataclass(frozen=True)
class ForecastRecord:
    log_odds: tuple[float, ...]
    outcome: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_odds", tuple(self.log_odds))
        if not self.log_odds or any(not math.isfinite(x) for x in self.log_odds):
            raise ValueError("a nonempty sequence of finite log odds is required")
        _validate_outcome(self.outcome)


@dataclass(frozen=True)
class Scores:
    log_loss: float
    brier: float


@dataclass(frozen=True)
class Evaluation:
    discount: float
    scores: dict[str, Scores]
    per_question: dict[str, tuple[Scores, ...]]


def _validate_outcome(outcome: int) -> None:
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")


def log_loss(log_odds: float, outcome: int) -> float:
    """Negative log likelihood, in nats, without probability clipping."""
    _validate_outcome(outcome)
    if not math.isfinite(log_odds):
        raise ValueError("log odds must be finite")
    wrong_direction = log_odds if outcome == 0 else -log_odds
    return max(wrong_direction, 0.0) + math.log1p(math.exp(-abs(wrong_direction)))


def brier_score(log_odds: float, outcome: int) -> float:
    _validate_outcome(outcome)
    # Use the error probability directly to retain small errors near p=1.
    error = logistic(log_odds if outcome == 0 else -log_odds)
    return error * error


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def prediction_logits(
    log_odds: Sequence[float], *, discount: float, overlap: float
) -> dict[str, float]:
    """Six forecasts, returned as logits so extreme losses remain accurate."""
    values = ForecastRecord(tuple(log_odds), 0).log_odds
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1]")
    effective_forecasters(len(values), overlap)
    total = math.fsum(values)
    if not math.isfinite(total):
        raise ValueError("sum of log odds must be finite")
    # log(mean p) - log(mean (1-p)); common log(n) cancels.
    mean_logit = _logsumexp([-log_loss(x, 1) for x in values]) - _logsumexp(
        [-log_loss(x, 0) for x in values]
    )
    return {
        "prior": 0.0,
        "individual": values[0],
        "arithmetic_mean": mean_logit,
        "independent": total,
        "oracle": total / (1 + (len(values) - 1) * overlap),
        "learned": discount * total,
    }


def fit_discount(training: Sequence[ForecastRecord]) -> float:
    """Minimize training log loss for alpha * sum(logits), alpha in [0,1].

    This constrained one-parameter model includes the prior and independence.
    It permits over-discounting below 1/n but neither reversals nor extremizing
    above independence. It need not remain calibrated under misspecification.
    Convexity gives monotone gradients; bisection avoids an optimizer dependency.
    A completely uninformative training set returns alpha=0 deterministically.
    """
    if not training:
        raise ValueError("training records must not be empty")
    totals = [math.fsum(row.log_odds) for row in training]
    if any(not math.isfinite(x) for x in totals):
        raise ValueError("summed training log odds must be finite")
    scale = max(abs(x) for x in totals)
    if scale == 0:
        return 0.0

    def gradient(alpha: float) -> float:
        # Scale leaves its sign unchanged and prevents large-gradient overflow.
        return math.fsum(
            (x / scale) * (logistic(alpha * x) if row.outcome == 0 else -logistic(-alpha * x))
            for x, row in zip(totals, training)
        )

    if gradient(0.0) >= 0:
        return 0.0
    if gradient(1.0) <= 0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low + high) / 2
        if gradient(mid) < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def evaluate(
    records: Sequence[ForecastRecord], *, discount: float, overlap: float
) -> Evaluation:
    """Score a fixed discount on held-out records; never fit on these labels."""
    if not records:
        raise ValueError("evaluation records must not be empty")
    per_question: dict[str, list[Scores]] = {}
    for row in records:
        for name, value in prediction_logits(row.log_odds, discount=discount, overlap=overlap).items():
            per_question.setdefault(name, []).append(
                Scores(log_loss(value, row.outcome), brier_score(value, row.outcome))
            )
    scores = {
        name: Scores(
            math.fsum(row.log_loss / len(rows) for row in rows),
            math.fsum(row.brier / len(rows) for row in rows),
        )
        for name, rows in per_question.items()
    }
    return Evaluation(discount, scores, {name: tuple(rows) for name, rows in per_question.items()})


def evaluate_train_test(
    training: Sequence[ForecastRecord], test: Sequence[ForecastRecord], *, overlap: float
) -> Evaluation:
    """Fit exclusively on training, then evaluate test. Caller ensures disjointness."""
    return evaluate(test, discount=fit_discount(training), overlap=overlap)


def paired_score_differences(
    evaluation: Evaluation, method: str, reference: str
) -> tuple[Scores, ...]:
    """Question-level method minus reference losses; negative favors method.

    Retaining pairs supports question-level bootstrap or replication summaries.
    This routine supplies no independence assumptions or confidence interval.
    """
    return tuple(
        Scores(left.log_loss - right.log_loss, left.brier - right.brier)
        for left, right in zip(evaluation.per_question[method], evaluation.per_question[reference])
    )
