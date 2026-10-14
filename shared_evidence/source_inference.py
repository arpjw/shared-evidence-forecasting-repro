"""Training-only recovery of independent source blocks from resolved forecasts.

Multiplying each training logit by Z=2Y-1 gives X_i=Z*ell_i. In this
symmetric Gaussian model E[X_i]=2a**2 and Cov(X)=4a**2 Sigma. Centered
signed-logit correlations therefore estimate noise overlap without requiring
both outcome classes. Unconditional forecast correlation cannot do this.
The graph threshold is a fixed conservative heuristic, not a calibrated test or
confidence bound. Connected components assume disjoint positive source blocks;
bridges, weak overlap, and distribution shift can defeat this model.

No true source labels, signal strengths, overlap values, or test outcomes enter
this module. In the zero-overlap limit the latent partition is unidentifiable.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .evaluate import ForecastRecord, fit_discount


@dataclass(frozen=True)
class SourceInferenceFit:
    """Frozen fit; group IDs are estimated column components, not source names."""

    group_ids: tuple[int, ...]
    overlap: float
    discount: float
    correlations: tuple[tuple[float, ...], ...]
    threshold: float
    residual_df: int

    @property
    def zero_variance_columns(self) -> tuple[int, ...]:
        """Columns whose signed training logits provide no covariance information.

        Their correlation rows and diagonals use zero as an explicit sentinel;
        this does not assert that an undefined Pearson correlation equals zero.
        """
        return tuple(i for i, row in enumerate(self.correlations) if row[i] == 0.0)

    @property
    def weights(self) -> tuple[float, ...]:
        sizes = {g: self.group_ids.count(g) for g in set(self.group_ids)}
        return tuple(1.0 / (1.0 + (sizes[g] - 1) * self.overlap)
                     for g in self.group_ids)

    def structural_log_odds(self, log_odds: Sequence[float]) -> float:
        """Estimated block posterior before fitted scalar calibration."""
        if len(log_odds) != len(self.group_ids):
            raise ValueError("forecast dimension must match fitted columns")
        if any(not math.isfinite(x) for x in log_odds):
            raise ValueError("forecast log odds must be finite")
        value = math.fsum(w * x for w, x in zip(self.weights, log_odds))
        if not math.isfinite(value):
            raise ValueError("weighted log odds must be finite")
        return value

    def predict_log_odds(self, log_odds: Sequence[float]) -> float:
        """Prediction accepts forecasts only; no evaluation outcomes are read."""
        return self.discount * self.structural_log_odds(log_odds)


def fit_source_inference(training: Sequence[ForecastRecord]) -> SourceInferenceFit:
    """Estimate blocks, their common overlap, and one bounded calibration slope.

    At least three training rows are required. All stages reuse the same training set; this
    is a fitted parametric procedure whose generalization is assessed on a fresh
    test split. The threshold is sqrt(2 log(n(n-1)/.05) / (m-1)), capped at one.
    Exact duplicate columns are linked even when the threshold reaches one.
    Zero residual variance supplies no evidence of a link. With no links the
    structural logits equal the independence sum, so calibration reduces to the
    original scalar learner. There is no hyperparameter search.
    """
    if not training:
        raise ValueError("training records must not be empty")
    n = len(training[0].log_odds)
    if any(len(row.log_odds) != n for row in training):
        raise ValueError("training records must have equal dimensions")
    m = len(training)
    if m < 3:
        raise ValueError("at least three training records are required")
    df = m - 1
    threshold = min(1.0, math.sqrt(2.0 * math.log(n * (n - 1) / 0.05) / df)) if n > 1 else 1.0

    # Positive rescaling per column avoids squaring large logits and leaves
    # Pearson correlations unchanged. Signing removes the outcome mean shift.
    residuals: list[tuple[float, ...]] = []
    norms: list[float] = []
    for i in range(n):
        scale = max(abs(row.log_odds[i]) for row in training)
        column = tuple((2 * row.outcome - 1) * row.log_odds[i] / scale if scale else 0.0
                       for row in training)
        mean = math.fsum(x / m for x in column)
        residual = tuple(x - mean for x in column)
        residuals.append(residual)
        norms.append(math.sqrt(math.fsum(x * x for x in residual)))
    correlations = [[0.0] * n for _ in range(n)]
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        correlations[i][i] = 1.0 if norms[i] else 0.0
        for j in range(i):
            r = (math.fsum((x / norms[i]) * (z / norms[j])
                          for x, z in zip(residuals[i], residuals[j]))
                 if norms[i] and norms[j] else 0.0)
            r = min(1.0, max(-1.0, r))
            correlations[i][j] = correlations[j][i] = r
            duplicate = norms[i] > 0 and all(row.log_odds[i] == row.log_odds[j] for row in training)
            if r > threshold or duplicate:
                adjacency[i].add(j)
                adjacency[j].add(i)
    groups = [-1] * n
    group = 0
    for i in range(n):
        if groups[i] != -1:
            continue
        pending = [i]
        groups[i] = group
        while pending:
            for j in sorted(adjacency[pending.pop()]):
                if groups[j] == -1:
                    groups[j] = group
                    pending.append(j)
        group += 1
    within = [correlations[i][j] for i in range(n) for j in range(i)
              if groups[i] == groups[j]]
    overlap = min(1.0, max(0.0, math.fsum(within) / len(within))) if within else 0.0
    preliminary = SourceInferenceFit(tuple(groups), overlap, 1.0,
                                     tuple(tuple(row) for row in correlations), threshold, df)
    discount = fit_discount(tuple(ForecastRecord((preliminary.structural_log_odds(row.log_odds),),
                                                row.outcome) for row in training))
    return SourceInferenceFit(preliminary.group_ids, overlap, discount,
                              preliminary.correlations, threshold, df)
