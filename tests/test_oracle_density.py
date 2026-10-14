"""Independent dense-Gaussian check of the closed-form group posterior."""

import unittest

from shared_evidence.groups import grouped_oracle_log_odds
from shared_evidence.simulate import SimulationConfig, simulate_questions


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small positive-definite system by pivoted elimination in tests."""
    size = len(rhs)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= scale
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [row[-1] for row in augmented]


class OracleDensityTests(unittest.TestCase):
    def test_group_formula_matches_dense_likelihood_ratio(self) -> None:
        for groups, overlap in [
            ((0, 0, 0, 1, 1), 0.4),
            ((0, 0, 1, 2, 2), 0.85),
            ((0, 1, 2, 3, 4), 0.7),
        ]:
            config = SimulationConfig(len(groups), 0.7, overlap, groups)
            covariance = [
                [1.0 if i == j else overlap if groups[i] == groups[j] else 0.0
                 for j in range(len(groups))]
                for i in range(len(groups))
            ]
            for question in simulate_questions(config, 12, seed=91, split="density-check"):
                weights = _solve(covariance, [1.0] * len(groups))
                dense_log_odds = 2 * config.signal_strength * sum(
                    weight * signal for weight, signal in zip(weights, question.signals)
                )
                grouped_log_odds = grouped_oracle_log_odds(
                    question.log_odds, question.group_ids, overlap
                )
                self.assertAlmostEqual(grouped_log_odds, dense_log_odds, places=11)


if __name__ == "__main__":
    unittest.main()
