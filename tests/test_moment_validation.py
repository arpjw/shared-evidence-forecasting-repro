"""Distributional and symmetry checks independent of the scoring pipeline."""

import statistics
import unittest

from shared_evidence.evaluate import log_loss
from shared_evidence.groups import grouped_oracle_log_odds
from shared_evidence.model import aggregate_log_odds
from shared_evidence.simulate import SimulationConfig, simulate_questions


class MomentValidationTests(unittest.TestCase):
    def test_oracle_logit_moments_and_signal_means(self) -> None:
        config = SimulationConfig(10, 0.5, 0.5)
        questions = simulate_questions(
            config, 30_000, seed=5_908, split="oracle-moment-validation"
        )
        effective = config.n_forecasters / (1 + 9 * config.overlap)
        expected_mean = 2 * config.signal_strength**2 * effective
        expected_variance = 4 * config.signal_strength**2 * effective
        for outcome in (0, 1):
            subset = [q for q in questions if q.outcome == outcome]
            self.assertGreater(len(subset), 14_000)
            signals = [q.signals[0] for q in subset]
            logits = [
                sum(q.log_odds) / (1 + 9 * config.overlap) for q in subset
            ]
            # With about 15,000 independent questions per class, the mean
            # tolerances exceed roughly four standard errors; the variance
            # tolerance exceeds roughly four Gaussian sampling SDs.
            sign = 1 if outcome else -1
            self.assertAlmostEqual(statistics.mean(signals), sign * 0.5, delta=0.04)
            self.assertAlmostEqual(statistics.variance(signals), 1.0, delta=0.05)
            self.assertAlmostEqual(statistics.mean(logits), sign * expected_mean, delta=0.05)
            self.assertAlmostEqual(
                statistics.variance(logits), expected_variance, delta=0.09
            )

    def test_permutation_and_label_symmetry(self) -> None:
        values = [0.3, -1.1, 0.8, 0.4]
        groups = ["a", "a", "b", "c"]
        overlap = 0.7
        order = [3, 1, 0, 2]
        first = grouped_oracle_log_odds(values, groups, overlap)
        permuted = grouped_oracle_log_odds(
            [values[i] for i in order], [groups[i] for i in order], overlap
        )
        flipped = grouped_oracle_log_odds([-value for value in values], groups, overlap)
        self.assertAlmostEqual(first, permuted)
        self.assertAlmostEqual(flipped, -first)
        self.assertAlmostEqual(log_loss(first, 1), log_loss(-first, 0))
        self.assertAlmostEqual(
            aggregate_log_odds(values, overlap),
            aggregate_log_odds([values[i] for i in order], overlap),
        )


if __name__ == "__main__":
    unittest.main()
