"""Check a fitted discount against a known population parameter."""

import math
import unittest

from shared_evidence.evaluate import fit_discount
from shared_evidence.model import logistic
from shared_evidence.simulate import SimulationConfig, simulate_questions


class PopulationDiscountTests(unittest.TestCase):
    def test_learned_discount_approaches_exact_overlap_factor(self) -> None:
        config = SimulationConfig(n_forecasters=10, signal_strength=0.5, overlap=0.5)
        training = simulate_questions(
            config, 30_000, seed=38_211, split="population-beta-validation"
        )
        fitted = fit_discount(training)
        exact = 1.0 / (1.0 + 9 * config.overlap)
        contributions = []
        for question in training:
            total = sum(question.log_odds)
            probability = logistic(exact * total)
            contributions.append(probability * (1 - probability) * total * total)
        information = math.fsum(contributions)
        # Six observed-information standard errors accommodates finite Monte
        # Carlo variation while still detecting a materially wrong discount.
        self.assertAlmostEqual(fitted, exact, delta=6 / math.sqrt(information))


if __name__ == "__main__":
    unittest.main()
