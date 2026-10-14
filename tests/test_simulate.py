"""Deterministic and distributional checks on synthetic questions."""

import math
import unittest

from shared_evidence.model import logistic
from shared_evidence.simulate import SimulationConfig, derive_seed, simulate_question, simulate_questions


class SimulationTests(unittest.TestCase):
    def test_replay_and_sharding(self) -> None:
        config = SimulationConfig(4, 0.7, 0.4)
        batch = simulate_questions(config, 20, seed=42, split="test")
        self.assertEqual(batch, simulate_questions(config, 20, seed=42, split="test"))
        self.assertEqual(batch[8:], simulate_questions(config, 12, seed=42, split="test", start=8))
        self.assertEqual(batch[9], simulate_question(config, seed=42, split="test", question_id=9))
        self.assertNotEqual(batch, simulate_questions(config, 20, seed=42, split="train"))
        self.assertNotEqual(derive_seed(1, "1"), derive_seed(1, 1))

    def test_boundaries_and_group_semantics(self) -> None:
        independent = simulate_questions(SimulationConfig(4, 0.7, 0), 10, seed=5)
        grouped = simulate_questions(SimulationConfig(4, 0.7, 0, (0, 1, 0, 1)), 10, seed=5)
        self.assertEqual([q.signals for q in independent], [q.signals for q in grouped])
        for q in simulate_questions(SimulationConfig(4, 0.7, 1), 10, seed=5):
            self.assertEqual(len(set(q.signals)), 1)
        for q in simulate_questions(SimulationConfig(4, 0.7, 1, (0, 1, 0, 1)), 10, seed=5):
            self.assertEqual(q.signals[0], q.signals[2])
            self.assertEqual(q.signals[1], q.signals[3])
            self.assertNotEqual(q.signals[0], q.signals[1])
            self.assertEqual(q.log_odds, tuple(1.4 * s for s in q.signals))
        self.assertTrue(all(x == 0 for x in simulate_question(SimulationConfig(3, 0, 0.5), seed=9).log_odds))

    def test_calibration_and_conditional_covariance(self) -> None:
        # Only one forecast per question enters calibration bins: correlated
        # forecasts within a question are not independent Bernoulli trials.
        config = SimulationConfig(3, 0.8, 0.6, (0, 0, 1))
        questions = simulate_questions(config, 30000, seed=8123, split="distribution-test")
        bins = [[] for _ in range(10)]
        products = [0.0, 0.0, 0.0]
        for q in questions:
            p = logistic(q.log_odds[0])
            bins[min(int(10 * p), 9)].append((q.outcome, p))
            noise = [s - 0.8 * (2 * q.outcome - 1) for s in q.signals]
            products[0] += noise[0] ** 2
            products[1] += noise[0] * noise[1]
            products[2] += noise[0] * noise[2]
        for bucket in bins:
            self.assertGreater(len(bucket), 200)
            residual = sum(y - p for y, p in bucket)
            variance = sum(p * (1 - p) for _, p in bucket)
            self.assertLess(abs(residual), 5 * math.sqrt(variance))
        for actual, expected in zip(products, (1, 0.6, 0)):
            self.assertAlmostEqual(actual / len(questions), expected, delta=0.03)

    def test_invalid_configuration(self) -> None:
        invalid = [(0, 1, 0), (True, 1, 0), (2.0, 1, 0), (2, -1, 0),
                   (2, math.nan, 0), (2, 1, -0.1), (2, 1, 1.1), (2, 1, math.inf)]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                SimulationConfig(*args)
        for groups in [(0,), (0, None), (0, True), "ab"]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                SimulationConfig(2, 1, 0.5, groups)
        config = SimulationConfig(2, 1, 0.5)
        for kwargs in ({"count": -1, "seed": 0}, {"count": 0, "seed": -1},
                       {"count": 0, "seed": 0, "split": ""},
                       {"count": 1, "seed": 0, "start": 0.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate_questions(config, **kwargs)


if __name__ == "__main__":
    unittest.main()
