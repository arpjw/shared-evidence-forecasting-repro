"""Small checks for the model's exact boundary cases."""

import math
import unittest

from shared_evidence import aggregate_log_odds, effective_forecasters, logit


class ModelTests(unittest.TestCase):
    def test_effective_count_boundaries(self) -> None:
        self.assertEqual(effective_forecasters(10, 0.0), 10.0)
        self.assertEqual(effective_forecasters(10, 1.0), 1.0)
        self.assertAlmostEqual(effective_forecasters(10, 0.5), 10 / 5.5)

    def test_independent_log_odds_add(self) -> None:
        self.assertAlmostEqual(
            aggregate_log_odds([logit(0.6), logit(0.1)], 0.0),
            1 / 7,
        )

    def test_identical_signals_count_once(self) -> None:
        self.assertAlmostEqual(
            aggregate_log_odds([logit(0.6)] * 9, 1.0),
            0.6,
        )

    def test_repeated_source_changes_worked_example(self) -> None:
        repeated = [logit(0.6)] * 9
        independent_dissent = logit(0.1)
        naive = aggregate_log_odds(repeated + [independent_dissent], 0.0)
        source_aware = aggregate_log_odds([repeated[0], independent_dissent], 0.0)
        self.assertGreater(naive, 0.8)
        self.assertAlmostEqual(source_aware, 1 / 7)

    def test_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            effective_forecasters(0, 0.5)
        with self.assertRaises(ValueError):
            aggregate_log_odds([math.nan], 0.5)
        with self.assertRaises(ValueError):
            logit(1.0)


if __name__ == "__main__":
    unittest.main()
