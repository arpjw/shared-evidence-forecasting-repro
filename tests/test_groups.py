"""Boundary and source-provenance checks for independent evidence groups."""

import unittest

from shared_evidence.groups import grouped_oracle_log_odds, grouped_oracle_probability
from shared_evidence.model import logit


class GroupTests(unittest.TestCase):
    def test_one_group_matches_exchangeable_pooling(self) -> None:
        values = [logit(0.7), logit(0.6), logit(0.8)]
        expected = sum(values) / (1 + 2 * 0.4)
        self.assertAlmostEqual(grouped_oracle_log_odds(values, [0, 0, 0], 0.4), expected)

    def test_distinct_groups_keep_independent_evidence(self) -> None:
        values = [logit(0.6)] * 9 + [logit(0.1)]
        groups = ["shared"] * 9 + ["independent"]
        self.assertAlmostEqual(grouped_oracle_probability(values, groups, 1.0), 1 / 7)

    def test_zero_overlap_ignores_group_assignment(self) -> None:
        values = [0.1, -0.5, 1.2]
        self.assertAlmostEqual(
            grouped_oracle_log_odds(values, [0, 0, 1], 0.0), sum(values)
        )

    def test_mismatched_lengths_rejected(self) -> None:
        with self.assertRaises(ValueError):
            grouped_oracle_log_odds([0.2, 0.3], [0], 0.5)


if __name__ == "__main__":
    unittest.main()
