"""End-to-end checks for deterministic replay and source-aware scoring."""

import unittest

from shared_evidence.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_replay_and_group_oracle(self) -> None:
        config = {
            "schema_version": 1,
            "phase": "test-only",
            "seed": 17,
            "train_count": 30,
            "test_count": 80,
            "replicates": 1,
            "conditions": [
                {"name": "grouped", "n_forecasters": 3, "signal_strength": 0.7,
                 "overlap": 1.0, "group_ids": [0, 0, 1]}
            ],
        }
        first = run_experiment(config)
        self.assertEqual(first, run_experiment(config))
        scores = first["runs"][0]["scores"]
        self.assertIn("grouped_oracle", scores)
        self.assertNotIn("oracle", scores)
        self.assertNotEqual(scores["grouped_oracle"], scores["exchangeable_formula"])
        config["seed"] = 999
        config["conditions"][0]["name"] = "changed"
        self.assertEqual(first["config"]["seed"], 17)
        self.assertEqual(first["config"]["conditions"][0]["name"], "grouped")

    def test_phase_separates_question_streams(self) -> None:
        config = {
            "schema_version": 1,
            "phase": "first",
            "seed": 17,
            "train_count": 20,
            "test_count": 50,
            "replicates": 1,
            "conditions": [
                {"name": "ordinary", "n_forecasters": 2,
                 "signal_strength": 0.7, "overlap": 0.3}
            ],
        }
        first = run_experiment(config)
        config["phase"] = "second"
        second = run_experiment(config)
        self.assertNotEqual(first["runs"], second["runs"])


if __name__ == "__main__":
    unittest.main()
