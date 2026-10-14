"""Checks that analysis pairs fitted replicates and refuses incomplete cells."""
import copy
import unittest

from shared_evidence.source_inference_analysis import analyze_result


def example_result():
    config = {"phase": "source-inference-sensitivity-test", "train_count": 20,
              "test_count": 10, "replicates": 3, "bootstrap_seed": 19,
              "bootstrap_resamples": 100, "interval_level": 0.95,
              "conditions": [{"name": "test"}]}
    runs = []
    for rep in range(3):
        methods = {name: {"log_loss": value + rep * 0.01, "brier": value / 2 + rep * 0.005}
                   for name, value in (("prior", 0.69), ("arithmetic_mean", 0.60),
                                       ("independent", 0.80), ("scalar", 0.55),
                                       ("inferred_groups", 0.50), ("grouped_oracle", 0.49))}
        runs.append({"condition": "test", "replicate": rep, "scores": methods,
                     "diagnostics": {"group_count": 2, "overlap": 0.8,
                                     "exact_recovery": True, "pair_error": 0.0},
                     "sign_counts": {"scalar_opposite": 2, "inferred_opposite": 0,
                                     "scalar_zero": 0, "inferred_zero": 0, "oracle_zero": 0,
                                     "scalar_opposite_repaired": 2}})
    return {"status": "locked_synthetic_evaluation", "config": config, "runs": runs}


class SourceInferenceAnalysisTests(unittest.TestCase):
    def test_paired_contrasts_ignore_shared_replicate_shift(self):
        result = example_result()
        first = analyze_result(result)
        second = analyze_result(copy.deepcopy(result))
        self.assertEqual(first, second)
        cell = first["conditions"]["test"]
        improvement = cell["contrasts"]["inferred_groups_minus_scalar"]["log_loss"]
        self.assertAlmostEqual(improvement["mean"], -0.05)
        self.assertAlmostEqual(improvement["bootstrap_interval"][0], -0.05)
        self.assertAlmostEqual(improvement["bootstrap_interval"][1], -0.05)
        self.assertEqual(cell["diagnostics"]["exact_recovery_fraction"], 1.0)
        self.assertEqual(cell["signs"]["repaired_fraction_among_scalar_opposites"], 1.0)

    def test_missing_or_duplicate_replicate_rejected(self):
        result = example_result()
        result["runs"][2]["replicate"] = 1
        with self.assertRaises(ValueError):
            analyze_result(result)

    def test_unidentifiable_recovery_remains_unscored(self):
        result = example_result()
        for run in result["runs"]:
            run["diagnostics"]["exact_recovery"] = None
            run["diagnostics"]["pair_error"] = None
        cell = analyze_result(result)["conditions"]["test"]
        self.assertIsNone(cell["diagnostics"]["exact_recovery_fraction"])
        self.assertIsNone(cell["diagnostics"]["mean_pair_error"])

    def test_small_positive_gap_is_reported_and_flagged(self):
        result = example_result()
        for run in result["runs"]:
            run["scores"]["scalar"]["log_loss"] = run["scores"]["grouped_oracle"]["log_loss"] + 0.0005
        cell = analyze_result(result)["conditions"]["test"]
        self.assertTrue(cell["gap_fraction_unstable"])
        self.assertFalse(cell["gap_fraction_undefined"])
        self.assertIsNotNone(cell["gap_fraction_closed"])


if __name__ == "__main__":
    unittest.main()
