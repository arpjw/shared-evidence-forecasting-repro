"""Analytic summaries, pairing, deterministic resampling, and missing-cell gates."""
import copy
import math
import unittest

from shared_evidence.analysis import analyze_experiment, summarize_replicates


def fixture() -> dict:
    return {
        "config": {"seed": 42, "phase": "test", "replicates": 3, "conditions": [{"name": "case"}]},
        "runs": [
            {"condition": "case", "replicate": i, "scores": {
                method: {"log_loss": value + offset, "brier": (value + offset) / 10}
                for method, offset in (("learned", 0), ("independent", 2), ("arithmetic_mean", 1))
            }} for i, value in enumerate((1, 2, 3))
        ],
    }


class AnalysisTests(unittest.TestCase):
    def test_known_mean_sd_se(self) -> None:
        summary = summarize_replicates([1, 2, 3], seed=7, bootstrap_resamples=100)
        self.assertEqual(summary["mean"], 2)
        self.assertEqual(summary["sample_sd"], 1)
        self.assertAlmostEqual(summary["standard_error"], 1 / math.sqrt(3))

    def test_exact_two_point_percentile_bootstrap(self) -> None:
        # Bootstrap means of [0, 2] have support 0, 1, 2 and tail mass 1/4.
        summary = summarize_replicates([0, 2], seed=31, bootstrap_resamples=1000)
        self.assertEqual(summary["bootstrap_interval"], [0, 2])

    def test_pairing_retains_constant_difference(self) -> None:
        report = analyze_experiment(fixture(), bootstrap_resamples=100)
        contrast = report["conditions"]["case"]["contrasts"]["learned_minus_independent"]["log_loss"]
        self.assertEqual(contrast["mean"], -2)
        self.assertEqual(contrast["sample_sd"], 0)
        self.assertEqual(contrast["bootstrap_interval"], [-2, -2])
        self.assertEqual(report["confidence_level"], 0.975)

    def test_nonconstant_contrasts_share_bootstrap_indices(self) -> None:
        result = fixture()
        for run, difference in zip(result["runs"], (1, 2, 5)):
            for metric in ("log_loss", "brier"):
                run["scores"]["learned"][metric] = 20
                run["scores"]["independent"][metric] = 20 - difference
                run["scores"]["arithmetic_mean"][metric] = 20 - 2 * difference
        kwargs = {"bootstrap_resamples": 31, "confidence_level": 0.8}
        contrasts = (("learned", "independent"), ("learned", "arithmetic_mean"))
        first = analyze_experiment(result, contrasts=contrasts, **kwargs)
        reversed_order = analyze_experiment(result, contrasts=contrasts[::-1], **kwargs)
        self.assertEqual(first, reversed_order)
        summaries = first["conditions"]["case"]["contrasts"]
        for metric in ("log_loss", "brier"):
            left = summaries["learned_minus_independent"][metric]
            right = summaries["learned_minus_arithmetic_mean"][metric]
            self.assertGreater(left["sample_sd"], 0)
            for endpoint_left, endpoint_right in zip(left["bootstrap_interval"], right["bootstrap_interval"]):
                self.assertAlmostEqual(endpoint_right, 2 * endpoint_left)

    def test_replay_order_independence_and_no_mutation(self) -> None:
        result = fixture()
        original = copy.deepcopy(result)
        first = analyze_experiment(result, bootstrap_resamples=100)
        self.assertEqual(result, original)
        result["runs"].reverse()
        self.assertEqual(first, analyze_experiment(result, bootstrap_resamples=100))
        result["config"]["phase"] = "another"
        self.assertNotEqual(first["analysis_seed"], analyze_experiment(result, bootstrap_resamples=100)["analysis_seed"])

    def test_missing_duplicate_and_invalid_cells(self) -> None:
        variants = []
        missing = fixture(); missing["runs"].pop(); variants.append(missing)
        duplicate = fixture(); duplicate["runs"].append(copy.deepcopy(duplicate["runs"][0])); variants.append(duplicate)
        method = fixture(); del method["runs"][0]["scores"]["learned"]; variants.append(method)
        metric = fixture(); del metric["runs"][0]["scores"]["learned"]["brier"]; variants.append(metric)
        nonfinite = fixture(); nonfinite["runs"][0]["scores"]["learned"]["brier"] = math.nan; variants.append(nonfinite)
        short = fixture(); short["config"]["replicates"] = 1; variants.append(short)
        for value in variants:
            with self.assertRaises(ValueError):
                analyze_experiment(value, bootstrap_resamples=10)

    def test_retains_experiment_status_and_design(self) -> None:
        result = fixture()
        result["status"] = "development_diagnostic"
        result["config"].update(train_count=20, test_count=40)
        report = analyze_experiment(result, bootstrap_resamples=10)
        self.assertEqual(report["source_experiment"], {
            "status": "development_diagnostic", "phase": "test",
            "train_count": 20, "test_count": 40, "replicates": 3,
        })

    def test_bad_bootstrap_parameters(self) -> None:
        for kwargs in ({"bootstrap_resamples": 1}, {"confidence_level": 1}, {"seed": -1}):
            arguments = {"seed": 1, **kwargs}
            with self.assertRaises(ValueError):
                summarize_replicates([1, 2], **arguments)
        with self.assertRaises(ValueError):
            summarize_replicates([1], seed=1)


if __name__ == "__main__":
    unittest.main()
