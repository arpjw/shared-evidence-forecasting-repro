"""Controls for the training-only source learner, not performance acceptance gates."""
import math
import unittest

from shared_evidence.evaluate import ForecastRecord, fit_discount, log_loss
from shared_evidence.simulate import SimulationConfig, simulate_questions
from shared_evidence.source_inference import fit_source_inference


def records(rho=0.9, n=10, count=400, strength=0.5):
    config = SimulationConfig(n, strength, rho, (0,) * (n - 1) + (1,))
    return tuple(ForecastRecord(q.log_odds, q.outcome) for q in
                 simulate_questions(config, count, seed=93217013, split="development-unit-source"))


class SourceInferenceTests(unittest.TestCase):
    def test_recovers_strong_unequal_blocks(self):
        fit = fit_source_inference(records())
        self.assertEqual(fit.group_ids, (0,) * 9 + (1,))
        self.assertLess(abs(fit.overlap - 0.9), 0.04)
        self.assertEqual(fit.zero_variance_columns, ())
        # Scalar cannot change this total's positive sign; structure can.
        values = (0.4,) * 9 + (-2.0,)
        self.assertGreater(sum(values), 0)
        self.assertLess(fit.predict_log_odds(values), 0)

    def test_exact_duplicates_at_endpoint(self):
        fit = fit_source_inference(records(rho=1.0))
        self.assertEqual(fit.group_ids, (0,) * 9 + (1,))
        self.assertAlmostEqual(fit.overlap, 1.0)
        self.assertAlmostEqual(fit.weights[0], 1 / 9)
        self.assertEqual(fit.weights[-1], 1.0)

    def test_independent_columns_despite_common_outcome(self):
        rows = records(rho=0, count=1000, strength=3.0)
        fit = fit_source_inference(rows)
        self.assertEqual(len(set(fit.group_ids)), 10)
        self.assertEqual(fit.overlap, 0.0)
        self.assertEqual(fit.discount, fit_discount(rows))

    def test_zero_strength_does_not_identify_groups(self):
        fit = fit_source_inference(records(strength=0))
        self.assertEqual(fit.group_ids, tuple(range(10)))
        self.assertEqual(fit.discount, 0.0)
        self.assertEqual(fit.zero_variance_columns, tuple(range(10)))
        self.assertEqual(tuple(fit.correlations[i][i] for i in range(10)), (0.0,) * 10)
        self.assertTrue(all(value == 0.0 for row in fit.correlations for value in row))
        self.assertEqual(fit.predict_log_odds((0.0,) * 10), 0.0)

    def test_permutations_and_outcome_symmetry(self):
        rows = records()
        fit = fit_source_inference(rows)
        permutation = (9, 2, 0, 8, 1, 3, 4, 7, 6, 5)
        permuted = fit_source_inference(tuple(ForecastRecord(tuple(r.log_odds[i] for i in permutation), r.outcome)
                                             for r in rows))
        reversed_fit = fit_source_inference(tuple(ForecastRecord(tuple(-x for x in r.log_odds), 1-r.outcome)
                                                 for r in rows))
        for i, original in enumerate(permutation):
            self.assertAlmostEqual(permuted.weights[i], fit.weights[original])
        self.assertAlmostEqual(permuted.discount, fit.discount)
        self.assertAlmostEqual(reversed_fit.discount, fit.discount)
        self.assertEqual(reversed_fit.weights, fit.weights)

    def test_single_column_and_prediction_contract(self):
        fit = fit_source_inference(records(n=1))
        self.assertEqual(fit.weights, (1.0,))
        self.assertTrue(math.isfinite(fit.predict_log_odds((1000.0,))))
        for values in ((), (1.0, 2.0), (math.inf,)):
            with self.assertRaises(ValueError):
                fit.predict_log_odds(values)

    def test_connected_component_can_merge_a_noisy_bridge(self):
        # Orthogonal u/v: adjacent correlations are 1/sqrt(2), endpoint 0.
        # A threshold graph has one component even without endpoint support.
        count = 128
        rows = []
        for k in range(count):
            u = math.sqrt(2) * math.sin(2 * math.pi * k / count)
            v = math.sqrt(2) * math.cos(2 * math.pi * k / count)
            y = k % 2
            signed = (u, (u + v) / math.sqrt(2), v)
            rows.append(ForecastRecord(tuple((2*y-1)*x for x in signed), y))
        fit = fit_source_inference(rows)
        self.assertGreater(fit.correlations[0][1], fit.threshold)
        self.assertGreater(fit.correlations[1][2], fit.threshold)
        self.assertAlmostEqual(fit.correlations[0][2], 0.0)
        self.assertEqual(fit.group_ids, (0, 0, 0))
        # Pooled rho includes the unsupported endpoint pair, not just edges.
        self.assertAlmostEqual(fit.overlap, math.sqrt(2) / 3)

    def test_fixed_threshold_and_residual_degrees_of_freedom(self):
        fit = fit_source_inference(records(count=400))
        self.assertEqual(fit.residual_df, 399)
        self.assertAlmostEqual(fit.threshold, math.sqrt(2*math.log(1800)/399))
        tiny = fit_source_inference((ForecastRecord((1., 2.), 0),
                                     ForecastRecord((2., 0.), 1),
                                     ForecastRecord((3., 1.), 0)))
        self.assertEqual(tiny.threshold, 1.0)

    def test_equal_size_groups_are_representable_by_one_scalar(self):
        config = SimulationConfig(4, 0.5, 1.0, (0, 0, 1, 1))
        rows = tuple(ForecastRecord(q.log_odds, q.outcome) for q in
                     simulate_questions(config, 400, seed=93217013,
                                        split="development-unit-equal-groups"))
        fit = fit_source_inference(rows)
        self.assertEqual(fit.group_ids, (0, 0, 1, 1))
        for values in ((3., 3., -1., -1.), (-1., 0., 1., 3.)):
            scalar_equivalent = fit.discount * fit.weights[0]
            self.assertAlmostEqual(fit.predict_log_odds(values),
                                   scalar_equivalent * math.fsum(values))
        self.assertEqual(len(set(fit.weights)), 1)

    def test_extreme_finite_training_logits_keep_correlations_finite(self):
        rows = tuple(ForecastRecord(tuple(1e150*x for x in row.log_odds), row.outcome)
                     for row in records(count=100))
        fit = fit_source_inference(rows)
        self.assertTrue(all(math.isfinite(x) for row in fit.correlations for x in row))
        self.assertTrue(math.isfinite(fit.discount))
        self.assertTrue(math.isfinite(fit.predict_log_odds(rows[0].log_odds)))

    def test_heldout_outcomes_cannot_change_fitted_predictions(self):
        fit = fit_source_inference(records(count=100))
        snapshot = fit
        logits = ((0.4,)*9 + (-2.,), (-0.1,)*9 + (1.,))
        test_a = tuple(ForecastRecord(values, 0) for values in logits)
        test_b = tuple(ForecastRecord(values, 1) for values in logits)
        predictions_a = tuple(fit.predict_log_odds(row.log_odds) for row in test_a)
        losses_a = tuple(log_loss(x, row.outcome) for x, row in zip(predictions_a, test_a))
        predictions_b = tuple(fit.predict_log_odds(row.log_odds) for row in test_b)
        losses_b = tuple(log_loss(x, row.outcome) for x, row in zip(predictions_b, test_b))
        self.assertEqual(predictions_a, predictions_b)
        self.assertNotEqual(losses_a, losses_b)
        self.assertEqual(fit, snapshot)

    def test_invalid_training(self):
        for rows in ((), (ForecastRecord((0.0,), 0),) * 2,
                     (ForecastRecord((0.0,), 0), ForecastRecord((0.0, 1.0), 1))):
            with self.assertRaises(ValueError):
                fit_source_inference(rows)

    def test_deterministic_with_single_outcome_class(self):
        rows = tuple(ForecastRecord((x, x), 1) for x in (-1.0, 0.0, 1.0, 2.0))
        fit = fit_source_inference(rows)
        self.assertEqual(fit, fit_source_inference(rows))
        self.assertEqual(fit.group_ids, (0, 0))

    def test_negative_residual_correlation_not_a_shared_source(self):
        rows = tuple(ForecastRecord((x, -x), y) for y in (0, 1) for x in (-2.0, -1.0, 1.0, 2.0))
        fit = fit_source_inference(rows)
        self.assertEqual(fit.group_ids, (0, 1))
        self.assertAlmostEqual(fit.correlations[0][1], -1.0)


if __name__ == "__main__":
    unittest.main()
