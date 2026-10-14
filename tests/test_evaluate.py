"""Analytic scoring, fitting, and leakage regression checks."""
import math
import unittest

from shared_evidence.evaluate import (
    ForecastRecord, brier_score, evaluate, evaluate_train_test, fit_discount,
    log_loss, paired_score_differences, prediction_logits,
)
from shared_evidence.model import logistic


class EvaluationTests(unittest.TestCase):
    def test_proper_scores_at_prior(self) -> None:
        for outcome in (0, 1):
            self.assertAlmostEqual(log_loss(0, outcome), math.log(2))
            self.assertEqual(brier_score(0, outcome), 0.25)

    def test_population_losses_minimized_at_true_probability(self) -> None:
        truth = 0.7
        true_logit = math.log(truth / (1 - truth))
        for loss in (log_loss, brier_score):
            optimum = truth * loss(true_logit, 1) + (1 - truth) * loss(true_logit, 0)
            for other in (-3, 0, 3):
                alternative = truth * loss(other, 1) + (1 - truth) * loss(other, 0)
                self.assertLess(optimum, alternative)

    def test_extreme_logits_without_clipping(self) -> None:
        self.assertEqual(log_loss(1000, 0), 1000)
        self.assertEqual(log_loss(-1000, 1), 1000)
        self.assertEqual(log_loss(1000, 1), 0)
        self.assertEqual(brier_score(-1000, 1), 1)
        for values, expected in [((1000, 1000), 1000), ((-1000, -1000), -1000), ((1000, -1000), 0)]:
            logits = prediction_logits(values, discount=0.5, overlap=1)
            self.assertAlmostEqual(logits["arithmetic_mean"], expected)

    def test_arithmetic_mean(self) -> None:
        values = (-2.3, 0.9, 1.6)
        predictions = prediction_logits(values, discount=0.3, overlap=0.2)
        self.assertAlmostEqual(logistic(predictions["arithmetic_mean"]), sum(map(logistic, values)) / 3)
        self.assertEqual(predictions["individual"], values[0])

    def test_oracle_boundaries(self) -> None:
        for overlap, expected in [(0, 9), (1, 3)]:
            logits = prediction_logits((3, 3, 3), discount=1, overlap=overlap)
            self.assertEqual(logits["oracle"], expected)

    def test_discount_analytic_interior_and_boundaries(self) -> None:
        training = [ForecastRecord((2,), y) for y in (1, 1, 1, 0)]
        self.assertAlmostEqual(fit_discount(training), math.log(3) / 2)
        self.assertEqual(fit_discount([ForecastRecord((2,), 1)]), 1)
        self.assertEqual(fit_discount([ForecastRecord((2,), 0)]), 0)
        self.assertEqual(fit_discount([ForecastRecord((0,), 1)]), 0)

    def test_test_labels_do_not_affect_fit(self) -> None:
        train = [ForecastRecord((2,), y) for y in (1, 1, 1, 0)]
        positive = evaluate_train_test(train, [ForecastRecord((4,), 1)], overlap=0.5)
        negative = evaluate_train_test(train, [ForecastRecord((4,), 0)], overlap=0.5)
        self.assertEqual(positive.discount, negative.discount)
        self.assertNotEqual(positive.scores["learned"], negative.scores["learned"])

    def test_paired_scores(self) -> None:
        result = evaluate([ForecastRecord((2,), 1), ForecastRecord((-1,), 0)], discount=0, overlap=0)
        pairs = paired_score_differences(result, "individual", "prior")
        self.assertTrue(all(row.log_loss < 0 and row.brier < 0 for row in pairs))
        self.assertAlmostEqual(sum(row.log_loss for row in pairs) / 2,
                               result.scores["individual"].log_loss - result.scores["prior"].log_loss)
        self.assertEqual(result.scores["prior"], result.scores["learned"])

    def test_invalid_inputs(self) -> None:
        for values, outcome in [((), 1), ((math.nan,), 0), ((1,), 2)]:
            with self.assertRaises(ValueError):
                ForecastRecord(values, outcome)
        with self.assertRaises(ValueError):
            fit_discount([])
        with self.assertRaises(ValueError):
            evaluate([], discount=1, overlap=0)
        with self.assertRaises(ValueError):
            prediction_logits((1,), discount=-1, overlap=0)
        with self.assertRaises(ValueError):
            log_loss(math.inf, 1)


if __name__ == "__main__":
    unittest.main()
