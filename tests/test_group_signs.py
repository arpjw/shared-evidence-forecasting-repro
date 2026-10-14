"""Integration checks for the streaming group decision diagnostic."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.group_signs import analyze_group_signs
from shared_evidence.archive import write_experiment_ledger
from shared_evidence.experiment import run_experiment


class GroupSignsTests(unittest.TestCase):
    def test_grouped_ledger_counts_and_receipt(self) -> None:
        config = {
            "schema_version": 1, "phase": "group-sign-test", "seed": 438,
            "train_count": 8, "test_count": 100, "replicates": 2,
            "conditions": [{"name": "unequal", "n_forecasters": 4,
                            "signal_strength": 0.5, "overlap": 1.0,
                            "group_ids": ["a", "a", "a", "b"]}],
        }
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.jsonl.gz"
            result["ledger_receipt"] = write_experiment_ledger(result, ledger)
            summary = root / "summary.json"
            summary.write_text(json.dumps(result), encoding="utf-8")
            report = analyze_group_signs(summary, ledger)
            cell = report["cells"]["unequal"]
            self.assertEqual(cell["questions"], 200)
            self.assertEqual(sum(r["questions"] for r in cell["replicates"]), 200)
            self.assertGreater(cell["opposite_sign_count"], 0)
            self.assertLessEqual(cell["opposite_sign_fraction"], 1)
            result["ledger_receipt"]["sha256"] = "0" * 64
            summary.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                analyze_group_signs(summary, ledger)


if __name__ == "__main__":
    unittest.main()
