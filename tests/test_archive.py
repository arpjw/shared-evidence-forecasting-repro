"""Integration check for summary to question-ledger preservation."""

import tempfile
import unittest
import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

from shared_evidence.archive import write_experiment_ledger
from shared_evidence.experiment import run_experiment
from shared_evidence.ledger import read_ledger, replay_ledger, write_ledger
from shared_evidence.verify import verify_summary_and_ledger
import json


class ArchiveTests(unittest.TestCase):
    def test_rejects_stream_descriptor_and_empty_source_hashes(self) -> None:
        config = {
            "schema_version": 1, "phase": "descriptor-test", "seed": 14,
            "train_count": 4, "test_count": 5, "replicates": 1,
            "conditions": [{"name": "shared", "n_forecasters": 3,
                            "signal_strength": 0.6, "overlap": 0.8}],
        }
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl.gz"
            result["ledger_receipt"] = write_experiment_ledger(result, ledger)
            metadata, records = read_ledger(ledger)
            summary = Path(directory) / "summary.json"

            claimed = copy.deepcopy(result)
            claimed["config"]["seed"] = 999
            claimed["config"]["train_count"] = 9000
            claimed_metadata = copy.deepcopy(metadata)
            claimed_metadata["config"] = claimed["config"]
            claimed["ledger_receipt"] = write_ledger(
                ledger, metadata=claimed_metadata, records=records
            )
            summary.write_text(json.dumps(claimed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stream descriptor"):
                verify_summary_and_ledger(summary, ledger)

            empty = copy.deepcopy(result)
            empty["source_sha256"] = {}
            metadata["source_sha256"] = {}
            empty["ledger_receipt"] = write_ledger(
                ledger, metadata=metadata, records=records
            )
            summary.write_text(json.dumps(empty), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash coverage"):
                verify_summary_and_ledger(summary, ledger)

    def test_locked_summary_must_match_frozen_config(self) -> None:
        config = {
            "schema_version": 1, "phase": "audit-only", "seed": 14,
            "train_count": 4, "test_count": 5, "replicates": 1,
            "conditions": [{"name": "shared", "n_forecasters": 3,
                            "signal_strength": 0.6, "overlap": 0.8}],
        }
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.jsonl.gz"
            result["ledger_receipt"] = write_experiment_ledger(result, ledger)
            result["status"] = "locked_synthetic_evaluation"
            result["manifest_sha256"] = "frozen-digest"
            summary = Path(directory) / "summary.json"
            summary.write_text(json.dumps(result), encoding="utf-8")
            production = Path(__file__).resolve().parent.parent / "configs/production-primary-v1.json"
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({
                "config_path": "configs/production-primary-v1.json",
                "config_sha256": hashlib.sha256(production.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            with patch("shared_evidence.verify.verify_manifest", return_value="frozen-digest"):
                with self.assertRaisesRegex(ValueError, "summary config differs"):
                    verify_summary_and_ledger(summary, ledger, manifest_path=manifest)

    def test_training_and_test_records_replay(self) -> None:
        config = {
            "schema_version": 1,
            "phase": "archive-test",
            "seed": 14,
            "train_count": 12,
            "test_count": 20,
            "replicates": 1,
            "conditions": [
                {"name": "shared", "n_forecasters": 3,
                 "signal_strength": 0.6, "overlap": 0.8,
                 "group_ids": ["a", "a", "b"]}
            ],
        }
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl.gz"
            receipt = write_experiment_ledger(result, path)
            self.assertEqual(receipt["count"], 32)
            _, records = read_ledger(path, expected_sha256=receipt["sha256"])
            self.assertEqual(len(records), 32)
            self.assertNotIn("predictions", records[0])
            self.assertIn("grouped_oracle", records[-1]["predictions"])
            self.assertEqual(
                replay_ledger(path, expected_sha256=receipt["sha256"], strict_sources=True),
                {"questions_checked": 32, "predictions_checked": 140},
            )
            result["ledger_receipt"] = receipt
            summary = Path(directory) / "summary.json"
            summary.write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(
                verify_summary_and_ledger(summary, path),
                {"questions_checked": 32, "predictions_checked": 140,
                 "runs_verified": 1},
            )
            changed = json.loads(summary.read_text(encoding="utf-8"))
            changed["runs"][0]["discount"] += 0.1
            summary.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stream descriptor|discount"):
                verify_summary_and_ledger(summary, path)
            changed["runs"][0]["discount"] = result["runs"][0]["discount"]
            changed["runs"][0]["scores"]["learned"]["log_loss"] += 0.1
            summary.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from replay"):
                verify_summary_and_ledger(summary, path)
            changed = json.loads(json.dumps(result))
            changed["runs"].clear()
            summary.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run coverage"):
                verify_summary_and_ledger(summary, path)


if __name__ == "__main__":
    unittest.main()
