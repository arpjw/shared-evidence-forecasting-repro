"""Byte integrity, replay and failure-atomicity checks for question ledgers."""
import copy
import gzip
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from shared_evidence.evaluate import prediction_logits
from shared_evidence.groups import grouped_oracle_log_odds
from shared_evidence.ledger import question_record, read_ledger, replay_ledger, write_ledger
from shared_evidence.simulate import SimulationConfig, simulate_questions


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "questions.jsonl.gz"
        self.config = SimulationConfig(3, 0.5, 0.7)
        self.metadata = {
            "phase": "ledger-test", "config": {"phase": "ledger-test"},
            "source_sha256": {}, "streams": {
                "test": {"simulation": asdict(self.config), "seed": 123,
                         "split": "ledger-test:test-0", "role": "test",
                         "count": 4, "start": 8, "discount": 0.4}}}
        questions = simulate_questions(self.config, 4, seed=123,
                                       split="ledger-test:test-0", start=8)
        self.records = [question_record(q, stream="test", predictions=prediction_logits(
            q.log_odds, discount=0.4, overlap=0.7)) for q in questions]

    def write(self, records=None, metadata=None):
        return write_ledger(self.path, metadata=metadata or self.metadata,
                            records=self.records if records is None else records)

    def test_deterministic_bytes_roundtrip_and_replay(self):
        receipt = self.write()
        first = self.path.read_bytes()
        other = self.path.with_name("renamed.gz")
        other_receipt = write_ledger(other, metadata=self.metadata, records=iter(self.records))
        self.assertEqual(first, other.read_bytes())
        self.assertEqual(receipt, other_receipt)
        metadata, records = read_ledger(self.path, expected_sha256=receipt["sha256"])
        self.assertEqual(records, tuple(self.records))
        self.assertEqual(metadata["phase"], "ledger-test")
        self.assertEqual(replay_ledger(self.path),
                         {"questions_checked": 4, "predictions_checked": 24})

    def test_content_corruption_and_external_digest(self):
        receipt = self.write()
        raw = gzip.decompress(self.path.read_bytes()).replace(b'"outcome":0', b'"outcome":1', 1)
        self.path.write_bytes(gzip.compress(raw, mtime=0))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_ledger(self.path, expected_sha256=receipt["sha256"])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_ledger(self.path)

    def test_replay_catches_validly_rehashed_wrong_data(self):
        records = copy.deepcopy(self.records)
        records[0]["signals"][0] += 0.1
        self.write(records)
        with self.assertRaisesRegex(ValueError, "simulator"):
            replay_ledger(self.path)
        records = copy.deepcopy(self.records)
        records[0]["predictions"]["learned"]["logit"] += 0.1
        self.write(records)
        with self.assertRaisesRegex(ValueError, "logit"):
            replay_ledger(self.path)
        records = copy.deepcopy(self.records)
        records[0]["predictions"]["learned"]["brier"] += 0.1
        self.write(records)
        with self.assertRaisesRegex(ValueError, "score"):
            replay_ledger(self.path)

    def test_failed_write_preserves_previous_file_and_removes_temp(self):
        self.write()
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.write(self.records[:-1])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        with self.assertRaises(ValueError):
            self.write([self.records[0], self.records[0]])

    def test_train_test_overlap_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["streams"]["train"] = dict(metadata["streams"]["test"], role="train")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.write(metadata=metadata)

    def test_source_hash_check_is_explicit(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["source_sha256"] = {"simulate.py": "0" * 64}
        self.write(metadata=metadata)
        replay_ledger(self.path)
        with self.assertRaisesRegex(ValueError, "source SHA256"):
            replay_ledger(self.path, strict_sources=True)

    def test_grouped_predictions_and_separate_training_stream(self):
        config = SimulationConfig(3, 0.5, 1.0, (0, 0, 1))
        metadata = copy.deepcopy(self.metadata)
        metadata["streams"]["test"]["simulation"] = asdict(config)
        metadata["streams"]["train"] = dict(metadata["streams"]["test"],
                                            split="ledger-test:train-0", role="train")
        records = []
        for stream_id, stream in metadata["streams"].items():
            for q in simulate_questions(config, 4, seed=123, split=stream["split"], start=8):
                logits = None
                if stream_id == "test":
                    logits = prediction_logits(q.log_odds, discount=0.4, overlap=1.0)
                    logits["exchangeable_formula"] = logits.pop("oracle")
                    logits["grouped_oracle"] = grouped_oracle_log_odds(q.log_odds, q.group_ids, 1.0)
                records.append(question_record(q, stream=stream_id, predictions=logits))
        self.write(records=records, metadata=metadata)
        self.assertEqual(replay_ledger(self.path),
                         {"questions_checked": 8, "predictions_checked": 28})

    def test_missing_trailer_rejected(self):
        self.write()
        raw = gzip.decompress(self.path.read_bytes())
        self.path.write_bytes(gzip.compress(b"\n".join(raw.splitlines()[:-1]) + b"\n", mtime=0))
        with self.assertRaisesRegex(ValueError, "trailer"):
            read_ledger(self.path)


if __name__ == "__main__":
    unittest.main()
