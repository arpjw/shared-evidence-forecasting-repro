"""Deterministic question ledgers with byte integrity and simulator replay.

A digest proves integrity relative to its receipt, not authenticity. Retain the
returned file SHA256 in the experiment summary. Gzip mtime and filename are
fixed; compression bytes may still differ across zlib versions. Replay uses
recorded streams and never estimates a parameter from held-out outcomes.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluate import brier_score, log_loss, prediction_logits
from .groups import grouped_oracle_log_odds
from .model import logistic
from .simulate import Question, SimulationConfig, simulate_question

SCHEMA_VERSION = 1


def _line(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def question_record(
    question: Question, *, stream: str,
    predictions: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Serialize a question and optional fixed predictions, without fitting."""
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string")
    record = {"type": "question", "stream": stream, **asdict(question)}
    if predictions is not None:
        if not predictions or any(not isinstance(name, str) or not name
                                  for name in predictions):
            raise ValueError("predictions must have nonempty method names")
        record["predictions"] = {
            name: {"logit": value, "probability": logistic(value),
                   "log_loss": log_loss(value, question.outcome),
                   "brier": brier_score(value, question.outcome)}
            for name, value in predictions.items()
        }
    # Normalize tuples to JSON lists and reject nonfinite numbers immediately.
    return json.loads(_line(record))


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for key in ("config", "source_sha256", "streams"):
        if not isinstance(metadata.get(key), dict):
            raise ValueError(f"metadata {key} must be an object")
    if not isinstance(metadata.get("phase"), str) or not metadata["phase"]:
        raise ValueError("metadata phase must be nonempty")
    if metadata["config"].get("phase") != metadata["phase"]:
        raise ValueError("metadata phase must match config phase")
    for name, digest in metadata["source_sha256"].items():
        if (not isinstance(name, str) or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("source_sha256 must contain SHA256 hex digests")
    for name, stream in metadata["streams"].items():
        if not isinstance(name, str) or not name or not isinstance(stream, dict):
            raise ValueError("invalid stream descriptor")
        SimulationConfig(**stream["simulation"])
        for key in ("seed", "count", "start"):
            value = stream.get(key, 0) if key == "start" else stream.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"stream {key} must be a nonnegative integer")
        if stream.get("role") not in ("train", "test"):
            raise ValueError("stream role must be train or test")
        if not isinstance(stream.get("split"), str) or not stream["split"]:
            raise ValueError("stream split must be nonempty")
    # The same seed and split must never occur in both training and test,
    # even under different conditions or different question ranges.
    roles: dict[tuple[int, str], str] = {}
    for stream in metadata["streams"].values():
        key = (stream["seed"], stream["split"])
        if key in roles and roles[key] != stream["role"]:
            raise ValueError("training and test streams overlap")
        roles[key] = stream["role"]


def write_ledger(
    path: str | Path, *, metadata: dict[str, Any],
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically write canonical JSONL.gz; failed writes preserve prior output.

    Metadata contains config, phase, source_sha256 and streams. A stream has
    simulation (SimulationConfig fields), seed, split, role, count and optional
    start. Records must be in stream order and ascending question ID per stream.
    This function streams records and does not hold the ledger in memory.
    """
    metadata = json.loads(_line(metadata))
    _validate_metadata(metadata)
    metadata.setdefault("python_version", platform.python_version())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    counts = dict.fromkeys(metadata["streams"], 0)
    count = 0
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=temporary,
                               compresslevel=6, mtime=0) as compressed:
                header = _line({"type": "header", "schema_version": SCHEMA_VERSION,
                                "metadata": metadata})
                compressed.write(header)
                digest.update(header)
                for record in records:
                    _validate_record_position(record, metadata, counts)
                    data = _line(record)
                    compressed.write(data)
                    digest.update(data)
                    count += 1
                _validate_counts(metadata, counts)
                compressed.write(_line({"type": "trailer", "count": count,
                                        "payload_sha256": digest.hexdigest()}))
            temporary.flush()
            os.fsync(temporary.fileno())
        file_digest = _file_digest(temporary_path)
        size = temporary_path.stat().st_size
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {"schema_version": SCHEMA_VERSION, "count": count,
            "payload_sha256": digest.hexdigest(), "sha256": file_digest,
            "bytes": size}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_record_position(record: dict[str, Any], metadata: dict[str, Any],
                              counts: dict[str, int]) -> None:
    stream_id = record.get("stream")
    if record.get("type") != "question" or stream_id not in counts:
        raise ValueError("invalid question record or unknown stream")
    stream = metadata["streams"][stream_id]
    question_id = record.get("question_id")
    expected = stream.get("start", 0) + counts[stream_id]
    if (isinstance(question_id, bool) or not isinstance(question_id, int)
            or question_id != expected):
        raise ValueError("duplicate, missing or reordered question ID")
    counts[stream_id] += 1
    if counts[stream_id] > stream["count"]:
        raise ValueError("too many questions in stream")


def _validate_counts(metadata: dict[str, Any], counts: dict[str, int]) -> None:
    if any(counts[name] != stream["count"]
           for name, stream in metadata["streams"].items()):
        raise ValueError("ledger does not cover declared streams")


def read_ledger(path: str | Path, *, expected_sha256: str | None = None
                ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Verify exact serialized bytes and coverage before returning any records."""
    path = Path(path)
    if expected_sha256 is not None and _file_digest(path) != expected_sha256:
        raise ValueError("compressed file SHA256 mismatch")
    digest = hashlib.sha256()
    records = []
    with gzip.open(path, "rb") as file:
        raw = file.readline()
        header = json.loads(raw)
        if header.get("type") != "header" or header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported ledger header")
        metadata = header["metadata"]
        _validate_metadata(metadata)
        counts = dict.fromkeys(metadata["streams"], 0)
        digest.update(raw)
        for raw in file:
            value = json.loads(raw)
            if value.get("type") == "trailer":
                if (value.get("count") != len(records)
                        or value.get("payload_sha256") != digest.hexdigest()):
                    raise ValueError("payload SHA256 or count mismatch")
                if file.read(1):
                    raise ValueError("unexpected data after ledger trailer")
                _validate_counts(metadata, counts)
                return metadata, tuple(records)
            _validate_record_position(value, metadata, counts)
            digest.update(raw)
            records.append(value)
    raise ValueError("missing ledger trailer")


def replay_ledger(path: str | Path, *, expected_sha256: str | None = None,
                  strict_sources: bool = False) -> dict[str, int]:
    """Recreate questions from seeds and independently recompute stored scores.

    This checks recorded predictions and scoring, not how parameters were
    fitted. The experiment must separately replay training-only fitted values.
    Exact simulator equality requires the recorded Python/random implementation.
    """
    metadata, records = read_ledger(path, expected_sha256=expected_sha256)
    if strict_sources:
        source_dir = Path(__file__).resolve().parent
        for name, digest in metadata["source_sha256"].items():
            source = source_dir / name
            if Path(name).name != name or not source.is_file() or _file_digest(source) != digest:
                raise ValueError("current source SHA256 mismatch")
    predictions_checked = 0
    for record in records:
        stream = metadata["streams"][record["stream"]]
        question = simulate_question(SimulationConfig(**stream["simulation"]),
                                     seed=stream["seed"], split=stream["split"],
                                     question_id=record["question_id"])
        expected = question_record(question, stream=record["stream"])
        if {key: record.get(key) for key in expected} != expected:
            raise ValueError("simulator replay mismatch")
        stored_predictions = record.get("predictions", {})
        if stored_predictions:
            if "discount" not in stream:
                raise ValueError("prediction replay requires stream discount")
            logits = prediction_logits(question.log_odds, discount=stream["discount"],
                                       overlap=stream["simulation"]["overlap"])
            if stream["simulation"].get("group_ids") is not None:
                logits["exchangeable_formula"] = logits.pop("oracle")
                logits["grouped_oracle"] = grouped_oracle_log_odds(
                    question.log_odds, question.group_ids, stream["simulation"]["overlap"])
            for name, prediction in stored_predictions.items():
                if name not in logits or prediction["logit"] != logits[name]:
                    raise ValueError("prediction logit replay mismatch")
        for prediction in stored_predictions.values():
            value = prediction["logit"]
            expected_scores = {"probability": logistic(value),
                               "log_loss": log_loss(value, question.outcome),
                               "brier": brier_score(value, question.outcome)}
            if any(not math.isfinite(prediction[key]) or prediction[key] != score
                   for key, score in expected_scores.items()):
                raise ValueError("prediction score replay mismatch")
            predictions_checked += 1
    return {"questions_checked": len(records), "predictions_checked": predictions_checked}
