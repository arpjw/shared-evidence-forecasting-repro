"""Connect experiment summaries to independently replayable question ledgers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluate import brier_score, log_loss, prediction_logits
from .groups import grouped_oracle_log_odds
from .ledger import question_record, write_ledger
from .simulate import SimulationConfig, simulate_questions


def write_experiment_ledger(
    result: dict[str, Any], path: str | Path
) -> dict[str, Any]:
    """Regenerate all streams from a completed result and save exact records.

    Training records include outcomes and individual logits so a separate
    verifier can refit discounts. Only test records include scored method
    predictions. This function replays, rather than reuses, run-time objects.
    """
    source_dir = Path(__file__).resolve().parent
    for name, digest in result["source_sha256"].items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed since experiment summary")
    config = result["config"]
    condition_by_name = {row["name"]: row for row in config["conditions"]}
    streams: dict[str, dict[str, Any]] = {}
    for run in result["runs"]:
        condition = run["condition"]
        replicate = run["replicate"]
        raw = condition_by_name[condition]
        simulation = SimulationConfig(
            raw["n_forecasters"], raw["signal_strength"], raw["overlap"],
            raw.get("group_ids"),
        )
        for role in ("train", "test"):
            stream_id = f"{condition}:r{replicate}:{role}"
            streams[stream_id] = {
                "simulation": asdict(simulation),
                "seed": config["seed"],
                "split": f"{config['phase']}:{role}-{replicate}",
                "role": role,
                "count": config[f"{role}_count"],
                "start": 0,
            }
            if role == "test":
                streams[stream_id]["discount"] = run["discount"]

    def records() -> Iterator[dict[str, Any]]:
        for run in result["runs"]:
            condition = run["condition"]
            replicate = run["replicate"]
            simulation = SimulationConfig(**streams[f"{condition}:r{replicate}:test"]["simulation"])
            for role in ("train", "test"):
                stream_id = f"{condition}:r{replicate}:{role}"
                stream = streams[stream_id]
                for question in simulate_questions(
                    simulation, stream["count"], seed=stream["seed"], split=stream["split"]
                ):
                    predictions = None
                    if role == "test":
                        predictions = prediction_logits(
                            question.log_odds, discount=run["discount"],
                            overlap=simulation.overlap,
                        )
                        if simulation.group_ids is not None:
                            predictions["exchangeable_formula"] = predictions.pop("oracle")
                            predictions["grouped_oracle"] = grouped_oracle_log_odds(
                                question.log_odds, question.group_ids, simulation.overlap
                            )
                    yield question_record(question, stream=stream_id, predictions=predictions)

    receipt = write_ledger(
        path,
        metadata={
            "phase": config["phase"],
            "config": config,
            "source_sha256": result["source_sha256"],
            "streams": streams,
        },
        records=records(),
    )
    for name, digest in result["source_sha256"].items():
        if hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed during ledger generation")
    return receipt
