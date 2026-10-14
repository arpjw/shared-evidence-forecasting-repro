"""Replayable development runs for the shared-evidence experiment.

Run configurations are explicit JSON. This runner reports descriptive scores;
it does not turn development runs into a frozen confirmatory result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from .evaluate import Scores, brier_score, evaluate_train_test, log_loss
from .groups import grouped_oracle_log_odds
from .simulate import SimulationConfig, simulate_questions

SOURCE_FILES = (
    "model.py", "simulate.py", "evaluate.py", "groups.py", "experiment.py",
    "ledger.py", "archive.py",
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _scores_dict(score: Scores) -> dict[str, float]:
    return {"log_loss": score.log_loss, "brier": score.brier}


def _grouped_oracle_scores(questions: tuple[Any, ...], overlap: float) -> Scores:
    logits = (
        grouped_oracle_log_odds(q.log_odds, q.group_ids, overlap)
        for q in questions
    )
    losses = [Scores(log_loss(x, q.outcome), brier_score(x, q.outcome))
              for x, q in zip(logits, questions)]
    return Scores(
        math.fsum(item.log_loss for item in losses) / len(losses),
        math.fsum(item.brier for item in losses) / len(losses),
    )


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run all specified conditions and return summaries plus replay metadata."""
    config = copy.deepcopy(config)
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    phase = config.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a nonempty stream namespace")
    source_dir = Path(__file__).resolve().parent
    source_sha256 = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count = _positive_int(config.get("train_count"), "train_count")
    test_count = _positive_int(config.get("test_count"), "test_count")
    replicates = _positive_int(config.get("replicates"), "replicates")
    raw_conditions = config.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("conditions must be a nonempty list")

    runs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw in raw_conditions:
        if not isinstance(raw, dict):
            raise ValueError("each condition must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("condition names must be unique nonempty strings")
        seen_names.add(name)
        simulation = SimulationConfig(
            n_forecasters=raw["n_forecasters"],
            signal_strength=raw["signal_strength"],
            overlap=raw["overlap"],
            group_ids=raw.get("group_ids"),
        )
        has_groups = raw.get("group_ids") is not None
        for replicate in range(replicates):
            # Different names prevent a training question from being reused in
            # test. The same names across conditions share outcomes and private
            # draws. Group labels also determine common-noise streams.
            training = simulate_questions(
                simulation, train_count, seed=seed, split=f"{phase}:train-{replicate}"
            )
            test = simulate_questions(
                simulation, test_count, seed=seed, split=f"{phase}:test-{replicate}"
            )
            evaluation = evaluate_train_test(training, test, overlap=simulation.overlap)
            scores = {method: _scores_dict(score)
                      for method, score in evaluation.scores.items()}
            if has_groups:
                # The exchangeable formula is generally wrong for independent
                # source groups, so do not call that output an oracle here.
                scores["exchangeable_formula"] = scores.pop("oracle")
                scores["grouped_oracle"] = _scores_dict(
                    _grouped_oracle_scores(test, simulation.overlap)
                )
            reference = "grouped_oracle" if has_groups else "oracle"
            comparisons = {}
            for method, baseline in (
                ("learned", "independent"),
                ("learned", "arithmetic_mean"),
                ("independent", reference),
            ):
                comparisons[f"{method}_minus_{baseline}"] = {
                    metric: scores[method][metric] - scores[baseline][metric]
                    for metric in ("log_loss", "brier")
                }
            runs.append({
                "condition": name,
                "replicate": replicate,
                "discount": evaluation.discount,
                "discount_boundary": evaluation.discount in (0.0, 1.0),
                "scores": scores,
                "comparisons": comparisons,
            })
    if any(
        hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != digest
        for name, digest in source_sha256.items()
    ):
        raise RuntimeError("source files changed during the experiment")
    return {
        "schema_version": 1,
        "status": "development_diagnostic",
        "python_version": platform.python_version(),
        "source_sha256": source_sha256,
        "config": config,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    data = args.config.read_bytes()
    config = json.loads(data)
    manifest_sha256 = None
    if args.manifest is not None:
        from .freeze import verify_manifest

        manifest_sha256 = verify_manifest(args.manifest, args.config)
    result = run_experiment(config)
    if manifest_sha256 is not None:
        result["status"] = "locked_synthetic_evaluation"
        result["manifest_sha256"] = manifest_sha256
    result["config_sha256"] = hashlib.sha256(data).hexdigest()
    if args.ledger is not None:
        from .archive import write_experiment_ledger

        result["ledger_receipt"] = write_experiment_ledger(result, args.ledger)
        result["ledger_path"] = str(args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as temporary:
        json.dump(result, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, args.output)
    print(f"Saved {len(result['runs'])} development runs to {args.output}")


if __name__ == "__main__":
    main()
