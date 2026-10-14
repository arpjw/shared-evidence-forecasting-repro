"""Independent replay of fitted discounts and summary scores from a ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import ForecastRecord, brier_score, evaluate, fit_discount, log_loss
from .experiment import SOURCE_FILES
from .freeze import PROJECT_ROOT, verify_manifest
from .groups import grouped_oracle_log_odds
from .ledger import read_ledger, replay_ledger


def verify_summary_and_ledger(
    summary_path: str | Path, ledger_path: str | Path,
    *, manifest_path: str | Path | None = None,
) -> dict[str, int]:
    """Recompute fitted parameters and every aggregate score from saved rows.

    The ledger's byte receipt, source hashes, simulator values and test
    predictions are checked first. This verifier additionally refits from
    recorded training questions and compares all saved summary scores.
    """
    result: dict[str, Any] = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    manifest = None
    if result.get("status") == "locked_synthetic_evaluation":
        if manifest_path is None:
            raise ValueError("locked evaluation requires its frozen manifest")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        digest = verify_manifest(manifest_path, PROJECT_ROOT / manifest["config_path"])
        if digest != result.get("manifest_sha256"):
            raise ValueError("summary manifest hash differs from frozen manifest")
        config_bytes = (PROJECT_ROOT / manifest["config_path"]).read_bytes()
        if (result.get("config") != json.loads(config_bytes)
                or result.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest()
                or result.get("config_sha256") != manifest["config_sha256"]):
            raise ValueError("summary config differs from frozen manifest")
    receipt = result["ledger_receipt"]
    replay = replay_ledger(
        ledger_path, expected_sha256=receipt["sha256"], strict_sources=True
    )
    metadata, records = read_ledger(ledger_path, expected_sha256=receipt["sha256"])
    if metadata["config"] != result["config"]:
        raise ValueError("ledger and summary configurations differ")
    if metadata["source_sha256"] != result["source_sha256"]:
        raise ValueError("ledger and summary source hashes differ")
    if set(result["source_sha256"]) != set(SOURCE_FILES):
        raise ValueError("summary source hash coverage differs from experiment sources")
    if manifest is not None:
        for name in SOURCE_FILES:
            if result["source_sha256"][name] != manifest["source_sha256"][f"shared_evidence/{name}"]:
                raise ValueError("summary source hash differs from frozen manifest")
    if receipt["count"] != len(records):
        raise ValueError("ledger and summary record counts differ")
    expected_runs = {
        (condition["name"], replicate)
        for condition in result["config"]["conditions"]
        for replicate in range(result["config"]["replicates"])
    }
    observed_runs = [(run["condition"], run["replicate"]) for run in result["runs"]]
    if len(observed_runs) != len(set(observed_runs)) or set(observed_runs) != expected_runs:
        raise ValueError("summary run coverage differs from configured grid")
    expected_streams = {
        f"{condition}:r{replicate}:{role}"
        for condition, replicate in expected_runs
        for role in ("train", "test")
    }
    if set(metadata["streams"]) != expected_streams:
        raise ValueError("ledger stream coverage differs from configured grid")
    condition_by_name = {row["name"]: row for row in result["config"]["conditions"]}
    for run in result["runs"]:
        condition = run["condition"]
        replicate = run["replicate"]
        raw = condition_by_name[condition]
        simulation = {
            "n_forecasters": raw["n_forecasters"],
            "signal_strength": raw["signal_strength"],
            "overlap": raw["overlap"],
            "group_ids": raw.get("group_ids"),
        }
        for role in ("train", "test"):
            stream_id = f"{condition}:r{replicate}:{role}"
            expected_descriptor = {
                "simulation": simulation,
                "seed": result["config"]["seed"],
                "split": f"{result['config']['phase']}:{role}-{replicate}",
                "role": role,
                "count": result["config"][f"{role}_count"],
                "start": 0,
            }
            if role == "test":
                expected_descriptor["discount"] = run["discount"]
            if metadata["streams"][stream_id] != expected_descriptor:
                raise ValueError(f"ledger stream descriptor differs from config: {stream_id}")
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stream[record["stream"]].append(record)
    if set(by_stream) != set(metadata["streams"]):
        raise ValueError("ledger stream coverage differs from metadata")
    for run in result["runs"]:
        condition = run["condition"]
        replicate = run["replicate"]
        train_id = f"{condition}:r{replicate}:train"
        test_id = f"{condition}:r{replicate}:test"
        training = tuple(
            ForecastRecord(tuple(row["log_odds"]), row["outcome"])
            for row in by_stream[train_id]
        )
        test_rows = by_stream[test_id]
        test = tuple(
            ForecastRecord(tuple(row["log_odds"]), row["outcome"])
            for row in test_rows
        )
        fitted = fit_discount(training)
        if not math.isclose(fitted, run["discount"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("saved discount differs from training-only refit")
        if run.get("discount_boundary") != (fitted in (0.0, 1.0)):
            raise ValueError("saved discount boundary diagnostic differs from refit")
        stream = metadata["streams"][test_id]
        if not math.isclose(fitted, stream["discount"], rel_tol=0, abs_tol=1e-12):
            raise ValueError("ledger discount differs from training-only refit")
        overlap = stream["simulation"]["overlap"]
        regenerated = evaluate(test, discount=fitted, overlap=overlap)
        expected_scores = {
            name: {"log_loss": score.log_loss, "brier": score.brier}
            for name, score in regenerated.scores.items()
        }
        if stream["simulation"].get("group_ids") is not None:
            expected_scores["exchangeable_formula"] = expected_scores.pop("oracle")
            group_losses = [
                (
                    log_loss(
                        grouped_oracle_log_odds(
                            row["log_odds"], row["group_ids"], overlap
                        ), row["outcome"]
                    ),
                    brier_score(
                        grouped_oracle_log_odds(
                            row["log_odds"], row["group_ids"], overlap
                        ), row["outcome"]
                    ),
                )
                for row in test_rows
            ]
            expected_scores["grouped_oracle"] = {
                "log_loss": math.fsum(pair[0] for pair in group_losses) / len(group_losses),
                "brier": math.fsum(pair[1] for pair in group_losses) / len(group_losses),
            }
        if any("predictions" in row for row in by_stream[train_id]):
            raise ValueError("training records unexpectedly contain scored predictions")
        if any(set(row.get("predictions", {})) != set(expected_scores) for row in test_rows):
            raise ValueError("test prediction method coverage differs from summary")
        if set(expected_scores) != set(run["scores"]):
            raise ValueError("saved method list differs from replay")
        for name, metrics in expected_scores.items():
            for metric, value in metrics.items():
                if not math.isclose(value, run["scores"][name][metric],
                                    rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"saved {name} {metric} differs from replay")
        reference = ("grouped_oracle" if "grouped_oracle" in expected_scores else "oracle")
        comparisons = {
            f"{method}_minus_{baseline}": {
                metric: expected_scores[method][metric] - expected_scores[baseline][metric]
                for metric in ("log_loss", "brier")
            }
            for method, baseline in (
                ("learned", "independent"),
                ("learned", "arithmetic_mean"),
                ("independent", reference),
            )
        }
        if comparisons != run["comparisons"]:
            raise ValueError("saved paired comparisons differ from replay")
    return {**replay, "runs_verified": len(result["runs"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    verified = verify_summary_and_ledger(
        args.summary, args.ledger, manifest_path=args.manifest
    )
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    main()
