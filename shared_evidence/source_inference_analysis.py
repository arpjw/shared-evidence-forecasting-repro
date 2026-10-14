"""Paired replicate analysis for the frozen source-inference follow-up.

The bootstrap resamples independent fitted replicates, never individual test
questions. Primary intervals cover exactly two declared log-loss contrasts at
97.5% each; all sensitivity intervals are descriptive and unadjusted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any

from .simulate import derive_seed

METHODS = ("prior", "arithmetic_mean", "independent", "scalar",
           "inferred_groups", "grouped_oracle")
METRICS = ("log_loss", "brier")
CONTRASTS = (("inferred_groups", "scalar"),
             ("inferred_groups", "grouped_oracle"))


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return (1 - fraction) * values[lower] + fraction * values[min(lower + 1, len(values) - 1)]


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _cell(rows: list[dict[str, Any]], config: dict[str, Any], name: str) -> dict[str, Any]:
    count = config["replicates"]
    ordered: list[dict[str, Any] | None] = [None] * count
    for row in rows:
        rep = row.get("replicate")
        if isinstance(rep, bool) or not isinstance(rep, int) or not 0 <= rep < count or ordered[rep] is not None:
            raise ValueError("invalid or duplicate replicate")
        if set(row.get("scores", {})) != set(METHODS):
            raise ValueError("incomplete method scores")
        for method in METHODS:
            if set(row["scores"][method]) != set(METRICS):
                raise ValueError("incomplete metrics")
            for metric in METRICS:
                _number(row["scores"][method][metric])
        ordered[rep] = row
    if any(row is None for row in ordered):
        raise ValueError("missing replicate")
    complete = [row for row in ordered if row is not None]
    method_scores = {
        method: {metric: {
            "mean": _mean([row["scores"][method][metric] for row in complete]),
            "replicate_sd": statistics.stdev(row["scores"][method][metric] for row in complete),
        } for metric in METRICS} for method in METHODS
    }
    level = _number(config["interval_level"])
    if not 0 < level < 1:
        raise ValueError("invalid interval level")
    draws = config["bootstrap_resamples"]
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("invalid bootstrap count")
    rng = random.Random(derive_seed(config["bootstrap_seed"], config["phase"], name, "paired-bootstrap"))
    differences = {
        f"{method}_minus_{reference}": {
            metric: [row["scores"][method][metric] - row["scores"][reference][metric]
                     for row in complete] for metric in METRICS
        } for method, reference in CONTRASTS
    }
    bootstrap = {label: {metric: [] for metric in METRICS} for label in differences}
    for _ in range(draws):
        indices = [rng.randrange(count) for _ in range(count)]
        for label, metrics in differences.items():
            for metric, values in metrics.items():
                bootstrap[label][metric].append(math.fsum(values[i] for i in indices) / count)
    tail = (1 - level) / 2
    contrasts = {}
    for label, metrics in differences.items():
        contrasts[label] = {}
        for metric, values in metrics.items():
            samples = sorted(bootstrap[label][metric])
            contrasts[label][metric] = {
                "mean": _mean(values), "replicate_sd": statistics.stdev(values),
                "interval_level": level,
                "bootstrap_interval": [_quantile(samples, tail), _quantile(samples, 1 - tail)],
            }
    inferred_regret = contrasts["inferred_groups_minus_grouped_oracle"]["log_loss"]["mean"]
    scalar_oracle_gap = method_scores["scalar"]["log_loss"]["mean"] - method_scores["grouped_oracle"]["log_loss"]["mean"]
    gap_closed = 1 - inferred_regret / scalar_oracle_gap if scalar_oracle_gap > 0 else None
    diagnostics = [row["diagnostics"] for row in complete]
    exact = [item["exact_recovery"] for item in diagnostics]
    pair = [item["pair_error"] for item in diagnostics]
    if any(value is None for value in exact) != all(value is None for value in exact):
        raise ValueError("inconsistent recovery applicability")
    sign_counts = {key: sum(row["sign_counts"][key] for row in complete)
                   for key in ("scalar_opposite", "inferred_opposite", "scalar_zero",
                               "inferred_zero", "oracle_zero", "scalar_opposite_repaired")}
    total_questions = count * config["test_count"]
    signs = {"counts": sign_counts, "test_questions": total_questions,
             "scalar_opposite_fraction": sign_counts["scalar_opposite"] / total_questions,
             "inferred_opposite_fraction": sign_counts["inferred_opposite"] / total_questions,
             "repaired_fraction_among_scalar_opposites":
                 sign_counts["scalar_opposite_repaired"] / sign_counts["scalar_opposite"]
                 if sign_counts["scalar_opposite"] else None}
    return {
        "methods": method_scores, "contrasts": contrasts,
        "diagnostics": {"mean_inferred_group_count": _mean([item["group_count"] for item in diagnostics]),
                        "mean_inferred_overlap": _mean([item["overlap"] for item in diagnostics]),
                        "exact_recovery_fraction": _mean([float(value) for value in exact]) if exact[0] is not None else None,
                        "mean_pair_error": _mean(pair) if pair[0] is not None else None},
        "signs": signs, "scalar_oracle_gap": scalar_oracle_gap,
        "gap_fraction_closed": gap_closed,
        "gap_fraction_unstable": 0 < scalar_oracle_gap < 0.001,
        "gap_fraction_undefined": scalar_oracle_gap <= 0,
    }


def analyze_result(result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    if result.get("status") != "locked_synthetic_evaluation":
        raise ValueError("analysis requires a locked evaluation")
    if config.get("replicates", 0) < 2:
        raise ValueError("at least two replicates required")
    names = [item["name"] for item in config["conditions"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("invalid condition names")
    runs = result["runs"]
    if len(runs) != len(names) * config["replicates"] or any(row["condition"] not in names for row in runs):
        raise ValueError("incomplete experiment")
    cells = {name: _cell([row for row in runs if row["condition"] == name], config, name)
             for name in names}
    output: dict[str, Any] = {"phase": config["phase"], "train_count": config["train_count"],
                              "test_count": config["test_count"], "replicates": config["replicates"],
                              "bootstrap_resamples": config["bootstrap_resamples"],
                              "resampling_unit": "independently_fitted_replicate", "conditions": cells}
    if config["phase"] == "source-inference-primary-v1":
        if names != ["groups-9-1-rho0.9"] or config["interval_level"] != 0.975:
            raise ValueError("primary design differs from protocol")
        cell = cells[names[0]]
        improvement = cell["contrasts"]["inferred_groups_minus_scalar"]["log_loss"]
        regret = cell["contrasts"]["inferred_groups_minus_grouped_oracle"]["log_loss"]
        output["primary_gates"] = {
            "beats_scalar": improvement["bootstrap_interval"][1] < 0,
            "oracle_regret_below_0.01": regret["bootstrap_interval"][1] < 0.01,
        }
        output["primary_gates"]["joint_pass"] = all(output["primary_gates"].values())
    return output


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--verification", type=Path, action="append", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != len(args.verification):
        raise ValueError("one verification receipt is required per result")
    from scripts.source_inference_lock import verify_lock
    from .source_inference_experiment import ROOT
    lock = json.loads(args.lock.read_text())
    expected_phases = {json.loads((ROOT / name).read_text())["phase"]
                       for name in lock["config_sha256"]}
    phases: dict[str, Any] = {}
    receipts = {}
    for input_path, receipt_path in zip(args.input, args.verification):
        raw = input_path.read_bytes()
        result = json.loads(raw)
        phase = result["config"]["phase"]
        if phase in phases:
            raise ValueError("duplicate phase")
        config_path = next(ROOT / name for name in lock["config_sha256"]
                           if json.loads((ROOT / name).read_text())["phase"] == phase)
        if verify_lock(args.lock, config_path) != result.get("lock_sha256"):
            raise ValueError("result lock mismatch")
        if _digest(config_path) != result.get("config_sha256"):
            raise ValueError("result config mismatch")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "verified" or
                receipt.get("summary_sha256") != hashlib.sha256(raw).hexdigest() or
                receipt.get("ledger_sha256") != result["ledger_receipt"]["sha256"]):
            raise ValueError("verification receipt does not match result")
        phases[phase] = analyze_result(result)
        receipts[phase] = receipt
    if set(phases) != expected_phases:
        raise ValueError("missing locked suite phase")
    output = {"schema_version": 1, "lock_sha256": _digest(args.lock),
              "status": "independently_verified_locked_synthetic_evaluation",
              "phases": phases, "verification_receipts": receipts,
              "multiplicity_note": "Only the two primary contrasts use simultaneous 97.5% intervals; sensitivity intervals are descriptive."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent, delete=False) as temporary:
        json.dump(output, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temp_path = Path(temporary.name)
    os.replace(temp_path, args.output)
    print(f"Saved analysis of {len(phases)} locked phases to {args.output}")


if __name__ == "__main__":
    main()
