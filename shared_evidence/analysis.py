"""Replication-level summaries with a paired percentile bootstrap.

A replicate includes a fresh training fit and held-out question batch. Resampling
replicate means retains training variability and avoids treating questions that
share a fitted discount as independent fitted experiments. Independence of
replicates is an input assumption, not established from numeric IDs.

Percentile intervals are approximate, especially with few replicates, and can
collapse for constant samples. A 97.5% interval per hypothesis is a Bonferroni
choice for exactly two prespecified hypotheses at nominal family error 5%; it
does not correct all condition/metric combinations returned by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .simulate import derive_seed

METRICS = ("log_loss", "brier")
DEFAULT_CONTRASTS = (("learned", "independent"), ("learned", "arithmetic_mean"))


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("scores must be finite numbers")
    return float(value)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Linear interpolation at (B-1)*p, matching the usual type-7 quantile."""
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(sorted_values) - 1)
    return (1 - fraction) * sorted_values[lower] + fraction * sorted_values[upper]


def summarize_replicates(
    values: Sequence[float], *, seed: int, bootstrap_resamples: int = 10000,
    confidence_level: float = 0.975,
) -> dict[str, Any]:
    """Equal-weight mean, sample SD/SE, and two-sided percentile interval."""
    if len(values) < 2:
        raise ValueError("at least two independent replicates are required")
    samples = tuple(_finite_number(value) for value in values)
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be an integer >= 2")
    confidence_level = _finite_number(confidence_level)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    # Validate the seed without changing the documented bootstrap stream.
    derive_seed(seed)
    rng = random.Random(seed)
    count = len(samples)
    mean = math.fsum(value / count for value in samples)
    sd = statistics.stdev(samples)
    draws = sorted(
        math.fsum(samples[rng.randrange(count)] / count for _ in range(count))
        for _ in range(bootstrap_resamples)
    )
    tail = (1 - confidence_level) / 2
    output = {
        "replicates": count,
        "mean": mean,
        "sample_sd": sd,
        "standard_error": sd / math.sqrt(count),
        "confidence_level": confidence_level,
        "bootstrap_interval": [_quantile(draws, tail), _quantile(draws, 1 - tail)],
    }
    if not all(math.isfinite(value) for value in (mean, sd, *output["bootstrap_interval"])):
        raise ValueError("summary overflowed; score magnitudes are too large")
    return output


def analyze_experiment(
    result: dict[str, Any], *, bootstrap_resamples: int = 10000,
    confidence_level: float = 0.975, analysis_seed: int | None = None,
    contrasts: Sequence[tuple[str, str]] = DEFAULT_CONTRASTS,
) -> dict[str, Any]:
    """Summarize complete experiment cells and paired within-replicate contrasts.

    Configuration defines the expected condition names and replicate IDs 0..R-1.
    Scores are recomputed into contrasts rather than trusting cached comparisons.
    Contrasts share bootstrap indices within each condition and metric.
    All methods within a condition must be present in every replicate. Different
    conditions may have different oracle methods. Run order never affects output.
    """
    config = result.get("config", {})
    count = config.get("replicates")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("configuration must specify at least two replicates")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("configuration must specify conditions")
    names = [condition.get("name") for condition in conditions if isinstance(condition, dict)]
    if len(names) != len(conditions) or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("condition names must be unique nonempty strings")
    if analysis_seed is None:
        phase = config.get("phase", "legacy")
        if not isinstance(phase, str):
            raise ValueError("phase must be a string")
        analysis_seed = derive_seed(config.get("seed"), "replicate-analysis-v1", phase)
    else:
        derive_seed(analysis_seed)
    contrasts = tuple(tuple(pair) for pair in contrasts)
    if any(len(pair) != 2 or any(not isinstance(method, str) or not method for method in pair) for pair in contrasts):
        raise ValueError("each contrast must name two methods")
    contrast_names = [f"{method}_minus_{reference}" for method, reference in contrasts]
    if len(set(contrast_names)) != len(contrast_names):
        raise ValueError("contrast names must be unique")
    cells: dict[str, dict[int, dict[str, dict[str, float]]]] = {name: {} for name in names}
    runs = result.get("runs")
    if not isinstance(runs, list):
        raise ValueError("runs must be a list")
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be an object")
        name, replicate = run.get("condition"), run.get("replicate")
        if not isinstance(name, str) or name not in cells:
            raise ValueError("unknown run condition")
        if isinstance(replicate, bool) or not isinstance(replicate, int) or not 0 <= replicate < count or replicate in cells[name]:
            raise ValueError("duplicate or invalid replicate ID")
        scores = run.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise ValueError("scores must be a nonempty method mapping")
        validated = {}
        for method, metrics in scores.items():
            if not isinstance(method, str) or not method or not isinstance(metrics, dict) or any(metric not in metrics for metric in METRICS):
                raise ValueError("missing method or score metric")
            validated[method] = {metric: _finite_number(metrics[metric]) for metric in METRICS}
        cells[name][replicate] = validated
    output_conditions = {}
    for name in sorted(cells):
        rows = cells[name]
        if set(rows) != set(range(count)):
            raise ValueError(f"missing replicate cells for {name}")
        methods = set(rows[0])
        if any(set(row) != methods for row in rows.values()):
            raise ValueError(f"missing method cells for {name}")
        if any(method not in methods for pair in contrasts for method in pair):
            raise ValueError(f"missing contrast method for {name}")
        def summarize(values: Sequence[float], kind: str, label: str, metric: str) -> dict[str, Any]:
            interval_level = (confidence_level if kind == "contrast" and metric == "log_loss"
                              else 0.95)
            return summarize_replicates(
                # All contrasts within condition/metric use identical replicate
                # index draws. Labels must not change this paired bootstrap.
                values, seed=derive_seed(
                    analysis_seed, name, kind,
                    label if kind == "method" else "shared-contrast-indices", metric,
                ),
                bootstrap_resamples=bootstrap_resamples, confidence_level=interval_level,
            )
        method_output = {
            method: {
                metric: summarize([rows[i][method][metric] for i in range(count)], "method", method, metric)
                for metric in METRICS
            } for method in sorted(methods)
        }
        contrast_output = {
            label: {
                metric: summarize(
                    [rows[i][method][metric] - rows[i][reference][metric] for i in range(count)],
                    "contrast", label, metric,
                ) for metric in METRICS
            } for label, (method, reference) in zip(contrast_names, contrasts)
        }
        output_conditions[name] = {"methods": method_output, "contrasts": contrast_output}
    return {
        "schema_version": 1, "analysis_seed": analysis_seed,
        "source_experiment": {
            "status": result.get("status", "unknown"),
            "phase": config.get("phase"),
            "train_count": config.get("train_count"),
            "test_count": config.get("test_count"),
            "replicates": count,
        },
        "resampling_unit": "independent_training_and_test_replicate",
        "bootstrap_method": "percentile", "bootstrap_resamples": bootstrap_resamples,
        "confidence_level": confidence_level, "conditions": output_conditions,
        "multiplicity_note": "Intervals do not jointly cover all returned conditions and metrics.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    input_bytes = args.input.read_bytes()
    result = json.loads(input_bytes)
    verification = None
    if result.get("status") == "locked_synthetic_evaluation":
        if args.manifest is None or args.ledger is None:
            raise ValueError("locked analysis requires manifest and ledger verification")
        from .freeze import PROJECT_ROOT, verify_manifest
        from .verify import verify_summary_and_ledger

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        digest = verify_manifest(args.manifest, PROJECT_ROOT / manifest["config_path"])
        if digest != result.get("manifest_sha256"):
            raise ValueError("summary manifest hash differs from frozen manifest")
        if args.bootstrap_resamples != manifest["analysis"]["bootstrap_resamples"]:
            raise ValueError("bootstrap count differs from frozen manifest")
        verification = verify_summary_and_ledger(
            args.input, args.ledger, manifest_path=args.manifest
        )
    analysis = analyze_experiment(
        result, bootstrap_resamples=args.bootstrap_resamples
    )
    if result.get("status") == "locked_synthetic_evaluation":
        if analysis["analysis_seed"] != manifest["analysis"]["analysis_seed"]:
            raise ValueError("analysis seed differs from frozen manifest")
        if analysis["confidence_level"] != manifest["analysis"]["confidence_level"]:
            raise ValueError("confidence level differs from frozen manifest")
    report = {
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "analysis": analysis,
    }
    if verification is not None:
        report["verification"] = verification
        report["manifest_sha256"] = result["manifest_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as temporary:
        json.dump(report, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, args.output)
    print(f"Saved analysis to {args.output}")


if __name__ == "__main__":
    main()
