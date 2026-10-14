"""Measure decisions a nonnegative scalar pool cannot recover in group stress runs.

The full ledger must also pass ``shared_evidence.verify``. This streaming
diagnostic checks the ledger byte receipt and all grouped test-row counts, then
compares the sign of summed individual logits with the stored group oracle.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def analyze_group_signs(summary_path: Path, ledger_path: Path) -> dict[str, Any]:
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    receipt = summary["ledger_receipt"]
    if _file_hash(ledger_path) != receipt["sha256"]:
        raise ValueError("ledger byte receipt mismatch")
    config = summary["config"]
    groups = {c["name"] for c in config["conditions"] if c.get("group_ids") is not None}
    if not groups or len(groups) != len(config["conditions"]):
        raise ValueError("diagnostic requires a fully grouped configuration")
    expected = {
        f"{name}:r{replicate}:test": (name, replicate)
        for name in groups for replicate in range(config["replicates"])
    }
    totals = {stream: {"questions": 0, "opposite": 0, "ties": 0,
                       "oracle_distance_sum": 0.0} for stream in expected}
    with gzip.open(ledger_path, "rt", encoding="ascii") as handle:
        header = json.loads(handle.readline())
        if header.get("type") != "header" or header["metadata"]["config"] != config:
            raise ValueError("ledger header configuration differs from summary")
        observed_all = 0
        trailer = None
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "trailer":
                trailer = row
                if handle.read(1):
                    raise ValueError("data follows ledger trailer")
                break
            observed_all += 1
            stream = row.get("stream")
            if stream not in expected:
                continue
            values = row["log_odds"]
            oracle = row["predictions"]["grouped_oracle"]["logit"]
            summed = math.fsum(values)
            if not math.isfinite(summed) or not math.isfinite(oracle):
                raise ValueError("nonfinite decision logit")
            item = totals[stream]
            item["questions"] += 1
            signs = (_sign(summed), _sign(oracle))
            if 0 in signs:
                item["ties"] += 1
            elif signs[0] != signs[1]:
                item["opposite"] += 1
                probability = 1 / (1 + math.exp(-oracle)) if oracle >= 0 else math.exp(oracle) / (1 + math.exp(oracle))
                item["oracle_distance_sum"] += abs(probability - 0.5)
    if trailer is None or trailer.get("count") != observed_all or observed_all != receipt["count"]:
        raise ValueError("ledger record count differs from trailer or receipt")
    if any(item["questions"] != config["test_count"] for item in totals.values()):
        raise ValueError("grouped test stream count differs from configuration")
    cells: dict[str, Any] = {}
    for name in sorted(groups):
        rows = [totals[f"{name}:r{replicate}:test"] for replicate in range(config["replicates"])]
        questions = sum(row["questions"] for row in rows)
        opposite = sum(row["opposite"] for row in rows)
        ties = sum(row["ties"] for row in rows)
        distance = math.fsum(row["oracle_distance_sum"] for row in rows)
        cells[name] = {
            "questions": questions,
            "opposite_sign_count": opposite,
            "opposite_sign_fraction": opposite / questions,
            "tie_count": ties,
            "mean_oracle_probability_distance_from_half_on_opposite":
                distance / opposite if opposite else None,
            "replicates": [
                {"replicate": replicate, "questions": row["questions"],
                 "opposite_sign_count": row["opposite"], "tie_count": row["ties"]}
                for replicate, row in enumerate(rows)
            ],
        }
    return {
        "schema_version": 1,
        "phase": config["phase"],
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "ledger_sha256": receipt["sha256"],
        "definition": "strict opposite signs of total report logit and group-oracle logit",
        "scope": "diagnostic of nonnegative scalar decision limitation; not an inference interval",
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_group_signs(args.summary, args.ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved grouped sign diagnostic for {len(report['cells'])} cells to {args.output}")


if __name__ == "__main__":
    main()
