"""Bind a synthetic evaluation to its protocol, code, and configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from .simulate import derive_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOUND_SOURCES = (
    "shared_evidence/model.py",
    "shared_evidence/simulate.py",
    "shared_evidence/evaluate.py",
    "shared_evidence/groups.py",
    "shared_evidence/experiment.py",
    "shared_evidence/ledger.py",
    "shared_evidence/archive.py",
    "shared_evidence/verify.py",
    "shared_evidence/analysis.py",
    "shared_evidence/freeze.py",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    config_path: str | Path, protocol_path: str | Path,
    *, bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Produce a pre-run lock; save it before inspecting production outcomes."""
    config_path = Path(config_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    try:
        config_name = str(config_path.relative_to(PROJECT_ROOT))
        protocol_name = str(protocol_path.relative_to(PROJECT_ROOT))
    except ValueError as error:
        raise ValueError("manifest inputs must be inside the project") from error
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config.get("phase"), str) or not config["phase"].startswith("production-"):
        raise ValueError("locked config phase must start with production-")
    if config.get("schema_version") != 1:
        raise ValueError("locked config schema_version must be 1")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 2:
        raise ValueError("bootstrap_resamples must be an integer >= 2")
    return {
        "schema_version": 1,
        "status": "frozen_before_execution",
        "scope": "synthetic_primary_condition",
        "development_outcomes_inspected": True,
        "config_path": config_name,
        "config_sha256": _digest(config_path),
        "protocol_path": protocol_name,
        "protocol_sha256": _digest(protocol_path),
        "source_sha256": {
            name: _digest(PROJECT_ROOT / name) for name in BOUND_SOURCES
        },
        "python_version": platform.python_version(),
        "analysis": {
            "analysis_seed": derive_seed(
                config["seed"], "replicate-analysis-v1", config["phase"]
            ),
            "bootstrap_resamples": bootstrap_resamples,
            "confidence_level": 0.975,
            "primary_metric": "log_loss",
            "primary_contrasts": [
                ["learned", "independent"],
                ["learned", "arithmetic_mean"],
            ],
            "resampling_unit": "independent_training_and_test_replicate",
        },
    }


def verify_manifest(path: str | Path, config_path: str | Path) -> str:
    """Reject any changed protocol, source, Python version, or config bytes."""
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "frozen_before_execution":
        raise ValueError("invalid frozen manifest")
    config_path = Path(config_path).resolve()
    if (str(config_path.relative_to(PROJECT_ROOT)) != manifest.get("config_path")
            or _digest(config_path) != manifest.get("config_sha256")):
        raise ValueError("config differs from frozen manifest")
    protocol_path = PROJECT_ROOT / manifest["protocol_path"]
    if _digest(protocol_path) != manifest.get("protocol_sha256"):
        raise ValueError("protocol differs from frozen manifest")
    if platform.python_version() != manifest.get("python_version"):
        raise ValueError("Python version differs from frozen manifest")
    if set(manifest.get("source_sha256", {})) != set(BOUND_SOURCES):
        raise ValueError("source list differs from frozen manifest")
    for name, digest in manifest["source_sha256"].items():
        if _digest(PROJECT_ROOT / name) != digest:
            raise ValueError(f"{name} differs from frozen manifest")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.config, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as temporary:
        json.dump(manifest, temporary, indent=2, sort_keys=True, allow_nan=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, args.output)
    print(f"Saved pre-run manifest to {args.output}")


if __name__ == "__main__":
    main()
