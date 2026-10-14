"""Describe every saved stress cell, retaining failures of the scalar model.

Intervals are descriptive replicate bootstraps, not simultaneous confidence
statements. This reader validates summaries; ledger replay is a separate step.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

# Support both importing this file in tests and running it from the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stress_lock import ROOT, verify_lock
from shared_evidence.analysis import analyze_experiment, summarize_replicates
from shared_evidence.experiment import SOURCE_FILES
from shared_evidence.simulate import SimulationConfig, derive_seed


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f"{label} must be a SHA256 digest")
    return value


def summarize_result(result: dict[str, Any], *, bootstrap_resamples: int = 10000) -> dict[str, Any]:
    """Validate complete coverage and compute statistics from stored scores."""
    if result.get('schema_version') != 1:
        raise ValueError('unsupported result schema')
    config = result['config']
    if config.get('schema_version') != 1:
        raise ValueError('unsupported config schema')
    for field in ('train_count', 'test_count', 'replicates'):
        _positive(config.get(field), field)
    if not isinstance(config.get('phase'), str) or not config['phase']:
        raise ValueError('phase must be nonempty')
    derive_seed(config.get('seed'))
    sources = result.get('source_sha256', {})
    if set(sources) != set(SOURCE_FILES):
        raise ValueError('experiment source hash coverage differs')
    for name, digest in sources.items():
        _hash(digest, name)
    _hash(result.get('config_sha256'), 'config_sha256')
    if result.get('status') == 'locked_synthetic_evaluation':
        _hash(result.get('manifest_sha256'), 'manifest_sha256')
    # This validates run IDs, completeness, finite metrics and method consistency.
    base = analyze_experiment(result, bootstrap_resamples=bootstrap_resamples,
                              confidence_level=0.95)
    cells = {}
    for condition in config['conditions']:
        name = condition['name']
        simulation = SimulationConfig(
            condition['n_forecasters'], condition['signal_strength'],
            condition['overlap'], condition.get('group_ids'))
        grouped = condition.get('group_ids') is not None
        oracle = 'grouped_oracle' if grouped else 'oracle'
        expected = {'learned', 'independent', 'arithmetic_mean', 'prior',
                    'individual', oracle}
        if grouped:
            expected.add('exchangeable_formula')
        rows = sorted((run for run in result['runs'] if run['condition'] == name),
                      key=lambda row: row['replicate'])
        discounts = []
        for row in rows:
            if set(row['scores']) != expected:
                raise ValueError(f'{name}: method set differs from implemented experiment')
            discount = _number(row.get('discount'), 'discount')
            if not 0 <= discount <= 1:
                raise ValueError('discount outside frozen fitting bounds')
            if not isinstance(row.get('discount_boundary'), bool) or row['discount_boundary'] != (discount in (0, 1)):
                raise ValueError('incorrect discount boundary flag')
            discounts.append(discount)
            for score in row['scores'].values():
                if score['log_loss'] < 0 or not 0 <= score['brier'] <= 1:
                    raise ValueError('score outside proper-loss domain')
        # Reuse the paired bootstrap implementation, adding oracle regret.
        one = dict(result, config=dict(config, conditions=[condition]), runs=rows)
        analysis = analyze_experiment(one, bootstrap_resamples=bootstrap_resamples,
            confidence_level=0.95, contrasts=(('learned', 'independent'),
            ('learned', 'arithmetic_mean'), ('learned', oracle)))
        sizes = sorted(Counter(simulation.groups).values(), reverse=True)
        n, rho = simulation.n_forecasters, simulation.overlap
        # The sufficient summed signal has mean +/- n*a and noise variance V.
        # Its population logistic slope is n/V, even for unequal groups.
        beta = n / (n + rho * (sum(size * size for size in sizes) - n))
        summary = analysis['conditions'][name]
        summary.update({
            'condition': condition, 'oracle_method': oracle,
            'group_sizes': sizes,
            'population_optimal_scalar': beta if simulation.signal_strength > 0 else None,
            'population_scalar_identified': simulation.signal_strength > 0,
            'oracle_group_weights': [1 / (1 + (size - 1) * rho) for size in sizes],
            'scalar_can_represent_oracle': simulation.signal_strength == 0 or rho == 0 or len(set(sizes)) == 1,
            'discount': summarize_replicates(discounts,
                seed=derive_seed(base['analysis_seed'], name, 'discount'),
                bootstrap_resamples=bootstrap_resamples, confidence_level=0.95),
            'discount_min': min(discounts), 'discount_max': max(discounts),
            'discount_zero_count': discounts.count(0), 'discount_one_count': discounts.count(1),
            'discount_below_exchangeable_range_count': sum(x < 1/n for x in discounts),
            'replicate_ids': [row['replicate'] for row in rows],
            'train_count_per_replicate': config['train_count'],
            'test_count_per_replicate': config['test_count'],
        })
        cells[name] = summary
    return {'phase': config['phase'], 'status': result.get('status', 'unknown'),
            'config': config, 'config_sha256': result['config_sha256'],
            'source_sha256': sources, 'manifest_sha256': result.get('manifest_sha256'),
            'ledger_receipt': result.get('ledger_receipt'),
            'analysis_seed': base['analysis_seed'],
            'expected_runs': len(cells) * config['replicates'],
            'observed_runs': len(result['runs']), 'cells': dict(sorted(cells.items()))}


def build_report(paths: Sequence[Path], *, bootstrap_resamples: int = 10000,
                 lock_path: Path | None = None, root: Path = ROOT) -> dict[str, Any]:
    """Read immutable experiment bytes; reject duplicated phases and inputs."""
    experiments = {}
    lock_digest = None
    locked_configs = {}
    if lock_path is not None:
        lock_digest = verify_lock(lock_path, root=root)
        lock = json.loads(lock_path.read_bytes())
        if bootstrap_resamples != lock['analysis']['bootstrap_resamples']:
            raise ValueError('bootstrap resamples differ from stress lock')
        for name, digest in lock['config_sha256'].items():
            config = json.loads((root / name).read_bytes())
            locked_configs[config['phase']] = (config, digest)
    hashes: set[str] = set()
    for path in paths:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        result = json.loads(raw)
        report = summarize_result(result, bootstrap_resamples=bootstrap_resamples)
        phase = report['phase']
        if lock_path is not None:
            if phase not in locked_configs:
                raise ValueError('input phase is absent from stress lock')
            config, config_digest = locked_configs[phase]
            if (json.dumps(result['config'], sort_keys=True, allow_nan=False)
                    != json.dumps(config, sort_keys=True, allow_nan=False)
                    or result['config_sha256'] != config_digest):
                raise ValueError('input config differs from stress lock')
            expected_sources = {name: lock['source_sha256']['shared_evidence/' + name]
                                for name in SOURCE_FILES}
            if result['source_sha256'] != expected_sources:
                raise ValueError('input sources differ from stress lock')
            if result.get('python_version') != lock['python_version']:
                raise ValueError('input Python version differs from stress lock')
        if phase in experiments or digest in hashes:
            raise ValueError('duplicate input or phase would double count experiments')
        hashes.add(digest)
        experiments[phase] = dict(report, input_sha256=digest, input_path=str(path))
    if lock_path is not None and set(experiments) != set(locked_configs):
        raise ValueError('missing locked stress phases')
    if not experiments:
        raise ValueError('at least one saved experiment is required')
    root = Path(__file__).resolve().parents[1]
    return {'schema_version': 1, 'status': 'descriptive_synthetic_stress_report',
            'bootstrap_resamples': bootstrap_resamples, 'confidence_level': 0.95,
            'stress_lock_sha256': lock_digest,
            'grouped_opposite_sign_diagnostic': 'Not implemented in this report; requires a separate matching-ledger diagnostic.',
            'analysis_source_sha256': {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ('scripts/stress_report.py', 'shared_evidence/analysis.py',
                             'shared_evidence/simulate.py')},
            'verification_scope': 'Summary schema, cells, score domains, and fitted bounds checked; ledger replay is separate.',
            'interpretation': 'Every input cell retained. Intervals are descriptive, not adjusted across cells. Conditions can share random draws; no pooled cross-cell inference is made. Negative realized oracle regret can occur from test sampling.',
            'experiments': dict(sorted(experiments.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    report = build_report(args.input, bootstrap_resamples=args.bootstrap_resamples,
                          lock_path=args.lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent,
                                     delete=False) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(f'Saved {sum(len(x["cells"]) for x in report["experiments"].values())} complete stress cells to {args.output}')


if __name__ == '__main__':
    main()
