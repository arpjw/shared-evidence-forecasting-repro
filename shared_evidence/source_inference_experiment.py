"""Replayable train-only source inference experiment with a required ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path

from .evaluate import ForecastRecord, fit_discount, prediction_logits
from .groups import grouped_oracle_log_odds
from .ledger import question_record, write_ledger
from .simulate import SimulationConfig, derive_seed, simulate_questions
from .source_inference import fit_source_inference

ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = (
    'shared_evidence/model.py', 'shared_evidence/simulate.py',
    'shared_evidence/evaluate.py', 'shared_evidence/groups.py',
    'shared_evidence/ledger.py', 'shared_evidence/source_inference.py',
    'shared_evidence/source_inference_experiment.py',
    'shared_evidence/source_inference_verify.py',
    'shared_evidence/source_inference_analysis.py', 'scripts/source_inference_lock.py',
)


def source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def validate_config(config: dict) -> None:
    if config.get('schema_version') != 1 or not config.get('phase', '').startswith('source-inference-'):
        raise ValueError('invalid source inference config')
    if 'validation_count' in config or 'thresholds' in config:
        raise ValueError('validation selection is not part of this train-only design')
    for name in ('seed', 'train_count', 'test_count', 'replicates'):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'invalid {name}')
    if config['train_count'] < 3 or config.get('threshold_error_parameter') != 0.05:
        raise ValueError('training requires at least three rows and threshold_error_parameter=.05')
    names = [item['name'] for item in config['conditions']]
    if not names or len(set(names)) != len(names):
        raise ValueError('conditions must have unique names')


def run_experiment(config: dict, ledger_path: Path) -> dict:
    validate_config(config)
    config = json.loads(json.dumps(config))
    hashes = source_hashes()
    streams, fits, runs = {}, {}, []
    for raw in config['conditions']:
        for rep in range(config['replicates']):
            key = f"{raw['name']}:{rep}"
            simulation = {k: raw[k] for k in ('n_forecasters', 'signal_strength', 'overlap', 'group_ids')}
            for role in ('train', 'test'):
                streams[f'{key}:{role}'] = {
                    'simulation': simulation, 'seed': derive_seed(config['seed'], config['phase'], raw['name'], rep, role),
                    'split': role, 'role': role, 'count': config[f'{role}_count'], 'start': 0,
                }
            stream = streams[f'{key}:train']
            questions = simulate_questions(SimulationConfig(**simulation), stream['count'], seed=stream['seed'], split='train')
            records = tuple(ForecastRecord(q.log_odds, q.outcome) for q in questions)
            fits[key] = {'source': asdict(fit_source_inference(records)), 'scalar': fit_discount(records)}
    # Fit states are persisted before any test row is generated. The generator
    # stores every generated train/test question, and each test prediction/score.
    metadata = {'phase': config['phase'], 'config': config, 'source_sha256': hashes,
                'python_version': platform.python_version(), 'streams': streams, 'fits': fits}

    def rows():
        for raw in config['conditions']:
            for rep in range(config['replicates']):
                key = f"{raw['name']}:{rep}"
                state = fits[key]
                source = state['source']
                groups = source['group_ids']
                weights = [1 / (1 + (groups.count(g) - 1) * source['overlap']) for g in groups]
                scores: dict[str, dict[str, list[float]]] = {}
                signs = {'scalar_opposite': 0, 'inferred_opposite': 0, 'scalar_zero': 0,
                         'inferred_zero': 0, 'oracle_zero': 0, 'scalar_opposite_repaired': 0}
                for role in ('train', 'test'):
                    stream_id = f'{key}:{role}'
                    stream = streams[stream_id]
                    questions = simulate_questions(SimulationConfig(**stream['simulation']), stream['count'], seed=stream['seed'], split=role)
                    for q in questions:
                        predictions = None
                        if role == 'test':
                            base = prediction_logits(q.log_odds, discount=state['scalar'], overlap=0.0)
                            predictions = {name: base[name] for name in ('prior', 'arithmetic_mean', 'independent')}
                            predictions['scalar'] = base['learned']
                            predictions['inferred_groups'] = source['discount'] * math.fsum(w*x for w,x in zip(weights,q.log_odds))
                            predictions['grouped_oracle'] = grouped_oracle_log_odds(q.log_odds, q.group_ids, raw['overlap'])
                        record = question_record(q, stream=stream_id, predictions=predictions)
                        if predictions is not None:
                            for method, score in record['predictions'].items():
                                bucket = scores.setdefault(method, {'log_loss': [], 'brier': []})
                                for metric in bucket:
                                    bucket[metric].append(score[metric])
                            s, i, o = (predictions[m] for m in ('scalar','inferred_groups','grouped_oracle'))
                            signs['scalar_opposite'] += s*o < 0
                            signs['inferred_opposite'] += i*o < 0
                            signs['scalar_opposite_repaired'] += s*o < 0 and i*o > 0
                            for label,value in (('scalar',s),('inferred',i),('oracle',o)):
                                signs[label+'_zero'] += value == 0
                        yield record
                truth = raw['group_ids']
                errors = sum((groups[i] == groups[j]) != (truth[i] == truth[j])
                             for i in range(len(groups)) for j in range(i))
                pair_count = len(groups)*(len(groups)-1)//2
                runs.append({'condition': raw['name'], 'replicate': rep, 'fit': state,
                    'scores': {method: {metric: math.fsum(values)/len(values) for metric,values in bucket.items()}
                               for method,bucket in scores.items()}, 'sign_counts': signs,
                    'diagnostics': {'group_count': len(set(groups)), 'overlap': source['overlap'],
                        'exact_recovery': errors == 0 if raw['overlap'] > 0 and raw['signal_strength'] > 0 else None,
                        'pair_error': errors / pair_count if raw['overlap'] > 0 and raw['signal_strength'] > 0 and pair_count else None}})
    receipt = write_ledger(ledger_path, metadata=metadata, records=rows())
    if source_hashes() != hashes:
        raise RuntimeError('sources changed during experiment')
    return {'schema_version': 1, 'status': 'development_diagnostic', 'config': config,
            'python_version': platform.python_version(), 'source_sha256': hashes,
            'ledger_receipt': receipt, 'runs': runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','ledger'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if 'development' not in config['phase'] and args.lock is None:
        raise ValueError('final runs require a pre-execution lock')
    if args.lock:
        from scripts.source_inference_lock import verify_lock
        lock_hash = verify_lock(args.lock, args.config)
    result = run_experiment(config, args.ledger)
    if args.lock:
        result.update(status='locked_synthetic_evaluation', lock_sha256=lock_hash)
    result['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(f"Saved {len(result['runs'])} runs")


if __name__ == '__main__':
    main()
