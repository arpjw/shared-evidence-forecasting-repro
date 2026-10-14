"""Create and verify a separate exploratory stress-suite lock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(config_paths: Sequence[Path], protocol: Path, *, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    paths = [path.resolve() for path in config_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('configs must be nonempty and unique')
    phases = []
    configs = {}
    for path in paths:
        name = str(path.relative_to(root))
        config = json.loads(path.read_bytes())
        phase = config.get('phase')
        if not isinstance(phase, str) or not phase or phase == 'production-primary-v1':
            raise ValueError('stress config must have a separate nonempty phase')
        phases.append(phase)
        configs[name] = _hash(path)
    if len(set(phases)) != len(phases):
        raise ValueError('stress phases must be unique')
    protocol = protocol.resolve()
    protocol_name = str(protocol.relative_to(root))
    sources = sorted((root / 'shared_evidence').glob('*.py'))
    sources.extend(sorted((root / 'scripts').glob('*.py')))
    return {'schema_version': 1, 'status': 'exploratory_after_primary',
            'freeze_stage': 'before_stress_execution', 'python_version': platform.python_version(),
            'analysis': {'bootstrap_resamples': 10000, 'confidence_level': 0.95,
                         'scope': 'descriptive_per_cell', 'bootstrap_method': 'percentile',
                         'resampling_unit': 'independent_training_and_test_replicate'},
            'config_sha256': dict(sorted(configs.items())),
            'protocol_sha256': {protocol_name: _hash(protocol)},
            'source_sha256': {str(path.relative_to(root)): _hash(path) for path in sources},
            'interpretation': 'Primary outcome was already inspected. This lock preserves subsequent exploratory design and code; it is not a new confirmatory registration.'}


def verify_lock(path: Path, *, root: Path = ROOT) -> str:
    raw = path.read_bytes()
    locked = json.loads(raw)
    if locked.get('schema_version') != 1 or locked.get('status') != 'exploratory_after_primary' or locked.get('freeze_stage') != 'before_stress_execution':
        raise ValueError('invalid exploratory stress lock')
    if locked.get('python_version') != platform.python_version():
        raise ValueError('Python version changed after lock')
    protocols = locked.get('protocol_sha256', {})
    if len(protocols) != 1:
        raise ValueError('lock must bind one protocol')
    root = root.resolve()
    for mapping in ('config_sha256', 'protocol_sha256', 'source_sha256'):
        if not isinstance(locked.get(mapping), dict) or not locked[mapping]:
            raise ValueError('missing locked file mapping')
        for name, digest in locked[mapping].items():
            target = (root / name).resolve()
            target.relative_to(root)
            if _hash(target) != digest:
                raise ValueError(f'{name} changed after lock')
    rebuilt = build_lock([root / name for name in locked['config_sha256']],
                         root / next(iter(protocols)), root=root)
    if rebuilt != locked:
        raise ValueError('lock metadata or file coverage changed')
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(verify_lock(args.verify))
        return
    if not args.config or args.protocol is None or args.output is None:
        parser.error('creation requires --config, --protocol, and --output')
    lock = build_lock(args.config, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of pre-run evidence.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(f'Saved stress lock to {args.output}')


if __name__ == '__main__':
    main()
