"""Freeze the complete source-inference suite before final outcomes exist."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from shared_evidence.source_inference_experiment import ROOT, source_hashes, validate_config


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(config_paths: list[Path], protocol: Path) -> dict:
    configs = {}
    for path in config_paths:
        config = json.loads(path.read_text())
        validate_config(config)
        if 'development' in config['phase']:
            raise ValueError('development configs cannot enter final lock')
        configs[str(path.resolve().relative_to(ROOT))] = digest(path)
    if not configs:
        raise ValueError('at least one final config required')
    return {'schema_version': 1, 'status': 'frozen_before_execution',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'source_sha256': source_hashes(),
        'config_sha256': configs, 'protocol_path': str(protocol.resolve().relative_to(ROOT)),
        'protocol_sha256': digest(protocol)}


def verify_lock(path: Path, config_path: Path) -> str:
    lock = json.loads(path.read_text())
    if lock.get('schema_version') != 1 or lock.get('status') != 'frozen_before_execution':
        raise ValueError('invalid lock')
    if lock['python_version'] != platform.python_version() or lock['source_sha256'] != source_hashes():
        raise ValueError('source or Python version changed since lock')
    if digest(ROOT / lock['protocol_path']) != lock['protocol_sha256']:
        raise ValueError('protocol changed since lock')
    for name, expected in lock['config_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('suite config changed since lock')
    name = str(config_path.resolve().relative_to(ROOT))
    if name not in lock['config_sha256']:
        raise ValueError('config absent from lock')
    return digest(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'docs/source_inference_protocol.md')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = build_lock(args.config, args.protocol)
    with args.output.open('x') as file:
        json.dump(lock, file, indent=2, sort_keys=True)
        file.write('\n')


if __name__ == '__main__':
    main()
