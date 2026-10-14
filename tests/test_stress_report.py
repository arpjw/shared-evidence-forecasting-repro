"""Adversarial completeness checks for descriptive stress reporting and locks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from scripts.stress_report import build_report, summarize_result
from scripts.stress_lock import build_lock, verify_lock
from shared_evidence.experiment import run_experiment


def fixture():
    config = {'schema_version': 1, 'phase': 'test-stress', 'seed': 91,
              'train_count': 5, 'test_count': 7, 'replicates': 2,
              'conditions': [
                  {'name': 'exchangeable', 'n_forecasters': 2, 'signal_strength': .5, 'overlap': 1},
                  {'name': 'unequal', 'n_forecasters': 3, 'signal_strength': .5,
                   'overlap': 1, 'group_ids': [0, 0, 1]}]}
    result = run_experiment(config)
    result['config_sha256'] = hashlib.sha256(json.dumps(config).encode()).hexdigest()
    return result


class StressReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = fixture()

    def test_population_scalar_and_paired_regret(self):
        result = summarize_result(self.result, bootstrap_resamples=20)
        self.assertEqual(result['expected_runs'], 4)
        cell = result['cells']['unequal']
        self.assertEqual(cell['population_optimal_scalar'], 3/5)
        self.assertFalse(cell['scalar_can_represent_oracle'])
        rows = [r for r in self.result['runs'] if r['condition'] == 'unequal']
        expected = sum(r['scores']['learned']['log_loss'] - r['scores']['grouped_oracle']['log_loss'] for r in rows)/2
        self.assertAlmostEqual(cell['contrasts']['learned_minus_grouped_oracle']['log_loss']['mean'], expected)
        self.assertEqual(cell['contrasts']['learned_minus_grouped_oracle']['log_loss']['confidence_level'], .95)
        self.assertTrue(result['cells']['exchangeable']['scalar_can_represent_oracle'])

    def test_rejects_corrupt_and_incomplete_results(self):
        changes = [lambda r: r['runs'].pop(),
                   lambda r: r['runs'].append(copy.deepcopy(r['runs'][0])),
                   lambda r: r['runs'][0].update(discount=1.1),
                   lambda r: r['runs'][0].update(discount_boundary='false'),
                   lambda r: r['source_sha256'].pop('model.py'),
                   lambda r: r['runs'][0]['scores']['learned'].update(brier=1.1),
                   lambda r: r['runs'][0]['scores'].pop('oracle'),
                   lambda r: r['config'].update(test_count=True)]
        for change in changes:
            result = copy.deepcopy(self.result)
            change(result)
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize_result(result, bootstrap_resamples=10)

    def test_order_determinism_and_cached_comparison_ignored(self):
        result = copy.deepcopy(self.result)
        expected = summarize_result(result, bootstrap_resamples=20)
        result['runs'].reverse()
        for row in result['runs']:
            row['comparisons'] = {'fake': 1000}
        self.assertEqual(expected, summarize_result(result, bootstrap_resamples=20))

    def test_input_hashes_and_duplicate_phase_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            path.write_text(json.dumps(self.result))
            report = build_report([path], bootstrap_resamples=10)
            self.assertEqual(report['experiments']['test-stress']['input_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                build_report([path, path], bootstrap_resamples=10)

    def test_report_binds_lock_and_rejects_altered_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'shared_evidence').mkdir()
            (root / 'scripts').mkdir()
            for name in self.result['source_sha256']:
                (root / 'shared_evidence' / name).write_text(name)
            protocol = root / 'protocol.md'
            protocol.write_text('stress protocol')
            config = root / 'config.json'
            config.write_text(json.dumps(self.result['config']))
            lock = build_lock([config], protocol, root=root)
            lockpath = root / 'lock.json'
            lockpath.write_text(json.dumps(lock))
            result = copy.deepcopy(self.result)
            result['config_sha256'] = lock['config_sha256']['config.json']
            result['source_sha256'] = {name: lock['source_sha256']['shared_evidence/' + name]
                                      for name in result['source_sha256']}
            path = root / 'result.json'
            path.write_text(json.dumps(result))
            # Keep this test narrow while checking the public frozen setting.
            original = summarize_result
            def fast(value, **kwargs):
                return original(value, bootstrap_resamples=10)
            with patch('scripts.stress_report.summarize_result', side_effect=fast):
                report = build_report([path], lock_path=lockpath, root=root)
                self.assertEqual(report['stress_lock_sha256'], hashlib.sha256(lockpath.read_bytes()).hexdigest())
                with self.assertRaises(ValueError):
                    build_report([path], lock_path=lockpath, root=root, bootstrap_resamples=20)
                variants = [lambda x: x['config'].update(schema_version=True),
                            lambda x: x['config'].update(seed=92),
                            lambda x: x.update(config_sha256='0'*64),
                            lambda x: x['source_sha256'].update(**{'model.py': '0'*64}),
                            lambda x: x['config'].update(phase='unlocked-phase'),
                            lambda x: x.update(python_version='0.0')]
                for change in variants:
                    altered = copy.deepcopy(result)
                    change(altered)
                    path.write_text(json.dumps(altered))
                    with self.assertRaises(ValueError):
                        build_report([path], lock_path=lockpath, root=root)
                path.write_text(json.dumps(result))
                second = root / 'config2.json'
                second_config = dict(result['config'], phase='missing-phase')
                second.write_text(json.dumps(second_config))
                lockpath.write_text(json.dumps(build_lock([config, second], protocol, root=root)))
                with self.assertRaises(ValueError):
                    build_report([path], lock_path=lockpath, root=root)
            lock['analysis']['confidence_level'] = .975
            lockpath.write_text(json.dumps(lock))
            with self.assertRaises(ValueError):
                verify_lock(lockpath, root=root)

    def test_lock_detects_source_config_and_coverage_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'shared_evidence').mkdir()
            (root / 'scripts').mkdir()
            files = ['shared_evidence/model.py', 'scripts/stress_lock.py', 'scripts/stress_report.py', 'protocol.md']
            for name in files:
                (root/name).write_text('original')
            config = root/'config.json'
            config.write_text(json.dumps(self.result['config']))
            lock = build_lock([config], root/'protocol.md', root=root)
            path = root/'lock.json'
            path.write_text(json.dumps(lock))
            self.assertEqual(verify_lock(path, root=root), hashlib.sha256(path.read_bytes()).hexdigest())
            for name in [*files, 'config.json']:
                target = root/name
                original = target.read_bytes()
                target.write_text('changed')
                with self.assertRaises(ValueError):
                    verify_lock(path, root=root)
                target.write_bytes(original)
            (root/'shared_evidence/new.py').write_text('new source')
            with self.assertRaises(ValueError):
                verify_lock(path, root=root)


if __name__ == '__main__':
    unittest.main()
