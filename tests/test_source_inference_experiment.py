"""Integration checks for complete replay and provenance failures."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared_evidence.source_inference_experiment import ROOT, run_experiment, validate_config
from shared_evidence.source_inference_verify import verify


class ExperimentTests(unittest.TestCase):
    def config(self):
        config=json.loads((ROOT/'configs/source-inference-development-v1.json').read_text())
        config.update(train_count=12,test_count=7,replicates=1)
        return config

    def test_replay_and_fit_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=Path(folder)/'ledger.gz'; summary=Path(folder)/'summary.json'
            result=run_experiment(self.config(),ledger)
            summary.write_text(json.dumps(result))
            receipt=verify(summary,ledger)
            self.assertEqual({k:receipt[k] for k in ('status','questions_checked','predictions_checked','fits_checked')},
                             {'status':'verified','questions_checked':19,'predictions_checked':42,'fits_checked':1})
            self.assertEqual(receipt['ledger_sha256'],result['ledger_receipt']['sha256'])
            result['runs'][0]['fit']['scalar']+=.1
            summary.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'numerical mismatch'):
                verify(summary,ledger)

    def test_resealed_false_fit_is_rejected(self):
        from shared_evidence.ledger import read_ledger, write_ledger
        with tempfile.TemporaryDirectory() as folder:
            ledger=Path(folder)/'ledger.gz'; summary=Path(folder)/'summary.json'
            result=run_experiment(self.config(),ledger)
            metadata,rows=read_ledger(ledger)
            key=next(iter(metadata['fits']))
            metadata['fits'][key]['source']['overlap']+=.1
            result['runs'][0]['fit']=copy.deepcopy(metadata['fits'][key])
            result['ledger_receipt']=write_ledger(ledger,metadata=metadata,records=rows)
            summary.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError,'fit.source.overlap'):
                verify(summary,ledger)

    def test_zero_signal_recovery_not_identifiable(self):
        config=self.config(); config['conditions'][0]['signal_strength']=0
        with tempfile.TemporaryDirectory() as folder:
            ledger=Path(folder)/'ledger.gz'; summary=Path(folder)/'summary.json'
            result=run_experiment(config,ledger)
            self.assertIsNone(result['runs'][0]['diagnostics']['exact_recovery'])
            summary.write_text(json.dumps(result)); verify(summary,ledger)

    def test_information_boundary(self):
        from shared_evidence.evaluate import ForecastRecord
        from shared_evidence.source_inference import fit_source_inference
        received=[]
        def checked_fit(rows):
            self.assertTrue(all(type(row) is ForecastRecord for row in rows))
            received.append(len(rows)); return fit_source_inference(rows)
        with tempfile.TemporaryDirectory() as folder, patch('shared_evidence.source_inference_experiment.fit_source_inference',checked_fit):
            run_experiment(self.config(),Path(folder)/'ledger.gz')
        self.assertEqual(received,[12])

    def test_reject_selection_and_ignored_parameter(self):
        for key,value in [('validation_count',12),('threshold_error_parameter',.1)]:
            config=self.config(); config[key]=value
            with self.assertRaises(ValueError): validate_config(config)


if __name__=='__main__': unittest.main()
