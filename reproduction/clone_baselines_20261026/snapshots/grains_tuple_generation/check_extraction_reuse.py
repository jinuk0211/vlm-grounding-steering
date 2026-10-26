"""Reject incomplete or altered extraction before loading a generation model."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_generation as run

sys.path.insert(0, str(run.UPSTREAM / 'GrAInS/src'))
from utils.steering import compute_steering_vector  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fit = self.root / 'results'
        extraction = self.fit / 'extraction'
        extraction.mkdir(parents=True)
        rng = np.random.default_rng(4)
        records, hidden, rows = [], {}, []
        previous = None
        for index in range(50):
            record = {'id': f'fit:{index}'}
            records.append(record)
            artifact = extraction / f'example_{index:03}.pt'
            raw = extraction / f'example_{index:03}_ig.pt'
            values = {key: rng.normal(size=(29, 8)).astype(np.float32) for key in
                      ['hidden_pos_response', 'hidden_neg_response',
                       'hidden_pos_response_ablated', 'hidden_neg_response_ablated']}
            torch.save(values, artifact)
            torch.save({'positive_ig': torch.ones(3, 8), 'negative_ig': -torch.ones(2, 8)}, raw)
            for key, value in values.items():
                hidden.setdefault(key, []).append(value)
            payload = {**record, 'artifact': str(artifact), 'sha256': sha(artifact),
                       'raw_ig': str(raw), 'raw_ig_sha256': sha(raw), 'previous_sha256': previous}
            previous = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                  allow_nan=False).encode()).hexdigest()
            rows.append({**payload, 'row_sha256': previous})
        (self.fit / 'extraction.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (self.fit / 'extraction_progress.json').write_text(json.dumps({'completed': 50}))
        self.sources = {'fixture': 'sealed'}
        header = {'source_sha256': self.sources, 'fit_records': records,
                  'model': run.parent.MODEL, 'grains': {'ig_steps': 20}}
        (self.fit / 'header.json').write_text(json.dumps(header))
        hidden = {key: np.stack(values) for key, values in hidden.items()}
        np.savez(self.fit / 'original_hidden_states.npz', **hidden)
        np.random.seed(42)
        torch.save(compute_steering_vector(hidden, 27, 'pca'), self.fit / 'original_pca_vector.pt')
        (self.root / 'execution_manifest_compat_20261026.json').write_text('{}')
        (self.root / 'provenance.json').write_text(json.dumps({
            'parent_manifest_sha256': sha(self.root / 'execution_manifest_compat_20261026.json'),
            'extraction_header_sha256': sha(self.fit / 'header.json'),
        }))
        for target, value in [('ROOT', self.root), ('UPSTREAM', self.root), ('FIT', self.fit)]:
            patcher = patch.object(run, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for obj, name in [(run, 'verify'), (run.parent, 'verify')]:
            patcher = patch.object(obj, name, return_value=self.sources)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_all_fifty_artifacts_reconstruct_npz_and_original_pca(self):
        header, vector, artifacts = run.validate_extraction()
        self.assertEqual(len(header['fit_records']), 50)
        self.assertEqual(len(artifacts), 104)
        self.assertEqual(vector.shape, (8,))

    def test_changed_raw_artifact_is_rejected_before_deserialization(self):
        artifact = self.fit / 'extraction/example_000.pt'
        artifact.write_bytes(b'invalid changed pickle')
        with self.assertRaisesRegex(RuntimeError, 'artifact path/SHA mismatch'):
            run.validate_extraction()

    def test_forty_nine_examples_cannot_be_reused_as_fifty(self):
        journal = self.fit / 'extraction.jsonl'
        journal.write_text('\n'.join(journal.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'all 50 completed'):
            run.validate_extraction()


if __name__ == '__main__':
    unittest.main(verbosity=2)
