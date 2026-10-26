"""Reuse sealed clone extraction; evaluate the original tuple-based hook."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from tuple_bridge import original_tuple_hook


ROOT = Path('/workspace/original-baselines-20261025/grains_generation_contract_20261026')
UPSTREAM = Path('/workspace/original-baselines-20261025/grains_gpu_retry_20261026')
FIT = UPSTREAM / 'runs_checkpoint/grains'
sys.path.insert(0, str(UPSTREAM / 'operations'))
import run_clone_benchmark_20261026 as parent  # noqa: E402

SEAL = hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify():
    if parent.common.digest(ROOT / 'manifest.json') != SEAL:
        raise RuntimeError('Continuation manifest changed')
    sources = read(ROOT / 'manifest.json')
    for name, sha in sources.items():
        if parent.common.digest(ROOT / name) != sha:
            raise RuntimeError('Continuation source changed: ' + name)
    parent.verify()
    return sources


def validate_extraction():
    verify()
    config = read(ROOT / 'provenance.json')
    if parent.common.digest(UPSTREAM / 'execution_manifest_compat_20261026.json') != config['parent_manifest_sha256']:
        raise RuntimeError('Upstream execution manifest changed')
    if parent.common.digest(FIT / 'header.json') != config['extraction_header_sha256']:
        raise RuntimeError('Original extraction header changed')
    header = read(FIT / 'header.json')
    if header['source_sha256'] != parent.verify() or len(header['fit_records']) != 50:
        raise RuntimeError('Extraction source/input header mismatch')
    if header['model'] != parent.MODEL or header['grains']['ig_steps'] != 20:
        raise RuntimeError('Extraction model or IG settings changed')
    previous, artifacts, hidden = None, {}, {}
    lines = (FIT / 'extraction.jsonl').read_text().splitlines()
    if len(lines) != 50 or read(FIT / 'extraction_progress.json')['completed'] != 50:
        raise RuntimeError('Requires all 50 completed extraction examples')
    for index, line in enumerate(lines):
        row = json.loads(line)
        payload = {k: v for k, v in row.items() if k != 'row_sha256'}
        sha = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False).encode()).hexdigest()
        if row['previous_sha256'] != previous or row['row_sha256'] != sha:
            raise RuntimeError('Extraction journal hash chain mismatch')
        previous = sha
        if any(row.get(k) != v for k, v in header['fit_records'][index].items()):
            raise RuntimeError('Extraction input order changed')
        for key, digest_key in [('artifact', 'sha256'), ('raw_ig', 'raw_ig_sha256')]:
            path = Path(row[key])
            expected = FIT / 'extraction' / f"example_{index:03}{'_ig' if key == 'raw_ig' else ''}.pt"
            if path != expected or parent.common.digest(path) != row[digest_key]:
                raise RuntimeError('Extraction artifact path/SHA mismatch')
            artifacts[str(path)] = row[digest_key]
        ig = torch.load(row['raw_ig'], map_location='cpu', weights_only=True)
        if not all(torch.isfinite(ig[k]).all().item() for k in ['positive_ig', 'negative_ig']):
            raise RuntimeError('Nonfinite original IG artifact')
        # These are this run's trusted, SHA-verified NumPy-containing outputs.
        values = torch.load(row['artifact'], map_location='cpu', weights_only=False)
        for key, value in values.items():
            if key.startswith('hidden_'):
                if not np.isfinite(value).all():
                    raise RuntimeError('Nonfinite hidden state artifact')
                hidden.setdefault(key, []).append(value)
    hidden = {k: np.stack(v) for k, v in hidden.items()}
    with np.load(FIT / 'original_hidden_states.npz', allow_pickle=False) as stored:
        if set(stored.files) != set(hidden) or any(not np.array_equal(stored[k], v) for k, v in hidden.items()):
            raise RuntimeError('NPZ does not match all 50 journaled raw artifacts')
    sys.path.insert(0, str(UPSTREAM / 'GrAInS/src'))
    from utils.steering import compute_steering_vector

    np.random.seed(42)
    expected = compute_steering_vector(hidden, 27, 'pca')
    vector = torch.load(FIT / 'original_pca_vector.pt', map_location='cpu', weights_only=True)
    torch.testing.assert_close(vector, expected, rtol=1e-5, atol=1e-6)
    for name in ['header.json', 'extraction.jsonl', 'original_hidden_states.npz', 'original_pca_vector.pt']:
        artifacts[str(FIT / name)] = parent.common.digest(FIT / name)
    return header, vector, artifacts


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        header, vector, artifacts = validate_extraction()
        parent.common.write_json(args.output / 'header.json', {
            **header, 'source_sha256': verify(), 'extraction_source_sha256': header['source_sha256'],
            'extraction_artifact_sha256': artifacts,
            'decoder_output_contract': 'Native Tensor -> original tuple hook -> native Tensor',
            'scope': 'Original clone formula, reusing sealed fit50; original tuple-based normalization',
        })
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        from utils.model import load_vlm_model_and_processor
        from utils.steering import add_steering_hook

        model, processor = load_vlm_model_and_processor(parent.MODEL)
        parent.common.write_json(args.output / 'loaded_model.json', {
            'device': str(model.device), 'dtype': str(next(model.parameters()).dtype),
            'hf_device_map': model.hf_device_map,
        })
        if model.device.type != 'cuda' or next(model.parameters()).dtype != torch.float32:
            raise RuntimeError('Expected original FP32 model on the existing GPU')
        for index, record in enumerate(header['test_records']):
            if parent.common.digest(record['image']) != header['image_sha256'][record['id']]:
                raise RuntimeError('Test image changed')
            for condition in ['base', 'grains']:
                if condition == 'base':
                    result = parent.grains_generate(model, processor, (vector, add_steering_hook), record, 'base')
                else:
                    with original_tuple_hook(model, 27, vector, 10.0, add_steering_hook):
                        # The original hook is already active; reuse the exact native input/generation adapter.
                        result = parent.grains_generate(model, processor, (vector, add_steering_hook), record, 'base')
                row = {**record, **result, 'method': condition}
                reasons = parent.failure_reasons(row, 256)
                row['health_failures'] = reasons
                row['output_checks'] = {**parent.common.output_checks(row, 256),
                                        'repeated': any('repetition' in reason for reason in reasons)}
                parent.common.append(args.output / 'samples.jsonl', row)
                rows.append(row)
            parent.common.write_json(args.output / 'generation_progress.json', {'completed_pairs': index + 1, 'of': 100})
            if (index + 1) % 10 == 0:
                verify()
                parent.score(rows, args, index + 1)
                verify()
        if any(parent.common.digest(path) != sha for path, sha in artifacts.items()):
            raise RuntimeError('Extraction artifacts changed during generation')
        verify()
        parent.common.write_json(args.output / 'complete.json', {
            'completed_pairs': 100, 'raw_rows': len(rows), 'full_reproduction_complete': False,
        })
    except Exception as error:
        parent.common.write_json(args.output / 'failure.json', {
            'error': repr(error), 'raw_rows': len(rows), 'completed_pairs': len(rows) // 2,
        })
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.method = 'grains'
    args.chair = Path('/workspace/vlm-baselines-20261024/upstream/chair/chair.py')
    args.annotations = parent.OLD / 'annotations'
    main(args)
