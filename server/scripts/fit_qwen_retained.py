"""Prepared Qwen-specific REVIS/VTI extraction through the selected native code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import torch
from author_clone_bridge import ROOT
from qwen_gpu23_adapter import GPU2, GPU3, load_qwen, require_gemma_complete, verify_run_protocol
from vlm_grounding.baselines import extract_baseline
from vlm_grounding.cli import _seed, _provenance, _validate_artifact


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(output, method):
    assert output.resolve().parent == ROOT / 'outputs'
    status = output / ('fit-' + method + '-status.json')
    state = {'pid': os.getpid(), 'method': method, 'started_at': time.time()}
    def update(**values):
        state.update(values, updated_at=time.time()); save(status, state)
    try:
        require_gemma_complete()
        assert os.environ['CUDA_VISIBLE_DEVICES'] in (GPU2, GPU3)
        protocol = verify_run_protocol(output)
        config = json.loads((output / 'config.json').read_text())
        config['model']['dtype'] = 'bfloat16'
        _seed(config); update(stage='loading_model')
        adapter = load_qwen(config['model'], output, 'fit-' + method)
        fit_path, cal_path = output / 'data/fit.jsonl', output / 'data/calibration.jsonl'
        fit = [json.loads(s) for s in fit_path.read_text().splitlines() if s.strip()]
        calibration = [json.loads(s) for s in cal_path.read_text().splitlines() if s.strip()]
        assert len(fit) == len(calibration) == 32
        records = fit + ([{**r, 'role': 'calibration'} for r in calibration] if method == 'revis' else [])
        for record in records:
            assert sha(record['image']) == record['metadata']['image_sha256']
        update(stage='extracting', extraction_records=len(records))
        artifact = extract_baseline(method, adapter, records, config['methods'][method])
        for group in ('vectors', 'vision_vectors'):
            if group in artifact:
                assert all(torch.isfinite(v).all() for v in artifact[group].values())
        assert any(v.norm() > 0 for v in artifact['vectors'].values())
        paths = [str(fit_path)] + ([str(cal_path)] if method == 'revis' else [])
        artifact['_provenance'] = _provenance(adapter, config, paths, records)
        artifact['gpu_verified'] = True
        artifact['source_commit'] = protocol['source_commit']
        _validate_artifact(artifact, config['model'], adapter=adapter)
        path = output / 'artifacts' / (method + '.pt')
        assert not path.exists()
        torch.save(artifact, path)
        save(output / (method + '-artifact.json'), {
            'status': 'verified_complete', 'artifact_sha256': sha(path), 'method': method,
            'model': config['model'], 'records': [r['id'] for r in records],
            'input_hashes': {p: sha(p) for p in paths}, 'source_protocol_sha256': protocol['sha256'],
            'gpu_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'gemma_vectors_reused': False})
        update(stage='complete', artifact_sha256=sha(path), completed_at=time.time())
    except BaseException:
        update(stage='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=('revis', 'vti'), required=True)
    args = parser.parse_args()
    main(args.output, args.method)
