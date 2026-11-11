"""Run every MMVP item with the repository's four Gemma steering baselines."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

import torch
from huggingface_hub import try_to_load_from_cache

from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline, extract_baseline
from vlm_grounding.cli import _provenance, _seed, _validate_artifact, _versions
from vlm_grounding.data import read_manifest, validate_disjoint

REPO = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = REPO / 'data/mmvp-full'
METHODS = ('base', 'revis', 'rudder', 'grains', 'vti')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def choice(response):
    match = re.fullmatch(r'\s*\(?([AB])\)?[.\s]*', response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def score(rows):
    assert len(rows) == 300 and [r['index'] for r in rows] == list(range(1, 301))
    pairs = [rows[i:i + 2] for i in range(0, 300, 2)]
    corrections = {99: 'B', 100: 'A', 279: 'B', 280: 'A'}
    corrected = [r['choice'] == corrections.get(r['index'], r['target']) for r in rows]
    return {'questions': 300, 'pairs': 150,
            'correct_questions': sum(r['correct'] for r in rows),
            'question_accuracy': sum(r['correct'] for r in rows) / 300,
            'correct_pairs': sum(all(r['correct'] for r in pair) for pair in pairs),
            'pair_accuracy': sum(all(r['correct'] for r in pair) for pair in pairs) / 150,
            'invalid_responses': sum(r['choice'] is None for r in rows),
            'corrected_label_question_accuracy': sum(corrected) / 300,
            'corrected_label_pair_accuracy': sum(all(corrected[i:i + 2])
                                               for i in range(0, 300, 2)) / 150,
            'inference_seconds': sum(r['elapsed_seconds'] for r in rows),
            'generated_tokens': sum(r['generated_tokens'] for r in rows)}


def run(args):
    output = Path(args.output).resolve()
    assert output.is_relative_to(REPO / 'outputs')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'artifacts').mkdir(exist_ok=True)
    status_path = output / f'status-gpu{args.gpu}.json'
    status = {'pid': os.getpid(), 'gpu_index': args.gpu,
              'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
              'methods': args.methods, 'stage': 'starting', 'started_at': time.time()}

    def update(**values):
        status.update(values)
        status['updated_at'] = time.time()
        write_json(status_path, status)

    try:
        config = json.loads((output / 'config.json').read_text())
        protocol = json.loads((output / 'protocol.json').read_text())
        assert protocol['evaluation_records'] == 300 and protocol['evaluation_pairs'] == 150
        evaluation = read_jsonl(DATA / 'evaluation.jsonl')
        assert len(evaluation) == 300 and {r['index'] for r in evaluation} == set(range(1, 301))
        fit = read_manifest(DATA / 'fit.jsonl')
        calibration = read_manifest(DATA / 'calibration.jsonl')
        validate_disjoint([fit, calibration])
        assert len(fit) == len(calibration) == 32
        test_hashes = {r['image_sha256'] for r in evaluation}
        assert not test_hashes & {r['metadata']['image_sha256'] for r in fit + calibration}
        update(stage='loading_model', completed_methods=[])
        _seed(config)
        assert torch.cuda.device_count() == 1
        adapter = load_adapter(config['model'])
        # HF may omit _commit_hash in the deserialized Gemma config. Resolve it
        # from the actual pinned, fully cached snapshot used for this run.
        snapshot_config = try_to_load_from_cache(
            config['model']['checkpoint'], 'config.json',
            revision=config['model']['revision'], cache_dir=os.environ['HF_HUB_CACHE'],
        )
        resolved = Path(snapshot_config).parent.name
        assert resolved == config['model']['revision']
        adapter.model.config._commit_hash = resolved
        original_prepare = adapter.prepare
        prepare_calls = 0

        def prepare(*positional, **keywords):
            nonlocal prepare_calls
            prepare_calls += 1
            if status.get('stage', '').startswith('extracting'):
                update(prepare_calls=prepare_calls)
            return original_prepare(*positional, **keywords)

        adapter.prepare = prepare
        print(json.dumps({'model_loaded': True, 'gpu': torch.cuda.get_device_name(0),
                          'revision': resolved, 'methods': args.methods}), flush=True)
        for method in args.methods:
            artifact = None
            artifact_path = output / 'artifacts' / f'{method}.pt'
            prepare_calls = 0
            if method not in ('base', 'rudder'):
                update(stage=f'extracting_{method}', method=method, prepare_calls=0,
                       stage_started_at=time.time())
                if artifact_path.exists():
                    artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
                    _validate_artifact(artifact, config['model'], adapter=adapter)
                else:
                    records = fit + ([{**r, 'role': 'calibration'} for r in calibration]
                                     if method == 'revis' else [])
                    artifact = extract_baseline(method, adapter, records,
                                                config['methods'][method])
                    vectors = list(artifact['vectors'].values())
                    assert all(torch.isfinite(v).all() for v in vectors)
                    assert any(float(v.norm()) > 0 for v in vectors)
                    paths = [str(DATA / 'fit.jsonl')]
                    if method == 'revis':
                        paths.append(str(DATA / 'calibration.jsonl'))
                    artifact['_provenance'] = _provenance(adapter, config, paths, records)
                    artifact['gpu_verified'] = True
                    artifact['source_commit'] = protocol['source_commit']
                    torch.save(artifact, artifact_path)
                print(json.dumps({'artifact_ready': method, 'path': str(artifact_path),
                                  'seconds': time.time() - status['stage_started_at']}), flush=True)
            rows_path = output / f'{method}.jsonl'
            metadata_path = output / f'{method}.metadata.json'
            existing = read_jsonl(rows_path) if rows_path.exists() else []
            assert [r['index'] for r in existing] == list(range(1, len(existing) + 1))
            assert all(r['protocol_sha256'] == protocol['sha256'] for r in existing)
            update(stage=f'evaluating_{method}', method=method, completed=len(existing),
                   total=300, stage_started_at=time.time())
            metadata = {'status': 'running', 'method': method, 'cohort_complete': False,
                        'requested_limit': None, 'expected_records': 300,
                        'completed_records': len(existing), 'protocol': protocol,
                        'versions': _versions(), 'checkpoint_revision': resolved,
                        'implementation': 'base' if method == 'base' else 'paper-derived-local-adaptation',
                        'author_reproduction': False,
                        'artifact_sha256': sha(artifact_path) if artifact is not None else None,
                        'gpu': torch.cuda.get_device_name(0), 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES']}
            write_json(metadata_path, metadata)
            warmup = adapter.prepare(fit[0]['image'], fit[0]['prompt'])
            with build_baseline(method, adapter, artifact, config['methods'].get(method, {})):
                adapter.generate(warmup, max_new_tokens=2)
            with rows_path.open('a') as stream:
                for record in evaluation[len(existing):]:
                    batch = adapter.prepare(record['image'], record['prompt'])
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with build_baseline(method, adapter, artifact, config['methods'].get(method, {})):
                        response = adapter.generate(batch, config['generation']['max_new_tokens'])
                    torch.cuda.synchronize()
                    parsed = choice(response)
                    row = {**record, 'method': method, 'response': response, 'choice': parsed,
                           'correct': parsed == record['target'],
                           'generated_tokens': adapter.last_generated_tokens,
                           'elapsed_seconds': time.perf_counter() - started,
                           'protocol_sha256': protocol['sha256']}
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    existing.append(row)
                    if len(existing) % 10 == 0:
                        update(completed=len(existing))
                        print(json.dumps({'method': method, 'completed': len(existing),
                                          'total': 300}), flush=True)
            assert len(existing) == 300
            metrics = score(existing)
            metadata.update(status='complete', cohort_complete=True, completed_records=300,
                            metrics=metrics, peak_gpu_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
            write_json(metadata_path, metadata)
            update(completed=300, completed_methods=[*status['completed_methods'], method])
            print(json.dumps({'method_complete': method, 'metrics': metrics}), flush=True)
            torch.cuda.empty_cache()
        update(stage='complete')
    except BaseException as error:
        update(stage='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    run(arguments)
