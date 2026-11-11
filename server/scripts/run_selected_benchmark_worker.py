"""One batch-1 model per worker; durable atomic claims distribute full cohorts."""
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import torch

from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed, _validate_artifact, _versions

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(args):
    output = Path(args.output).resolve()
    assert output.is_relative_to(ROOT / 'outputs')
    status_path = output / f'worker-{args.worker}.json'
    state = {'worker': args.worker, 'pid': os.getpid(), 'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'],
             'started_at': time.time(), 'completed_tasks': 0, 'completed_records': 0}

    def update(**kwargs):
        state.update(kwargs, updated_at=time.time())
        save(status_path, state)

    try:
        update(stage='checking_protocol')
        config = json.loads((output / 'config.json').read_text())
        protocol = json.loads((output / 'protocol.json').read_text())
        content = {k: v for k, v in protocol.items() if k != 'sha256'}
        assert hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
        for name, expected in protocol['source_hashes'].items():
            assert sha(ROOT / name) == expected, f'Source changed: {name}'
        for name, expected in protocol['manifest_hashes'].items():
            assert sha(Path(protocol['data_directory']) / name) == expected
        datasets = {bench: read(Path(protocol['data_directory']) / filename)
                    for bench, filename in [('pope', 'pope-unique.jsonl'), ('chair', 'chair.jsonl'), ('mmhal', 'mmhal.jsonl')]}
        _seed(config)
        assert torch.cuda.device_count() == 1
        update(stage='loading_model')
        adapter = load_adapter(config['model'])
        adapter.model.config._commit_hash = config['model']['revision']
        artifacts = {'base': None, 'rudder': None}
        for method in ('revis', 'grains', 'vti'):
            path = output / 'artifacts' / f'{method}.pt'
            assert sha(path) == protocol['artifact_hashes'][method]
            artifact = torch.load(path, map_location='cpu', weights_only=True)
            _validate_artifact(artifact, config['model'], adapter=adapter)
            assert all(torch.isfinite(v).all() for v in artifact['vectors'].values())
            assert any(float(v.norm()) > 0 for v in artifact['vectors'].values())
            artifacts[method] = artifact
        update(stage='ready', gpu_name=torch.cuda.get_device_name(0), versions=_versions())
        tasks = json.loads((output / 'tasks.json').read_text())
        adopted = {}
        for cached_file in (output / 'adopted').glob('*.jsonl'):
            for cached_row in read(cached_file):
                adopted[(cached_row['method'], cached_row['inference_key'])] = cached_row
        for task in tasks:
            claim = output / 'claims' / task['id']
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            save(claim / 'owner.json', {'worker': args.worker, 'pid': os.getpid(), 'claimed_at': time.time()})
            method, benchmark = task['method'], task['benchmark']
            rows = datasets[benchmark][task['start']:task['stop']]
            assert rows and len(rows) == task['stop'] - task['start']
            update(stage='evaluating', task_id=task['id'], method=method, benchmark=benchmark,
                   task_completed=0, task_total=len(rows), task_started_at=time.time())
            path = output / 'predictions' / f"{task['id']}.jsonl"
            assert not path.exists(), 'Claimed task already has predictions; audit before retry'
            with path.open('x') as sink:
                for i, record in enumerate(rows):
                    assert sha(record['image']) == record['image_sha256']
                    if (method, record['inference_key']) in adopted:
                        cached = adopted[(method, record['inference_key'])]
                        assert cached['image'] == record['image'] and cached['prompt'] == record['prompt']
                        assert cached['max_new_tokens'] == record['max_new_tokens']
                        prediction = {**cached, **record, 'protocol_sha256': protocol['sha256'], 'adopted_from_protocol': cached['protocol_sha256']}
                        sink.write(json.dumps(prediction, allow_nan=False) + '\n')
                        sink.flush()
                        update(task_completed=i + 1, completed_records=state['completed_records'] + 1)
                        continue
                    _seed(config)
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    batch = adapter.prepare(record['image'], record['prompt'])
                    with build_baseline(method, adapter, artifacts[method], config['methods'][method]):
                        response = adapter.generate(batch, max_new_tokens=record['max_new_tokens'])
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    prediction = {**record, 'method': method, 'prediction': response,
                                  'generated_tokens': adapter.last_generated_tokens,
                                  'elapsed_seconds': elapsed, 'protocol_sha256': protocol['sha256'],
                                  'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'worker': args.worker,
                                  'max_gpu_allocated_gb': torch.cuda.max_memory_allocated() / 1e9}
                    sink.write(json.dumps(prediction, allow_nan=False) + '\n')
                    sink.flush()
                    del batch
                    update(task_completed=i + 1, completed_records=state['completed_records'] + 1,
                           last_seconds=elapsed, last_generated_tokens=adapter.last_generated_tokens,
                           last_memory_gb=prediction['max_gpu_allocated_gb'])
            save(claim / 'complete.json', {'worker': args.worker, 'pid': os.getpid(), 'records': len(rows),
                                         'prediction_sha256': sha(path), 'completed_at': time.time(), 'protocol_sha256': protocol['sha256']})
            update(completed_tasks=state['completed_tasks'] + 1)
            print(json.dumps({'task_complete': task['id'], 'records': len(rows), 'worker': args.worker}), flush=True)
        update(stage='complete', completed_at=time.time())
    except BaseException:
        update(stage='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', required=True)
    main(parser.parse_args())
