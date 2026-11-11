"""Prepared Qwen suite worker: all rows fresh, guarded, with resumable prefixes."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

import torch
from author_clone_bridge import ROOT, load_authors, verify_clone_sources, rudder_context
from qwen_author_compat import grains_on_qwen_layer_device
from qwen_five_profile_api import GPU2, GPU3, PAIR, load_qwen, require_gemma_complete, verify_run_protocol
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed, _validate_artifact, _versions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(args):
    output = args.output
    assert output.resolve().parent == ROOT / 'outputs'
    state = {'pid': os.getpid(), 'worker': args.worker, 'dtype': args.dtype,
             'started_at': time.time(), 'completed_records': 0}
    def update(**values):
        state.update(values, updated_at=time.time()); save(output / ('worker-' + args.worker + '.json'), state)
    try:
        require_gemma_complete()
        assert re.fullmatch('[a-z0-9-]+', args.lease_epoch)
        visible = os.environ['CUDA_VISIBLE_DEVICES']
        assert visible in (GPU2, GPU3) and args.dtype in ('bfloat16','float16')
        protocol = verify_run_protocol(output)
        assert verify_clone_sources() == protocol['author_sources']
        config = json.loads((output / 'config.json').read_text())
        config['model']['dtype'] = args.dtype
        _seed(config); update(stage='loading_model')
        adapter = load_qwen(config['model'], output, args.worker)
        authors = load_authors()
        seals = json.loads((output / 'artifacts-ready.json').read_text())
        assert seals['status'] == 'verified_complete' and seals['source_protocol_sha256'] == protocol['sha256']
        artifacts = {}; vector = None
        if args.dtype == 'bfloat16':
            for method in ('revis', 'vti'):
                path = output / 'artifacts' / (method + '.pt')
                assert sha(path) == seals['methods'][method]['sha256']
                artifact = torch.load(path, map_location='cpu', weights_only=True)
                _validate_artifact(artifact, config['model'], adapter=adapter)
                for group in ('vectors', 'vision_vectors'):
                    if group in artifact:
                        artifact[group] = {k: v.to(adapter.device) for k, v in artifact[group].items()}
                artifacts[method] = artifact
        else:
            path = output / 'grains-author.pt'
            assert sha(path) == seals['methods']['grains_clone']['sha256']
            artifact = torch.load(path, map_location='cpu', weights_only=True)
            assert artifact['model'] == {**config['model'], 'dtype': 'float32'} and not artifact['gemma_vectors_reused']
            assert artifact['author_commit'] == protocol['author_sources']['GrAInS']['commit']
            vector = artifact['vector'].to(adapter.device)
            assert vector.dtype == torch.float32, 'Keep the original author vector dtype'
        datasets = {b: read(output / 'data' / (b + '.jsonl')) for b in ('pope', 'chair')}
        tasks = json.loads((output / 'tasks.json').read_text())
        lease_root = output / 'leases' / args.lease_epoch; lease_root.mkdir(parents=True, exist_ok=True)
        update(stage='ready', versions=_versions(), gpu_visible_devices=visible)
        for task in tasks:
            if task['dtype'] != args.dtype:
                continue
            claim = output / 'claims' / task['id']
            path = output / 'predictions' / (task['id'] + '.jsonl')
            marker = claim / 'complete.json'
            if marker.exists():
                complete = json.loads(marker.read_text())
                assert complete['sha256'] == sha(path) and complete['records'] == task['stop'] - task['start']
                continue
            try:
                (lease_root / task['id']).mkdir()
            except FileExistsError:
                continue
            claim.mkdir(exist_ok=True)
            references = datasets[task['benchmark']][task['start']:task['stop']]
            prefix = read(path) if path.exists() else []
            assert len(prefix) <= len(references)
            assert not prefix or path.read_bytes().endswith(b'\n')
            for row, ref in zip(prefix, references):
                assert all(row.get(k) == v for k, v in ref.items())
                assert row['method'] == task['method'] and row['dtype'] == args.dtype
                assert row['protocol_sha256'] == protocol['sha256'] and row['model_family'] == 'qwen2_5_vl'
                assert row['checkpoint'] == config['model']['checkpoint'] and row['revision'] == config['model']['revision']
                assert row['artifact_sha256'] == seals['methods'].get(task['method'], {}).get('sha256')
                assert row['gpu_uuid'] in (GPU2, GPU3)
            save(claim / 'owner.json', {'worker': args.worker, 'pid': os.getpid(), 'started_at': time.time(),
                                      'lease_epoch': args.lease_epoch, 'preserved_prefix_records': len(prefix)})
            update(stage='generating', task=task['id'], benchmark=task['benchmark'], method=task['method'], task_completed=len(prefix))
            with path.open('a' if path.exists() else 'x') as sink:
                for index, record in enumerate(references[len(prefix):], start=len(prefix)):
                    assert sha(record['image']) == record['image_sha256']; _seed(config)
                    batch = adapter.prepare(record['image'], record['prompt']); diagnostics = {}
                    if task['method'] == 'rudder_clone':
                        context = rudder_context(adapter, batch, task['benchmark'], authors, diagnostics)
                    elif task['method'] == 'grains_clone':
                        context = grains_on_qwen_layer_device(adapter, vector, authors, diagnostics)
                    elif task['method'] in ('revis', 'vti'):
                        context = build_baseline(task['method'], adapter, artifacts[task['method']], config['methods'][task['method']])
                    else:
                        context = contextlib.nullcontext()
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with context:
                        response = adapter.generate(batch, max_new_tokens=record['max_new_tokens'])
                    torch.cuda.synchronize()
                    if task['method'] in ('rudder_clone', 'grains_clone'):
                        assert diagnostics['calls'] > 0
                    row = {**record, 'method': task['method'], 'prediction': response,
                        'generated_tokens': adapter.last_generated_tokens, 'elapsed_seconds': time.perf_counter() - start,
                        'protocol_sha256': protocol['sha256'], 'worker': args.worker, 'dtype': args.dtype,
                        'gpu_uuid': visible, 'model_family': 'qwen2_5_vl',
                        'checkpoint': config['model']['checkpoint'], 'revision': config['model']['revision'],
                        'author_diagnostics': diagnostics,
                        'artifact_sha256': seals['methods'].get(task['method'], {}).get('sha256'),
                        'max_gpu_allocated_gb': [torch.cuda.max_memory_allocated(i)/1e9 for i in range(torch.cuda.device_count())]}
                    del batch
                    sink.write(json.dumps(row, allow_nan=False) + '\n'); sink.flush()
                    update(completed_records=state['completed_records'] + 1, task_completed=index + 1,
                           last_seconds=row['elapsed_seconds'], last_generated_tokens=row['generated_tokens'])
            save(marker, {'worker': args.worker, 'records': len(references), 'sha256': sha(path), 'completed_at': time.time()})
        assert verify_clone_sources() == protocol['author_sources']
        update(stage='complete', completed_at=time.time())
    except BaseException:
        update(stage='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), required=True)
    parser.add_argument('--lease-epoch', required=True)
    main(parser.parse_args())
