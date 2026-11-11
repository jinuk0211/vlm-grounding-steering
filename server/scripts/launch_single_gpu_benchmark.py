"""Run one requested full cohort, adopting only compatible completed responses."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
DATA = ROOT / 'data/hallucination-suite'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(args):
    parent_pointer = ROOT / 'outputs' / ('pope-selected-current.json' if args.benchmark == 'pope' else 'hallucination-suite-current.json')
    parent = Path(json.loads(parent_pointer.read_text())['output'])
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    for name, digest in parent_protocol['source_hashes'].items():
        assert sha(ROOT / name) == digest
    gpu = 'GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {line.split(',')[0].strip(): int(line.split(',')[1]) for line in devices.splitlines()}
    assert used[gpu] < 1000, 'GPU0 is in use: inspect actual processes before relaunch'
    stamp = datetime.datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'outputs' / f'{args.benchmark}-selected-{stamp}'
    output.mkdir()
    for directory in ['data', 'claims', 'predictions', 'artifacts', 'source-snapshot', 'adopted']:
        (output / directory).mkdir()
    if args.benchmark == 'pope':
        rows = [r for r in read(DATA / 'pope.jsonl') if r['subset'] == 'adversarial']
        assert len(rows) == 3000 and len({r['inference_key'] for r in rows}) == 3000
        cap = 128
    else:
        rows = read(DATA / 'chair.jsonl')
        assert len(rows) == 500
        cap = args.caption_tokens
        rows = [{**row, 'max_new_tokens': cap} for row in rows]
    for benchmark, filename in [('pope', 'pope-unique.jsonl'), ('chair', 'chair.jsonl'), ('mmhal', 'mmhal.jsonl')]:
        values = rows if args.benchmark == benchmark else []
        (output / 'data' / filename).write_text(''.join(json.dumps(row) + '\n' for row in values))
    expected = {r['inference_key']: r for r in rows}
    adopted = {method: {} for method in parent_protocol['methods']}
    for file in sorted((parent / 'predictions').glob('*.jsonl')):
        for line in file.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row['benchmark'] != args.benchmark or row['inference_key'] not in expected:
                continue
            target = expected[row['inference_key']]
            assert row['protocol_sha256'] == parent_protocol['sha256']
            assert row['image'] == target['image'] and row['prompt'] == target['prompt']
            if row['max_new_tokens'] != cap:
                continue
            assert row['inference_key'] not in adopted[row['method']]
            adopted[row['method']][row['inference_key']] = row
    for method, cache in adopted.items():
        (output / 'adopted' / f'{method}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in cache.values()))
    config = json.loads((parent / 'config.json').read_text())
    save(output / 'config.json', config)
    for method in ('revis', 'grains', 'vti'):
        shutil.copy2(parent / 'artifacts' / f'{method}.pt', output / 'artifacts' / f'{method}.pt')
    files = ['src/vlm_grounding/backend.py', 'src/vlm_grounding/baselines.py', 'src/vlm_grounding/evaluation.py',
             'scripts/run_single_gpu_benchmark_worker.py', 'scripts/launch_single_gpu_benchmark.py']
    protocol = {**parent_protocol, 'selected_benchmark': args.benchmark, 'data_directory': str(output / 'data'),
                'evaluation_records_per_method': {args.benchmark: len(rows)},
                'pope_subset': 'adversarial' if args.benchmark == 'pope' else None,
                'max_new_tokens': cap, 'resource_policy': {'gpu_index': 0, 'gpu_uuid': gpu, 'automatic_other_gpu_fallback': False},
                'vector_device_cache': {'preserve_source_dtype': True, 'value_update_checks': 410, 'actual_generation_token_ids_identical': True}, 'parent_protocol_sha256': parent_protocol['sha256'],
                'parent_output': str(parent), 'source_hashes': {f: sha(ROOT / f) for f in files},
                'manifest_hashes': {p.name: sha(p) for p in (output / 'data').glob('*.jsonl')},
                'adopted_hashes': {p.name: sha(p) for p in (output / 'adopted').glob('*.jsonl')},
                'adopted_counts': {m: len(v) for m, v in adopted.items()},
                'user_scope_change': 'POPE adversarial only plus CHAIR first; other POPE types and MMHal deferred'}
    proof = json.loads((ROOT / 'outputs/gpu-vector-cache-generation-validation.json').read_text())
    assert proof['status'] == 'passed' and proof['all_generated_token_ids_identical']
    protocol['vector_device_cache']['actual_generation_proof'] = proof
    protocol.pop('sha256')
    protocol['generation'] = {**protocol['generation'], 'selected_max_new_tokens': cap}
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    save(output / 'protocol.json', protocol)
    for file in files:
        target = output / 'source-snapshot' / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / file, target)
    block = 60 if args.benchmark == 'pope' else 8
    tasks = [{'id': f'{args.benchmark}-{method}-{start:05d}', 'benchmark': args.benchmark,
              'method': method, 'start': start, 'stop': min(start + block, len(rows))}
             for start in range(0, len(rows), block) for method in protocol['methods']]
    save(output / 'tasks.json', tasks)
    jobs = []
    for slot in range(4):
        worker = f'gpu0-{slot}'
        log = output / f'{worker}.log'
        command = f'cd {ROOT}; source scripts/ljw_env.sh; exec python scripts/run_single_gpu_benchmark_worker.py --output {output} --worker {worker}'
        with log.open('ab') as sink:
            process = subprocess.Popen(['bash', '-lc', command], env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                                       stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({'worker': worker, 'pid': process.pid, 'gpu': 0, 'uuid': gpu, 'log': str(log)})
        save(output / 'jobs.json', jobs)
    pointer = {'output': str(output), 'jobs': jobs, 'protocol_sha256': protocol['sha256'], 'benchmark': args.benchmark}
    save(ROOT / 'outputs' / f'{args.benchmark}-selected-current.json', pointer)
    print(json.dumps({**pointer, 'adopted_counts': protocol['adopted_counts'], 'cases_per_method': len(rows), 'max_new_tokens': cap}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['pope', 'chair'], required=True)
    parser.add_argument('--caption-tokens', type=int, choices=[128, 512], default=512)
    main(parser.parse_args())
