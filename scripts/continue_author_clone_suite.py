"""Validate a cold pause, then continue only when --execute is explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-4a6e1799-a39c-86ed-abdf-afad03791c12'
SOURCES = ['scripts/continue_author_clone_suite.py',
           'scripts/continue_author_clone_entry.py',
           'scripts/continue_author_clone_worker.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def live(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0] not in ('Z', 'X')
    except FileNotFoundError:
        return False


def check(output):
    assert output.resolve().parent == ROOT / 'outputs'
    checkpoint = json.loads((output / 'PAUSED_CHECKPOINT.json').read_text())
    assert checkpoint['status'] == 'paused_processes_terminated'
    assert checkpoint['output'] == str(output) and checkpoint['gpu'] == GPU
    assert not checkpoint['invalid_partial_lines']
    for old in checkpoint['processes_before']:
        if live(old['pid']):
            fields = (Path('/proc') / str(old['pid']) / 'stat').read_text().split(') ', 1)[1].split()
            assert int(fields[19]) != old.get('start_ticks'), 'A paused process is still live'
    protocol = json.loads((output / 'protocol.json').read_text())
    assert sha(output / 'protocol.json') == checkpoint['protocol_sha256']
    for file, digest in protocol['source_hashes'].items():
        assert sha(ROOT / file) == digest, file
    amendment = json.loads((output / 'execution-amendment.json').read_text())
    for file, digest in amendment['source_hashes'].items():
        assert sha(ROOT / file) == digest, file
    old_source = (ROOT / 'scripts/run_author_clone_worker.py').read_text()
    new_source = (ROOT / 'scripts/continue_author_clone_worker.py').read_text()
    first, last = '                    key=(task', '                    sink.write'
    assert old_source[old_source.index(first):old_source.index(last)] == new_source[new_source.index(first):new_source.index(last)]
    sealed = {v['path']: v for v in checkpoint['prediction_files']}
    actual = {str(p.relative_to(output)): p for p in (output / 'predictions').glob('*.jsonl')}
    assert set(sealed) == set(actual)
    for name, file in actual.items():
        assert sha(file) == sealed[name]['sha256'], name
        assert not file.stat().st_size or file.read_bytes().endswith(b'\n'), name
    tasks = json.loads((output / 'tasks.json').read_text())
    datasets = {b: read(output / 'data' / (b + '.jsonl')) for b in ('pope', 'chair')}
    counts = {}; remaining = 0; prefixes = {}
    for task in tasks:
        file = output / 'predictions' / (task['id'] + '.jsonl')
        rows = read(file) if file.exists() else []
        refs = datasets[task['benchmark']][task['start']:task['stop']]
        assert len(rows) <= len(refs)
        for row, ref in zip(rows, refs):
            assert all(row.get(k) == v for k, v in ref.items())
            assert row['protocol_sha256'] == protocol['sha256']
            assert row['method'] == task['method'] and row['dtype'] == task['dtype']
            assert row['gpu_uuid'] == GPU
        marker = output / 'claims' / task['id'] / 'complete.json'
        if marker.exists():
            seal = json.loads(marker.read_text())
            assert len(rows) == len(refs) == seal['records'] and sha(file) == seal['sha256']
        else:
            remaining += len(refs) - len(rows)
            if rows:
                prefixes[task['id']] = len(rows)
        key = task['benchmark'] + ':' + task['method']
        counts[key] = counts.get(key, 0) + len(rows)
    assert counts == checkpoint['counts']
    return {'status': 'resume_check_passed', 'output': str(output), 'counts': counts,
            'remaining_responses': remaining, 'partial_prefixes': prefixes,
            'source_hashes': {f: sha(ROOT / f) for f in SOURCES},
            'generation_body_identical': True, 'models_launched': False,
            'checkpoint_sha256': sha(output / 'PAUSED_CHECKPOINT.json'), 'checked_at': time.time()}


def execute(output, checked):
    assert not (output / 'continuation-amendment.json').exists(), 'Inspect existing continuation instead of restarting'
    gpu_processes = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not gpu_processes, 'GPU0 is occupied; do not use another GPU automatically'
    previous_jobs = json.loads((output / 'jobs.json').read_text())
    save(output / 'pre-continuation-jobs.json', previous_jobs)
    save(output / 'continuation-amendment.json', {**checked,
         'pre_continuation_jobs_sha256': sha(output / 'pre-continuation-jobs.json'),
         'preserves_existing_response_bytes': True,
         'changes': 'Atomic task ownership and appending after a validated saved prefix; generation body unchanged'})
    (output / 'continuation-claims').mkdir()
    jobs = [j for j in previous_jobs if j['extract']]
    tag = str(int(time.time()))
    def launch(name, dtype):
        worker = name + '-' + tag
        with (output / (worker + '.log')).open('xb') as log:
            process = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/continue_author_clone_entry.py',
                '--output', str(output), '--worker', worker, '--dtype', dtype], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        job = {'worker': worker, 'pid': process.pid, 'dtype': dtype, 'extract': False,
               'log': str(output / (worker + '.log'))}
        jobs.append(job); save(output / 'jobs.json', jobs)
        return job
    pointer = json.loads((ROOT / 'outputs/author-clone-rerun-current.json').read_text())
    pointer['controller_pid'] = os.getpid(); pointer['log'] = str(output / 'continued-controller.log')
    save(ROOT / 'outputs/author-clone-rerun-current.json', pointer)
    bf16 = [launch('bf16-continuation-' + str(i), 'bfloat16') for i in range(3)]
    fp32 = [launch('fp32-continuation-0', 'float32')]
    try:
        while True:
            states = {}
            for job in jobs:
                file = output / ('extraction-status.json' if job['extract'] else 'worker-' + job['worker'] + '.json')
                state = json.loads(file.read_text()) if file.exists() else {'stage': 'starting'}
                active = live(job['pid']); states[job['worker']] = {**state, 'actual_live': active}
                if state['stage'] == 'failed' or (not active and state['stage'] != 'complete'):
                    raise RuntimeError(json.dumps({'job': job, 'state': state}))
            if len(fp32) == 1 and sum(live(j['pid']) for j in bf16) <= 1 and live(fp32[0]['pid']):
                fp32.append(launch('fp32-continuation-1', 'float32'))
            save(output / 'status.json', {'stage': 'generating', 'controller_pid': os.getpid(),
                'updated_at': time.time(), 'workers': states, 'continued_from_pause': True,
                'preserved_records': sum(checked['counts'].values()), 'total_records': 21000})
            if all(not live(j['pid']) for j in jobs):
                break
            time.sleep(15)
        save(output / 'status.json', {'stage': 'auditing', 'controller_pid': os.getpid(), 'updated_at': time.time()})
        with (output / 'audit.log').open('ab') as log:
            subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/audit_author_clone_suite.py', '--output', str(output)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((output / 'report.json').read_text())['status'] == 'verified_complete'
        save(output / 'status.json', {'stage': 'complete', 'controller_pid': os.getpid(), 'completed_at': time.time()})
    except BaseException:
        save(output / 'status.json', {'stage': 'failed', 'controller_pid': os.getpid(), 'error': traceback.format_exc(), 'updated_at': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert not (args.execute and args.check)
    result = check(args.output)
    save(args.output / 'resume-check.json', result)
    print(json.dumps(result), flush=True)
    if args.execute:
        execute(args.output, result)
