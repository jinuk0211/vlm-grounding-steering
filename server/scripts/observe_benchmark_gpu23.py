"""Bounded CPU observation of actual task instances, never starts GPU work."""
import argparse
import collections
import json
import os
from pathlib import Path
import time

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
GEMMA = ROOT / 'outputs/author-clone-rerun-20261027-165446'
GPU2 = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3 = 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'

def read(path):
    return json.loads(Path(path).read_text())

def identity(job, script=None, output=None, cpu_only=False):
    pid = job.get('pid', job.get('controller_pid'))
    expected = job.get('start_ticks', job.get('controller_start_ticks'))
    assert pid is not None and expected is not None, job
    path = Path('/proc') / str(pid)
    result = {'pid':pid, 'start_ticks':expected, 'live':False}
    try:
        fields = (path / 'stat').read_text().split(') ',1)[1].split()
        if fields[0] in ('Z','X') or int(fields[19]) != expected:
            return result
        assert path.stat().st_uid == os.getuid()
        assert os.readlink(path / 'cwd') == str(ROOT)
        args = (path / 'cmdline').read_bytes().decode().split('\0')
        if script is not None:assert script in args
        if output is not None:assert str(output) in args
        env = (path / 'environ').read_bytes().split(b'\0')
        visible = next((s.decode().split('=',1)[1] for s in env if s.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
        if cpu_only:assert visible == ''
        if job.get('visible_devices'):
            assert visible == job['visible_devices']
            assert set(visible.split(',')) <= {GPU2,GPU3}
        result.update(live=True, process_state=fields[0], visible_devices=visible)
    except FileNotFoundError:
        pass
    return result

def counts(output):
    totals = collections.Counter()
    tasks = output / 'tasks.json'
    if not tasks.exists():return {}
    for task in read(tasks):
        if task['benchmark'] != 'chair':continue
        path = output / 'predictions' / (task['id'] + '.jsonl')
        if not path.exists():continue
        raw = path.read_bytes()
        raw = raw[:raw.rfind(b'\n') + 1]
        rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
        assert len(rows) <= task['stop'] - task['start']
        assert all(row['method'] == task['method'] for row in rows)
        totals[task['method']] += len(rows)
    return dict(totals)

def snapshot(include_counts=False):
    state = read(GEMMA / 'status.json')
    controller = identity(state, 'scripts/resume_gpu23_suite_20261029.py', GEMMA, True)
    assert state['stage'] != 'failed', state
    assert controller['live'] or state['stage'] == 'complete', state
    workers = []
    for job in read(GEMMA / 'jobs.json'):
        if job.get('resume_epoch') != state['resume_epoch']:continue
        proof = identity(job, 'scripts/resume_gpu23_entry_20261029.py', GEMMA)
        file = GEMMA / ('worker-' + job['worker'] + '.json')
        ws = read(file) if file.exists() else {}
        assert ws.get('stage') != 'failed', ws
        workers.append(dict(proof, worker=job['worker'], stage=ws.get('stage'),
            completed=ws.get('completed_records'), method=ws.get('method'), last_seconds=ws.get('last_seconds')))
    pointer = read(ROOT / 'outputs/qwen-author-suite-current.json')
    qout = Path(pointer['output'])
    qs = read(qout / 'status.json')
    qcontroller = identity(qs, 'scripts/run_qwen_author_suite.py', qout, True)
    assert qs['stage'] != 'failed', qs
    assert qcontroller['live'] or qs['stage'] == 'complete', qs
    result = {'gemma_stage':state['stage'], 'gemma_phase':state.get('phase'),
        'gemma_controller':controller, 'gemma_workers':workers,
        'qwen_stage':qs['stage'], 'qwen_controller':qcontroller,
        'qwen_gpu_started':qs.get('qwen_gpu_models_launched'),
        'qwen_output':str(qout), 'qwen_jobs':qs.get('jobs'), 'checked_at':time.time()}
    if include_counts:
        result['gemma_chair_counts'] = counts(GEMMA)
        result['gemma_chair_total'] = sum(result['gemma_chair_counts'].values())
        result['qwen_chair_counts'] = counts(qout)
        watch = GEMMA / 'bf16-score-watch-status.json'
        if watch.exists():
            ws=read(watch)
            result['bf16_scorer']={k:ws.get(k) for k in ['stage','report','output','archive','counts','error'] if k in ws}
    return result

def main(seconds):
    assert 0 <= seconds <= 45 and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    before = snapshot(True)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(min(5, max(0,deadline-time.monotonic())))
        current = snapshot()
        if (current['gemma_stage'],current['gemma_phase'],current['qwen_stage']) != (before['gemma_stage'],before['gemma_phase'],before['qwen_stage']):break
    after = snapshot(True)
    after['observed_seconds'] = seconds
    after['gemma_new_captions_during_observation'] = after['gemma_chair_total'] - before['gemma_chair_total']
    destination = ROOT / 'outputs/benchmark-gpu23-observation-latest.json'
    tmp = destination.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(after,indent=2,allow_nan=False)+'\n');tmp.replace(destination)
    print(json.dumps(after,allow_nan=False))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=int,default=45)
    main(parser.parse_args().seconds)
