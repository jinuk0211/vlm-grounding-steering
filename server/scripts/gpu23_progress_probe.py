"""Observe committed rows and exact task process identities; never mutate a job."""
import collections
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde', 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'}
pointer = json.loads((ROOT / 'outputs/author-clone-rerun-current.json').read_text())
output = Path(pointer['output'])
assert output.resolve().parent == ROOT / 'outputs'


def observe(pid, script, worker=None):
    process = Path('/proc') / str(pid)
    try:
        fields = (process / 'stat').read_text().split(') ', 1)[1].split()
        state = fields[0]
        result = {'pid': pid, 'state': state, 'start_ticks': int(fields[19])}
        if state in ('Z', 'X'):
            return result
        if process.stat().st_uid != os.getuid():
            return {**result, 'state': 'pid_reused_unrelated'}
        command = [v.decode() for v in (process / 'cmdline').read_bytes().split(b'\0') if v]
        match = any(v.endswith('/' + script) or v == script for v in command)
        match = match and '--output' in command and command[command.index('--output') + 1] == str(output)
        if worker is not None:
            match = match and '--worker' in command and command[command.index('--worker') + 1] == worker
        if not match:
            return {**result, 'state': 'pid_reused_unrelated'}
        assert os.readlink(str(process / 'cwd')) == str(ROOT)
        visible = next((v.split(b'=', 1)[1].decode() for v in (process / 'environ').read_bytes().split(b'\0')
                        if v.startswith(b'CUDA_VISIBLE_DEVICES=')), None)
        return {**result, 'cuda': visible, 'identity_verified': True}
    except FileNotFoundError:
        return {'pid': pid, 'state': 'exited'}
    except PermissionError:
        # A process can enter kernel cleanup between stat and environ reads.
        return {'pid': pid, 'state': 'transition_unverified'}


status = json.loads((output / 'status.json').read_text())
live = []; states = []; transitions = []
for job in json.loads((output / 'jobs.json').read_text()):
    script = 'extract_author_grains.py' if job['extract'] else 'continue_gpu23_entry.py'
    process = observe(job['pid'], script, None if job['extract'] else job['worker'])
    file = output / ('extraction-status.json' if job['extract'] else 'worker-' + job['worker'] + '.json')
    state = json.loads(file.read_text()) if file.exists() else {'stage': 'starting'}
    if process.get('identity_verified'):
        assert not job['extract'], 'Unexpected active historical extraction process'
        assert process['cuda'] == job['visible_devices'] and set(process['cuda'].split(',')) <= ALLOWED
        live.append({'worker': job['worker'], **process})
    elif process['state'] == 'transition_unverified':
        transitions.append(process)
    states.append({'worker': job['worker'], 'pid': job['pid'], 'actual_state': process['state'],
                   'stage': state['stage'], 'last_seconds': state.get('last_seconds'),
                   'last_generated_tokens': state.get('last_generated_tokens'),
                   'task': state.get('task'), 'completed_records': state.get('completed_records'),
                   'error': state.get('error')})
counts = collections.Counter(); inflight = 0
for file in (output / 'predictions').glob('chair-*.jsonl'):
    data = file.read_bytes()
    if data and not data.endswith(b'\n'):
        boundary = data.rfind(b'\n') + 1
        inflight += len(data) - boundary
        data = data[:boundary]
    for line in data.splitlines():
        if line.strip():
            row = json.loads(line); counts[row['method']] += 1
controller = observe(pointer['controller_pid'], 'continue_gpu23_suite.py')
print(json.dumps({'stage': status['stage'], 'phase': status.get('phase'), 'error': status.get('error'),
    'controller_pid': pointer['controller_pid'], 'controller_state': controller['state'],
    'counts': dict(counts), 'live': live, 'workers': states,
    'transitions_unverified': transitions, 'inflight_tail_bytes': inflight,
    'gpu23': subprocess.check_output(['nvidia-smi', '-i', '2,3',
        '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True),
    'output': str(output), 'time': time.time()}))
