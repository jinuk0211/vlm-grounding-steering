"""Wait for the existing clone run to release its GPU, then continue generation."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/original-baselines-20261025/grains_generation_contract_20261026')
UPSTREAM = Path('/workspace/original-baselines-20261025/grains_gpu_retry_20261026/runs_checkpoint')
lock = (ROOT / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
expected = json.loads((ROOT / 'manifest.json').read_text())
manifest_sha256 = hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest()
child = None
interrupted = 0


def status(phase, **extra):
    value = {'at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
             'queue_pid': os.getpid(), **extra}
    temporary = ROOT / 'queue_status.tmp'
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(ROOT / 'queue_status.json')
    with (ROOT / 'queue_events.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def verify():
    if hashlib.sha256((ROOT / 'manifest.json').read_bytes()).hexdigest() != manifest_sha256:
        raise RuntimeError('Continuation manifest changed while queued')
    for name, sha in expected.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Continuation source changed: ' + name)


def stop_child():
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def interrupt(signum, frame):
    global interrupted
    interrupted = signum


def memory():
    cg = Path('/sys/fs/cgroup')
    limit = (cg / 'memory.max').read_text().strip()
    stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    working = int((cg / 'memory.current').read_text()) - int(stats.get('inactive_file', 0))
    return limit, working


signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGINT, interrupt)
try:
    previous_phase = None
    while True:
        if interrupted:
            raise InterruptedError(str(interrupted))
        verify()
        state_file = UPSTREAM / 'queue_status.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        running = [line for line in subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
                   if str(UPSTREAM.parent) in line and
                   ('queue_clone_benchmark_20261026.py' in line or 'run_clone_benchmark_20261026.py' in line)]
        phase = 'waiting_for_existing_clone_terminal'
        if state.get('phase') in ['terminal', 'queue_error', 'user_or_operator_interrupted'] and not running:
            if not (UPSTREAM / 'grains/original_pca_vector.pt').exists():
                raise RuntimeError('Upstream terminated without completed extraction/PCA')
            gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                 capture_output=True, text=True)
            limit, working = memory()
            phase = 'waiting_for_observed_gpu_and_memory_idle'
            if gpu.returncode == 0 and not gpu.stdout.strip() and limit != 'max' and int(limit) - working >= 48 * 1024**3:
                break
        if phase != previous_phase:
            status(phase)
            previous_phase = phase
        time.sleep(15)
    verify()
    if interrupted:
        raise InterruptedError(str(interrupted))
    with (ROOT / 'run.log').open('ab') as stream:
        child = subprocess.Popen(['bash', str(ROOT / 'run_generation.sh')], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status('running_original_tuple_generation', worker_pid=child.pid)
        while child.poll() is None:
            if interrupted:
                raise InterruptedError(str(interrupted))
            limit, working = memory()
            if limit != 'max' and working > int(limit) - 4 * 1024**3:
                stop_child()
                status('memory_guard_stopped_own_method', limit_bytes=int(limit), working_set_bytes=working)
                break
            time.sleep(1)
        code = child.wait()
    status('terminal', exit_code=code, whole_reproduction_complete=False)
except Exception as error:
    stop_child()
    status('queue_error', error=repr(error))
    raise
