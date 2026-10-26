"""Resource queue only: invoke the original repositories' command-line programs."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/workspace/original-baselines-20261025/grains_gpu_retry_20261026')
OUT = ROOT / 'runs_checkpoint'
OUT.mkdir(exist_ok=True)
lock = (OUT / 'queue.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
PROTECTED = ('launch_vti_strength_calibration.sh', 'run_vti_strength_calibration.py',
             'launch_vti_selected_generation.sh', 'run_vti_selected_generation.py')
expected = json.loads((ROOT / 'execution_manifest_compat_20261026.json').read_text())
results = {}
child = None
stop_signal = 0


def status(phase, **extra):
    value = {'at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
             'queue_pid': os.getpid(), 'method_exit_codes': results, **extra}
    temporary = OUT / 'queue_status.tmp'
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(OUT / 'queue_status.json')
    with (OUT / 'queue_events.jsonl').open('a') as journal:
        journal.write(json.dumps(value) + '\n')


def verify():
    for relative, digest in expected.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source/config changed: ' + relative)


def memory_usage():
    cg = Path('/sys/fs/cgroup')
    limit = (cg / 'memory.max').read_text().strip()
    current = int((cg / 'memory.current').read_text())
    statistics = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
    return limit, max(0, current - int(statistics.get('inactive_file', '0')))


def available():
    ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True, check=True)
    if any(name in ps.stdout for name in PROTECTED):
        return False, 'waiting_for_existing_vti'
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True)
    if gpu.returncode != 0 or gpu.stdout.strip():
        return False, 'waiting_for_observed_gpu_idle'
    limit, current = memory_usage()
    if limit == 'max' or int(limit) - current < 48 * 1024**3:
        return False, 'waiting_for_observed_memory_headroom'
    return True, 'ready'


def interrupt(signum, frame):
    global stop_signal
    stop_signal = signum


def stop_child():
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def check_interruption():
    if stop_signal:
        stop_child()
        status('user_or_operator_interrupted', signal=stop_signal)
        raise SystemExit(128 + stop_signal)


signal.signal(signal.SIGTERM, interrupt)
signal.signal(signal.SIGINT, interrupt)
try:
    for method in ('grains',):
        while True:
            check_interruption()
            ready, phase = available()
            if ready:
                break
            status(phase, next_method=method)
            time.sleep(30)
        verify()
        check_interruption()
        with (OUT / (method + '_launcher.log')).open('ab') as log:
            child = subprocess.Popen(['bash', str(ROOT / 'operations' / ('run_clone_' + method + '_20261026.sh'))],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status('running_original_' + method, launcher_pid=child.pid)
            while child.poll() is None:
                check_interruption()
                limit, working_set = memory_usage()
                if limit != 'max' and working_set > int(limit) - 4 * 1024**3:
                    stop_child()
                    status('memory_guard_stopped_own_method', method=method, launcher_pid=child.pid,
                           limit_bytes=int(limit), working_set_bytes=working_set)
                    break
                time.sleep(1)
            results[method] = child.wait()
            child = None
        status('method_finished', method=method, exit_code=results[method])
    status('terminal', all_commands_succeeded=all(code == 0 for code in results.values()),
           whole_reproduction_complete=False)
except Exception as exc:
    stop_child()
    status('queue_error', error=repr(exc))
    raise
