"""Run a Gemma command on GPU 2, restoring the user's LJW Explorer afterwards."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path('/home/project-b/LJW')
REPO = ROOT / 'vlm-grounding-steering'
EXPLORER = ROOT / 'neuron-explorer'
GPU = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'


def gpu_processes():
    rows = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits',
    ], text=True).splitlines()
    return [int(row.split(',')[1]) for row in rows if row.split(',')[0].strip() == GPU]


def running(pid):
    path = Path('/proc') / str(pid)
    try:
        return path.exists() and path.joinpath('stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def gpu_idle():
    used, utilization = subprocess.check_output([
        'nvidia-smi', '-i', '2', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True).strip().split(',')
    return int(used) <= 10 and int(utilization) == 0 and not gpu_processes()


def main():
    arguments = sys.argv[1:] or ['smoke']
    command, extra = arguments[0], arguments[1:]
    if command not in {'smoke', 'run', 'discover', 'fit', 'extract', 'suite'}:
        raise SystemExit('Choose smoke, run, discover, fit, extract or suite')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = {'gpu_index': 2, 'gpu_uuid': GPU, 'status': 'starting', 'command': command}
    report = REPO / 'outputs/setup' / f'gpu2-validation-{stamp}.json'
    pids = gpu_processes()
    saved = None
    if pids:
        if len(pids) != 1:
            raise RuntimeError('GPU 2 has more than the expected LJW Explorer process')
        pid = pids[0]
        proc = Path('/proc') / str(pid)
        cwd = (proc / 'cwd').resolve()
        args = [part.decode() for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
        if cwd != EXPLORER or args != ['.venv/bin/python', 'explorer/server.py']:
            raise RuntimeError('GPU 2 process is not the expected LJW Explorer; leaving it alone')
        if proc.stat().st_uid != os.getuid():
            raise RuntimeError('Explorer process belongs to a different user')
        with urlopen('http://127.0.0.1:8792/api/status', timeout=5) as response:
            if json.load(response).get('status') != 'ready':
                raise RuntimeError('Explorer is not ready/idle; do not interrupt its startup')
        environment = dict(part.decode().split('=', 1) for part in
                           (proc / 'environ').read_bytes().split(b'\0') if part)
        if environment.get('CUDA_VISIBLE_DEVICES') != GPU:
            raise RuntimeError('Explorer GPU environment differs from the expected device')
        saved = (pid, args, environment)
    try:
        if saved:
            print('VALIDATION_STAGE=pausing_ljw_explorer', flush=True)
            os.kill(saved[0], signal.SIGTERM)
            for _ in range(40):
                if not running(saved[0]):
                    break
                time.sleep(0.5)
            if running(saved[0]):
                raise RuntimeError('Explorer did not exit gracefully; no forced stop was attempted')
            result['paused_explorer_pid'] = saved[0]
        # GPU utilization can still report the last kernel briefly after exit.
        for _ in range(40):
            if gpu_idle():
                break
            time.sleep(0.5)
        if not gpu_idle():
            raise RuntimeError('GPU 2 is still occupied')
        print('VALIDATION_STAGE=gemma_command_gpu2 command=' + command, flush=True)
        completed = subprocess.run(
            ['bash', 'scripts/gemma.sh', command, '2', *extra], cwd=REPO,
            timeout=420 if command == 'smoke' else None,
        )
        result['command_exit_code'] = completed.returncode
        result['status'] = 'passed' if completed.returncode == 0 else 'command_failed'
    except Exception as error:
        result['status'] = 'failed'
        result['error'] = str(error)
    finally:
        if saved and not running(saved[0]):
            print('VALIDATION_STAGE=restoring_ljw_explorer', flush=True)
            log = EXPLORER / 'logs' / f'explorer-restored-{stamp}.log'
            with log.open('xb') as stream:
                server = subprocess.Popen(
                    saved[1], cwd=EXPLORER, env=saved[2], stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
            result['restored_explorer_pid'] = server.pid
            result['restored_explorer_log'] = str(log)
            result['explorer_restored'] = False
            for _ in range(90):
                if server.poll() is not None:
                    break
                try:
                    with urlopen('http://127.0.0.1:8792/api/status', timeout=2) as response:
                        if json.load(response).get('status') == 'ready':
                            result['explorer_restored'] = True
                            break
                except OSError:
                    pass
                time.sleep(1)
        report.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({**result, 'report': str(report)}, indent=2), flush=True)
    return int(result['status'] != 'passed' or
               (saved is not None and not result.get('explorer_restored', False)))


if __name__ == '__main__':
    raise SystemExit(main())
