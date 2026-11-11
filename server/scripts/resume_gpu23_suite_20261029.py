"""Prefix-preserving second continuation; fixed protocol, only GPU2 and GPU3."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from continue_gpu23_suite import selected_gpus_empty, GPU2, GPU3, PAIR

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['scripts/resume_gpu23_suite_20261029.py', 'scripts/resume_gpu23_entry_20261029.py',
           'scripts/audit_gpu23_resume_20261029.py']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def ticks(pid):
    return int((Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()[19])

def same_instance(job):
    p = Path('/proc') / str(job['pid'])
    try:
        fields = (p / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return False
        expected = job.get('start_ticks', job.get('ticks'))
        if expected is not None:
            return int(fields[19]) == expected
        cmd = (p / 'cmdline').read_bytes().decode().split('\0')
        return p.stat().st_uid == os.getuid() and os.readlink(p / 'cwd') == str(ROOT) and any('scripts/' in s for s in cmd) and str(job.get('output', 'UNSET')) in cmd
    except (OSError, ProcessLookupError):
        return False

def validate(output):
    assert output.resolve().parent == ROOT / 'outputs'
    assert json.loads((output / 'status.json').read_text())['stage'] == 'paused'
    pause = json.loads((output / 'pause-20261028.json').read_text())
    assert pause['all_our_processes_exited']
    for job in pause['targets']:
        assert not same_instance(job), job
    protocol = json.loads((output / 'protocol.json').read_text())
    body = {k:v for k,v in protocol.items() if k != 'sha256'}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
    for name in ['protocol.json', 'execution-amendment.json', 'hardware-amendment.json', 'continuation-amendment.json']:
        sealed = json.loads((output / name).read_text())
        for file, digest in sealed.get('source_hashes', {}).items():
            assert sha(ROOT / file) == digest, file
    for file, digest in protocol['manifest_hashes'].items():
        assert sha(output / 'data' / file) == digest
    for method, digest in protocol['retained_artifact_hashes'].items():
        assert sha(output / 'artifacts' / (method + '.pt')) == digest
    grains = json.loads((output / 'grains-artifact.json').read_text())
    assert grains['status'] == 'verified_complete' and sha(output / 'grains-author.pt') == grains['artifact_sha256']
    from author_clone_bridge import verify_clone_sources
    assert verify_clone_sources() == protocol['author_sources']
    original = json.loads((output / 'PAUSED_CHECKPOINT.json').read_text())
    for record in original['prediction_files']:
        assert hashlib.sha256((output / record['path']).read_bytes()[:record['bytes']]).hexdigest() == record['sha256']
    datasets = {b:[json.loads(s) for s in (output / 'data' / (b + '.jsonl')).read_text().splitlines() if s.strip()] for b in ['pope', 'chair']}
    counts = collections.Counter()
    files = []
    owner_history = {}
    for task in json.loads((output / 'tasks.json').read_text()):
        file = output / 'predictions' / (task['id'] + '.jsonl')
        raw = file.read_bytes() if file.exists() else b''
        assert not raw or raw.endswith(b'\n'), file
        rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
        refs = datasets[task['benchmark']][task['start']:task['stop']]
        assert len(rows) <= len(refs)
        for row, ref in zip(rows, refs):
            assert all(row.get(k) == v for k,v in ref.items())
            assert row['method'] == task['method'] and row['dtype'] == task['dtype']
            assert row['protocol_sha256'] == protocol['sha256']
        marker = output / 'claims' / task['id'] / 'complete.json'
        if marker.exists():
            done = json.loads(marker.read_text())
            assert len(rows) == len(refs) == done['records'] and done['sha256'] == sha(file)
        owner = output / 'claims' / task['id'] / 'owner.json'
        if owner.exists():
            owner_history[task['id']] = json.loads(owner.read_text())
        counts[task['benchmark'] + ':' + task['method']] += len(rows)
        if file.exists():
            files.append({'path':str(file.relative_to(output)), 'bytes':len(raw), 'records':len(rows), 'sha256':hashlib.sha256(raw).hexdigest()})
    assert dict(counts) == pause['counts'], dict(counts)
    selected_gpus_empty()
    return {'status':'validated_paused_prefixes', 'counts':dict(counts), 'prediction_files':files,
            'owner_history':owner_history, 'protocol_sha256':protocol['sha256'],
            'frozen_file_hashes':{n:sha(output/n) for n in ['protocol.json','execution-amendment.json','hardware-amendment.json','continuation-amendment.json','pause-20261028.json']},
            'new_source_hashes':{n:sha(ROOT/n) for n in SOURCES},
            'generation_body_unchanged':True, 'previous_responses_unchanged':True,
            'lease_change_only':'Fresh per-resume claims, preserved original claims and completion markers',
            'audit_change':'Accept explicitly stopped historical worker states; verify actual instance, all raw coverage and prefixes',
            'gpu_indices':[2,3], 'created_at':time.time()}

def run(output, execute):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    lock = (output / '.resume-controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    checked = validate(output)
    if not execute:
        print(json.dumps({k:v for k,v in checked.items() if k not in ['prediction_files','owner_history']}), flush=True)
        return
    epoch = 'resume-20261029-' + str(int(time.time()))
    folder = output / 'resumes' / epoch
    folder.mkdir(parents=True)
    (folder / 'claims').mkdir()
    save(folder / 'amendment.json', checked)
    previous_jobs = json.loads((output / 'jobs.json').read_text())
    save(folder / 'previous-jobs.json', previous_jobs)
    save(folder / 'previous-status.json', json.loads((output / 'status.json').read_text()))
    state = {'stage':'starting', 'resume_epoch':epoch, 'controller_pid':os.getpid(),
             'controller_start_ticks':ticks(os.getpid()), 'gpu_indices':[2,3], 'counts_at_resume':checked['counts'], 'started_at':time.time()}
    def update(**kw):
        state.update(kw, updated_at=time.time()); save(output / 'status.json', state)
    jobs = list(previous_jobs)
    active = []
    def launch(label, dtype, visible):
        name = label + '-' + epoch
        log = output / (name + '.log')
        with log.open('xb') as stream:
            child = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/resume_gpu23_entry_20261029.py', '--output', str(output), '--worker', name, '--dtype', dtype], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible, LJW_RESUME_EPOCH=epoch), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'worker':name, 'pid':child.pid, 'start_ticks':ticks(child.pid), 'dtype':dtype, 'extract':False, 'visible_devices':visible, 'log':str(log), 'resume_epoch':epoch}
        jobs.append(record); save(output / 'jobs.json', jobs)
        active.append((child,record)); return child,record
    def wait(group, phase):
        while True:
            snapshots = []
            finished = True
            for child, job in group:
                code = child.poll()
                finished &= code is not None
                path = output / ('worker-' + job['worker'] + '.json')
                ws = json.loads(path.read_text()) if path.exists() else {}
                assert code in (None,0), {'job':job,'code':code,'state':ws}
                assert ws.get('stage') != 'failed', ws
                if code == 0:
                    assert ws['stage'] == 'complete', ws
                snapshots.append({'worker':job['worker'], 'pid':job['pid'], 'returncode':code, 'state':ws})
            update(stage='generating', phase=phase, workers=snapshots)
            if finished:return
            time.sleep(15)
    try:
        update()
        pointer = json.loads((ROOT / 'outputs/author-clone-rerun-current.json').read_text())
        pointer.update(controller_pid=os.getpid(), controller_start_ticks=state['controller_start_ticks'], resume_epoch=epoch)
        save(ROOT / 'outputs/author-clone-rerun-current.json', pointer)
        bf16 = [launch('gpu2-bf16','bfloat16',GPU2), launch('gpu3-bf16','bfloat16',GPU3)]
        wait(bf16, 'bf16_replicas')
        selected_gpus_empty()
        wait([launch('gpu23-fp32','float32',PAIR)], 'fp32_sharded')
        selected_gpus_empty()
        update(stage='auditing')
        with (folder / 'audit.log').open('xb') as stream:
            subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/audit_gpu23_resume_20261029.py', '--output', str(output), '--epoch', epoch], cwd=ROOT, env=dict(os.environ,CUDA_VISIBLE_DEVICES=''), stdout=stream,stderr=subprocess.STDOUT,check=True)
        report = json.loads((output / 'report.json').read_text())
        assert report['status'] == 'verified_complete'
        update(stage='complete', report_sha256=sha(output / 'report.json'), completed_at=time.time())
    except BaseException:
        # Stop only children owned by this controller if any phase fails.
        for child, job in active:
            if child.poll() is None:child.terminate()
        for child, job in active:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        update(stage='failed',error=traceback.format_exc()); raise

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();run(args.output,args.execute)
