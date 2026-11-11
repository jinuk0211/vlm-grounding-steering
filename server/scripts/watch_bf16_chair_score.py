"""One-time CPU milestone scoring as soon as the four BF16 cohorts reach500."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
RUN = ROOT / 'outputs/author-clone-rerun-20261027-165446'
METHODS = ('base', 'revis', 'vti', 'rudder_clone')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    lock = (RUN / '.bf16-score-watch.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'pid': os.getpid(), 'started_at': time.time(), 'gpu_compute_started': False,
        'full_evaluation_complete': False, 'analysis_sha256': sha(ROOT / 'scripts/score_author_partial.py'),
        'watcher_sha256': sha(__file__)}
    status_path = RUN / 'bf16-score-watch-status.json'
    def update(**values):
        state.update(values, updated_at=time.time()); save(status_path, state)
    try:
        tasks = [t for t in json.loads((RUN / 'tasks.json').read_text()) if t['benchmark'] == 'chair' and t['method'] in METHODS]
        refs = [json.loads(s) for s in (RUN / 'data/chair.jsonl').read_text().splitlines() if s.strip()]
        expected = {r['id'] for r in refs}; assert len(expected) == 500
        while True:
            collected = {m: set() for m in METHODS}
            for task in tasks:
                path = RUN / 'predictions' / (task['id'] + '.jsonl')
                if not path.exists():
                    continue
                raw = path.read_bytes(); committed = raw[:raw.rfind(b'\n') + 1]
                for line in committed.splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert row['method'] == task['method'] and row['id'] in expected
                    assert row['id'] not in collected[task['method']]
                    collected[task['method']].add(row['id'])
            counts = {m: len(v) for m, v in collected.items()}
            if all(ids == expected for ids in collected.values()):
                break
            generation = json.loads((RUN / 'status.json').read_text())
            update(stage='waiting_for_bf16_500', counts=counts, generation_stage=generation['stage'],
                generation_phase=generation.get('phase'), generation_error=generation.get('error'))
            time.sleep(45)
        # If the full audit has already finished, use that report directly.
        final = RUN / 'report.json'
        if final.exists() and json.loads(final.read_text()).get('status') == 'verified_complete':
            update(stage='complete', report=str(final), report_sha256=sha(final), uses_completed_full_report=True)
            return
        assert sha(ROOT / 'scripts/score_author_partial.py') == state['analysis_sha256']
        output = RUN / ('bf16-full500-' + time.strftime('%Y%m%d-%H%M%S'))
        log = RUN / (output.name + '.log')
        with log.open('xb') as stream:
            process = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/score_author_partial.py',
                '--output', str(output)], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdout=stream, stderr=subprocess.STDOUT)
            update(stage='official_scoring', output=str(output), scoring_pid=process.pid, counts=counts, log=str(log))
            assert process.wait() == 0, 'CPU scorer failed; inspect ' + str(log)
        report = json.loads((output / 'report.json').read_text())
        assert report['status'] == 'verified_interim_snapshot'
        assert report['cohorts']['bf16_common']['n'] == 500 and not report['full_evaluation_complete']
        archive = RUN / (output.name + '.zip')
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for file in output.rglob('*'):
                if file.is_file():
                    z.write(file, str(file.relative_to(output)))
        update(stage='complete', report=str(output / 'report.json'), report_sha256=sha(output / 'report.json'),
            archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
            completed_at=time.time(), uses_completed_full_report=False)
    except BaseException:
        update(stage='failed', error=traceback.format_exc()); raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    main()
