"""Finish this single experiment when its actual workers and scorer are ready."""
import collections
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    current = json.loads((ROOT / 'outputs/hallucination-suite-current.json').read_text())
    output = Path(current['output'])
    path = output / 'completion-status.json'
    while True:
        states = [json.loads((output / f"worker-{job['worker']}.json").read_text()) for job in current['jobs']]
        live = []
        for job in current['jobs']:
            process = Path('/proc') / str(job['pid'])
            live.append(process.exists() and process.joinpath('stat').read_text().split(') ', 1)[1][0] != 'Z')
        scorer = json.loads((ROOT / 'data/hallucination-suite/chair-cache-status.json').read_text())
        failed = [state for state, active in zip(states, live) if state['stage'] == 'failed' or (not active and state['stage'] != 'complete')]
        if failed or scorer['stage'] == 'failed':
            save(path, {'stage': 'failed', 'workers': failed, 'scorer': scorer, 'updated_at': time.time()})
            raise RuntimeError('A worker/scorer failed; preserve outputs for repair')
        done = len(list((output / 'claims').glob('*/complete.json')))
        save(path, {'stage': 'generating', 'completed_tasks': done, 'total_tasks': 805,
                    'workers_live': sum(live), 'completed_unique_generations': sum(s['completed_records'] for s in states),
                    'scorer_stage': scorer['stage'], 'updated_at': time.time()})
        if all(state['stage'] == 'complete' for state in states) and not any(live) and scorer['stage'] == 'complete':
            break
        time.sleep(30)
    save(path, {'stage': 'auditing', 'started_at': time.time()})
    result = subprocess.run([os.sys.executable, 'scripts/audit_hallucination_suite.py', '--output', str(output)], cwd=ROOT)
    if result.returncode:
        save(path, {'stage': 'audit_failed', 'exit_code': result.returncode, 'updated_at': time.time()})
        raise SystemExit(result.returncode)
    report = json.loads((output / 'report.json').read_text())
    save(path, {'stage': 'complete', 'report_status': report['status'], 'completed_at': time.time()})


if __name__ == '__main__':
    main()
