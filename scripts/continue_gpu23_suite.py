"""Continue preserved CHAIR outputs on the explicitly requested GPUs 2 and 3."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from continue_author_clone_suite import check as historical_check, live, save

ROOT = Path(__file__).resolve().parents[1]
GPU2 = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3 = 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
PAIR = GPU2 + ',' + GPU3
SOURCES = ['scripts/continue_gpu23_suite.py', 'scripts/continue_gpu23_entry.py',
           'scripts/continue_gpu23_worker.py', 'scripts/gpu23_device_adapter.py',
           'scripts/continue_author_clone_suite.py', 'scripts/preflight_gpu23.py',
           'scripts/audit_author_clone_suite.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_gpus_empty():
    rows = subprocess.check_output(['nvidia-smi', '-i', '2,3', '--query-gpu=index,uuid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    assert {tuple(v.strip() for v in r.split(',')) for r in rows} == {('2', GPU2), ('3', GPU3)}
    processes = subprocess.check_output(['nvidia-smi', '-i', '2,3', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True).strip()
    assert not processes, 'Selected GPUs are occupied; no other GPU fallback: ' + processes


def check(output):
    checked = historical_check(output)
    assert not (output / 'continuation-amendment.json').exists()
    assert not (output / 'hardware-amendment.json').exists()
    original = (ROOT / 'scripts/run_author_clone_worker.py').read_text()
    resumed = (ROOT / 'scripts/continue_gpu23_worker.py').read_text()
    first = '                    key=(task'
    assert original[original.index(first):original.index('                    sink.write')] == resumed[resumed.index(first):resumed.index("                    row['hardware_execution']")]
    preflight = json.loads((output / 'gpu23-preflight.json').read_text())
    preflight_job = json.loads((output / 'gpu23-preflight-job.json').read_text())
    assert preflight['status'] == 'passed' and not live(preflight_job['pid'])
    assert preflight['device_proof']['visible_devices'] == PAIR
    assert preflight['device_proof']['parameter_dtypes'] == ['torch.float32']
    assert preflight['device_proof']['parameter_devices'] == ['cuda:0', 'cuda:1']
    assert not preflight['device_proof']['cpu_or_disk_offload']
    tasks=json.loads((output/'tasks.json').read_text())
    datasets={b:[json.loads(s) for s in (output/'data'/(b+'.jsonl')).read_text().splitlines() if s.strip()] for b in ('pope','chair')}
    for task in tasks:
        path=output/'predictions'/(task['id']+'.jsonl')
        prior=[json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        for record in datasets[task['benchmark']][task['start']+len(prior):task['stop']]:
            assert not (output/'cache'/task['method']/(record['id'].replace(':','_')+'.json')).exists(), 'Unexpected outstanding legacy cache'
    checked.update(source_hashes={f: sha(ROOT / f) for f in SOURCES},
                   preflight_sha256=sha(output / 'gpu23-preflight.json'),
                   hardware_change_requested='2026-10-28 user: use GPU 2 and GPU 3',
                   bf16_visible_devices=[GPU2, GPU3], fp32_visible_devices=PAIR,
                   old_gpu0_responses_preserved=True, generation_body_identical=True)
    return checked


def execute(output, checked):
    selected_gpus_empty()
    previous_jobs = json.loads((output / 'jobs.json').read_text())
    save(output / 'pre-continuation-jobs.json', previous_jobs)
    hardware = {**checked, 'historical_gpu_uuid': 'GPU-4a6e1799-a39c-86ed-abdf-afad03791c12',
        'historical_and_current_visible_devices': ['GPU-4a6e1799-a39c-86ed-abdf-afad03791c12', GPU2, GPU3, PAIR],
        'current_gpu_indices': [2, 3], 'automatic_gpu_fallback': False,
        'dtype_and_generation_settings_unchanged': True,
        'fp32_placement': 'balanced across GPU2/GPU3; full FP32; no quantization or CPU/disk offload',
        'phases': ['two BF16 replicas, one per GPU', 'one full-FP32 model sharded over both GPUs'],
        'created_at': time.time()}
    save(output / 'hardware-amendment.json', hardware)
    save(output / 'continuation-amendment.json', {**checked,
        'pre_continuation_jobs_sha256': sha(output / 'pre-continuation-jobs.json'),
        'preserves_existing_response_bytes': True,
        'hardware_amendment_sha256': sha(output / 'hardware-amendment.json'),
        'changes': 'User-authorized GPU2/GPU3 placement and prefix-preserving task continuation; generation body unchanged'})
    (output / 'continuation-claims').mkdir()
    jobs = [j for j in previous_jobs if j['extract']]
    tag = str(int(time.time()))
    started = time.time()

    def launch(label, dtype, visible):
        worker = label + '-' + tag
        with (output / (worker + '.log')).open('xb') as log:
            process = subprocess.Popen([str(ROOT / '.venv/bin/python'), 'scripts/continue_gpu23_entry.py',
                '--output', str(output), '--worker', worker, '--dtype', dtype], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        job = {'worker': worker, 'pid': process.pid, 'dtype': dtype, 'extract': False,
               'visible_devices': visible, 'log': str(output / (worker + '.log'))}
        jobs.append(job); save(output / 'jobs.json', jobs)
        return job

    def states():
        records = {}
        for job in jobs:
            file = output / ('extraction-status.json' if job['extract'] else 'worker-' + job['worker'] + '.json')
            state = json.loads(file.read_text()) if file.exists() else {'stage': 'starting'}
            active = live(job['pid']); records[job['worker']] = {**state, 'actual_live': active}
            if state['stage'] == 'failed' or (not active and state['stage'] != 'complete'):
                raise RuntimeError(json.dumps({'job': job, 'state': state}))
        return records

    pointer = json.loads((ROOT / 'outputs/author-clone-rerun-current.json').read_text())
    pointer.update(controller_pid=os.getpid(), log=str(output / 'gpu23-controller.log'),
                   scope='Historical GPU0 responses preserved; remaining CHAIR on user-selected GPU2/GPU3 only')
    save(ROOT / 'outputs/author-clone-rerun-current.json', pointer)
    try:
        bf16 = [launch('gpu2-bf16', 'bfloat16', GPU2), launch('gpu3-bf16', 'bfloat16', GPU3)]
        phase = 'bf16_replicas'; fp32 = None
        while True:
            current = states()
            if phase == 'bf16_replicas' and all(not live(j['pid']) for j in bf16):
                selected_gpus_empty()
                fp32 = launch('gpu23-fp32', 'float32', PAIR)
                phase = 'fp32_sharded'
            save(output / 'status.json', {'stage': 'generating', 'phase': phase,
                'controller_pid': os.getpid(), 'started_at': started, 'updated_at': time.time(),
                'workers': current, 'gpu_indices': [2, 3], 'hardware_amendment': True,
                'preserved_records': sum(checked['counts'].values()), 'total_records': 21000})
            if fp32 is not None and not live(fp32['pid']):
                break
            time.sleep(15)
        save(output / 'status.json', {'stage': 'auditing', 'controller_pid': os.getpid(),
                                      'gpu_indices': [2, 3], 'updated_at': time.time()})
        with (output / 'audit.log').open('ab') as log:
            subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/audit_author_clone_suite.py',
                '--output', str(output)], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((output / 'report.json').read_text())['status'] == 'verified_complete'
        save(output / 'status.json', {'stage': 'complete', 'controller_pid': os.getpid(),
                                      'gpu_indices': [2, 3], 'completed_at': time.time()})
    except BaseException:
        save(output / 'status.json', {'stage': 'failed', 'controller_pid': os.getpid(),
                                      'gpu_indices': [2, 3], 'error': traceback.format_exc(), 'updated_at': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert not (args.execute and args.check)
    result = check(args.output)
    save(args.output / 'gpu23-resume-check.json', result)
    print(json.dumps(result), flush=True)
    if args.execute:
        execute(args.output, result)
