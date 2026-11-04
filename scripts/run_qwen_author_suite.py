"""Queue fresh Qwen evaluation after verified Gemma completion, on GPUs 2/3."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
GEMMA = ROOT / 'outputs/author-clone-rerun-20261027-165446'
PLAN = ROOT / 'outputs/qwen25-vl7b-next-plan-20261028'
GPU2 = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3 = 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
PAIR = GPU2 + ',' + GPU3
METHODS = ['base', 'revis', 'vti', 'rudder_clone', 'base_fp32', 'grains_clone']
SOURCES = ['src/vlm_grounding/backend.py', 'src/vlm_grounding/baselines.py',
    'src/vlm_grounding/cli.py', 'src/vlm_grounding/evaluation.py',
    'scripts/author_clone_bridge.py', 'scripts/qwen_author_compat.py',
    'scripts/qwen_gpu23_adapter.py', 'scripts/preflight_qwen_authors.py',
    'scripts/fit_qwen_retained.py', 'scripts/extract_qwen_author_grains.py',
    'scripts/run_qwen_author_worker.py', 'scripts/run_qwen_author_suite.py',
    'scripts/audit_qwen_author_suite.py', 'scripts/audit_hallucination_suite.py',
    'data/hallucination-suite/sources/pope_evaluate.py', 'upstream/chair/chair.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ticks(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()
    return int(fields[19])


def selected_gpus_empty():
    lines = subprocess.check_output(['nvidia-smi', '-i', '2,3', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    assert {tuple(s.strip() for s in line.split(',')) for line in lines} == {('2', GPU2), ('3', GPU3)}
    processes = subprocess.check_output(['nvidia-smi', '-i', '2,3', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    assert not processes, 'GPUs 2/3 occupied; no fallback or interference: ' + processes


def check_inputs():
    config = json.loads((ROOT / 'configs/qwen2_5_vl_7b.json').read_text())
    model = config['model']
    assert model['checkpoint'] == 'Qwen/Qwen2.5-VL-7B-Instruct'
    assert model['revision'] == 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
    assert model['family'] == 'qwen2_5_vl' and model['dtype'] == 'bfloat16'
    assert model['image_size'] == 448 and model['attn_implementation'] == 'eager'
    assert config['generation']['seed'] == 42
    assert config['methods']['revis'] == {'alpha': 1.6, 'risk_quantile': 0.8}
    assert config['methods']['vti'] == {'alpha': 0.5, 'vision_alpha': 0.5, 'vision_pooling': 'position'}
    fit = read(ROOT / 'data/mmvp-full/fit.jsonl')
    calibration = read(ROOT / 'data/mmvp-full/calibration.jsonl')
    assert len(fit) == len(calibration) == 32
    assert read(GEMMA / 'data/fit50.jsonl') == (fit + calibration)[:50]
    dev = {r['group_id']: sha(r['image']) for r in fit + calibration}
    assert len(dev) == len(set(dev.values())) == 64
    datasets = {b: read(GEMMA / 'data' / (b + '.jsonl')) for b in ('pope', 'chair')}
    for bench, rows in datasets.items():
        assert len(rows) == len({r['id'] for r in rows}) == (3000 if bench == 'pope' else 500)
        assert all(r['max_new_tokens'] == (128 if bench == 'pope' else 512) for r in rows)
        assert not set(dev.values()) & {r['image_sha256'] for r in rows}
        assert all(sha(r['image']) == r['image_sha256'] for r in rows)
    tasks = json.loads((GEMMA / 'tasks.json').read_text())
    assert len(tasks) == len({t['id'] for t in tasks}) == 678
    for bench, rows in datasets.items():
        for method in METHODS:
            selected = [t for t in tasks if t['benchmark'] == bench and t['method'] == method]
            assert [i for t in selected for i in range(t['start'], t['stop'])] == list(range(len(rows)))
            assert all(t['dtype'] == ('float32' if method in ('base_fp32', 'grains_clone') else 'bfloat16') for t in selected)
    weight_proof = json.loads((PLAN / 'weight-cache-verification.json').read_text())
    assert weight_proof['status'] == 'verified_cached_weight_files'
    assert weight_proof['all_paths_within_LJW'] and weight_proof['all_hf_lfs_sha256_verified']
    assert len(weight_proof['files']) == 5 and all(r['hf_lfs_hash_match'] for r in weight_proof['files'])
    for record in weight_proof['files']:
        path = Path(record['resolved_path']).resolve()
        assert path.is_relative_to(ROOT.parent) and path.name == record['sha256']
        assert path.stat().st_size == record['bytes']
    return {'status': 'cpu_inputs_verified_gpu_not_started', 'methods': METHODS,
        'records_per_method': {'pope': 3000, 'chair': 500}, 'tasks': len(tasks),
        'development_test_image_hash_overlap': 0, 'weights_downloaded': False,
        'source_hashes': {name: sha(ROOT / name) for name in SOURCES},
        'weight_verification_sha256': sha(PLAN / 'weight-cache-verification.json')}


def freeze_protocol(output, preflight):
    from author_clone_bridge import verify_clone_sources
    from qwen_gpu23_adapter import require_gemma_complete
    require_gemma_complete()
    checked = check_inputs()
    for name in ('data', 'claims', 'predictions', 'artifacts', 'leases', 'source-snapshot', 'author-snapshot', 'proofs'):
        (output / name).mkdir()
    config = json.loads((ROOT / 'configs/qwen2_5_vl_7b.json').read_text())
    config['model']['local_files_only'] = True
    save(output / 'config.json', config)
    for name in ('pope.jsonl', 'chair.jsonl', 'fit50.jsonl'):
        shutil.copy2(GEMMA / 'data' / name, output / 'data' / name)
    for name in ('fit.jsonl', 'calibration.jsonl'):
        shutil.copy2(ROOT / 'data/mmvp-full' / name, output / 'data' / name)
    shutil.copy2(GEMMA / 'tasks.json', output / 'tasks.json')
    shutil.copy2(PLAN / 'weight-cache-verification.json', output / 'proofs/weight-cache-verification.json')
    sources = checked['source_hashes']; clones = verify_clone_sources()
    proofs = {}
    versions = None
    for phase in ('bf16', 'fp32'):
        proof = json.loads((preflight / (phase + '-preflight.json')).read_text())
        assert proof['status'] == 'passed' and proof['author_sources'] == clones
        assert all(sources[name] == digest for name, digest in proof['source_hashes'].items())
        for file, digest in proof['development_manifest_hashes'].items():
            assert sha(file) == digest and sha(output / 'data' / Path(file).name) == digest
        if versions is not None:
            assert versions == proof['dependency_versions']
        versions = proof['dependency_versions']
        target = output / 'proofs' / (phase + '-preflight.json')
        shutil.copy2(preflight / target.name, target); proofs[phase] = sha(target)
    for name, digest in sources.items():
        target = output / 'source-snapshot' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target); assert sha(target) == digest
    for name, info in clones.items():
        for filename, digest in info['files'].items():
            target = output / 'author-snapshot' / name / filename; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'upstream-author-20261026' / name / filename, target)
            assert sha(target) == digest
    parent = json.loads((GEMMA / 'protocol.json').read_text())
    protocol = {'scope': 'Fresh Qwen2.5-VL-7B evaluation after completed Gemma: POPE adversarial 3000 and CHAIR fixed500 per profile',
        'model': config['model'], 'methods': METHODS, 'gpu_indices': [2, 3], 'other_gpu_fallback': False,
        'records_per_method': {'pope_adversarial': 3000, 'chair_coco_val2014': 500},
        'generation': parent['generation'], 'method_parameters': parent['method_parameters'],
        'comparisons': parent['comparisons'], 'dataset_provenance': parent['dataset_provenance'],
        'chair_scoring': parent['chair_scoring'], 'deferred': parent['deferred'],
        'source_commit': parent['source_cli_main_commit'], 'source_hashes': sources, 'author_sources': clones,
        'manifest_hashes': {p.name: sha(p) for p in (output / 'data').glob('*.jsonl')},
        'tasks_sha256': sha(output / 'tasks.json'), 'config_sha256': sha(output / 'config.json'),
        'preflight_hashes': proofs, 'dependency_versions': versions,
        'cached_weight_verification_sha256': sha(output / 'proofs/weight-cache-verification.json'),
        'gemma_verified_report_sha256': sha(GEMMA / 'report.json'),
        'development_test_hash_overlap': 0, 'all_predictions_fresh': True,
        'method_labels': {'vti': 'VTI local adaptation, not original-author reproduction',
            'revis': 'REVIS retained local adaptation', 'rudder_clone': 'Pinned original author functions with native Qwen adapter',
            'grains_clone': 'Pinned original author functions with native Qwen adapter'},
        'gemma_vectors_reused': False, 'author_reproduction_complete': False,
        'author_core_functions_called': True,
        'architecture_compatibility': ['Native Qwen image-first chat input and native multimodal generation',
            'Original token-only IG/ablation, 20 Gauss-Legendre nodes, PCA layer27, sequence-axis norm and multiplier10 retained',
            'Hidden-state output device coalescing for original cross-layer stack only; no full visual-path IG claim'],
        'protocol_overrides': ['Same user-selected common image448, POPE cap128 and CHAIR cap512',
            'Fresh Qwen-specific vectors from the fixed disjoint 64-image development cohort'],
        'created_at': time.time()}
    protocol['sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    save(output / 'protocol.json', protocol)
    return protocol


def run(output):
    assert output.resolve().parent == ROOT / 'outputs'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Controller must be CPU-only'
    # One queue owns all Qwen stage transitions, even across differently named runs.
    suite_lock = (ROOT / 'outputs/.qwen-author-suite-controller.lock').open('a')
    fcntl.flock(suite_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir()
    lock = (output / '.controller.lock').open('x')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'controller_pid': os.getpid(), 'controller_start_ticks': ticks(os.getpid()),
        'started_at': time.time(), 'gpu_indices': [2, 3], 'other_gpu_fallback': False,
        'qwen_gpu_models_launched': False, 'jobs': []}
    jobs, preflight_jobs = [], []
    def update(**values):
        state.update(values, updated_at=time.time()); save(output / 'status.json', state)
    def launch(script, extra, visible, kind, child_output=output, **metadata):
        label = metadata.get('worker', metadata.get('label', kind))
        log_path = child_output / (label + '.log')
        with log_path.open('xb') as stream:
            process = subprocess.Popen([str(ROOT / '.venv/bin/python'), script, '--output', str(child_output), *extra],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': process.pid, 'start_ticks': ticks(process.pid), 'script': script,
            'kind': kind, 'output': str(child_output), 'visible_devices': visible,
            'log': str(log_path), **metadata}
        target = preflight_jobs if kind == 'preflight' else jobs
        if kind != 'audit':
            target.append(record)
            save(output / ('preflight-jobs.json' if kind == 'preflight' else 'jobs.json'), target)
        else:
            save(output / 'audit-job.json', record)
        if visible:
            update(qwen_gpu_models_launched=True)
        return process, record
    def wait(group, phase):
        while True:
            progress = []
            finished = True
            for process, job in group:
                result = process.poll()
                finished &= result is not None
                file = Path(job['output']) / job['state_file']
                status = json.loads(file.read_text()) if file.exists() else {}
                assert result in (None, 0), {'job': job, 'returncode': result, 'state': status}
                assert status.get('stage') != 'failed' and status.get('status') != 'failed', status
                if result == 0:
                    assert status.get('stage', status.get('status')) in ('complete', 'passed', 'verified_complete'), status
                progress.append({'pid': job['pid'], 'kind': job['kind'], 'returncode': result,
                    'stage': status.get('stage', status.get('status')), 'worker': job.get('worker'),
                    'completed_records': status.get('completed_records', status.get('completed'))})
            update(stage=phase, jobs=progress)
            if finished:
                return
            time.sleep(15)
    try:
        checked = check_inputs(); save(output / 'cpu-preparation.json', checked)
        save(ROOT / 'outputs/qwen-author-suite-current.json', {'output': str(output), 'controller_pid': os.getpid(),
            'controller_start_ticks': state['controller_start_ticks'], 'gpu_indices': [2, 3],
            'scope': 'Queued after verified Gemma; all fresh Qwen results and model-specific artifacts'})
        while True:
            previous = json.loads((GEMMA / 'status.json').read_text())
            if previous['stage'] == 'complete':
                from qwen_gpu23_adapter import require_gemma_complete
                require_gemma_complete()
                break
            update(stage='waiting_for_gemma', gemma_stage=previous['stage'], gemma_phase=previous.get('phase'),
                gemma_error=previous.get('error'), qwen_gpu_models_launched=False)
            time.sleep(30)
        selected_gpus_empty()
        preflight = output.with_name(output.name + '-preflight'); preflight.mkdir()
        for phase, visible in [('bf16', GPU2), ('fp32', PAIR)]:
            selected_gpus_empty()
            job = launch('scripts/preflight_qwen_authors.py', ['--phase', phase], visible,
                'preflight', preflight, label=phase, state_file=phase + '-preflight.json', dtype='bfloat16' if phase == 'bf16' else 'float32')
            wait([job], 'preflight_' + phase)
        selected_gpus_empty()
        protocol = freeze_protocol(output, preflight)
        update(stage='fitting_retained', protocol_sha256=protocol['sha256'])
        group = []
        for method, visible in [('revis', GPU2), ('vti', GPU3)]:
            group.append(launch('scripts/fit_qwen_retained.py', ['--method', method], visible, 'fit',
                label='fit-' + method, state_file='fit-' + method + '-status.json',
                device_proof='fit-' + method + '-device-map.json', dtype='bfloat16'))
        wait(group, 'fitting_retained')
        selected_gpus_empty()
        job = launch('scripts/extract_qwen_author_grains.py', [], PAIR, 'extraction',
            label='grains-extraction', state_file='extraction-status.json',
            device_proof='grains-extraction-device-map.json', dtype='float32')
        wait([job], 'fitting_grains_fp32')
        selected_gpus_empty()
        seals = {}
        for method, path in [('revis', output / 'artifacts/revis.pt'), ('vti', output / 'artifacts/vti.pt'),
                             ('grains_clone', output / 'grains-author.pt')]:
            sidecar = json.loads((output / (('grains' if method == 'grains_clone' else method) + '-artifact.json')).read_text())
            assert sidecar['status'] == 'verified_complete' and sidecar['source_protocol_sha256'] == protocol['sha256']
            assert sidecar['artifact_sha256'] == sha(path)
            seals[method] = {'sha256': sha(path)}
        save(output / 'artifacts-ready.json', {'status': 'verified_complete', 'source_protocol_sha256': protocol['sha256'],
            'methods': seals, 'created_at': time.time(), 'gemma_vectors_reused': False})
        tag = str(int(time.time())); group = []
        for label, visible in [('gpu2-bf16', GPU2), ('gpu3-bf16', GPU3)]:
            worker = label + '-' + tag
            group.append(launch('scripts/run_qwen_author_worker.py', ['--worker', worker, '--dtype', 'bfloat16',
                '--lease-epoch', 'bf16-' + tag], visible, 'generation', worker=worker, dtype='bfloat16',
                state_file='worker-' + worker + '.json', device_proof=worker + '-device-map.json'))
        wait(group, 'generating_bf16')
        selected_gpus_empty()
        worker = 'gpu23-fp32-' + tag
        job = launch('scripts/run_qwen_author_worker.py', ['--worker', worker, '--dtype', 'float32',
            '--lease-epoch', 'fp32-' + tag], PAIR, 'generation', worker=worker, dtype='float32',
            state_file='worker-' + worker + '.json', device_proof=worker + '-device-map.json')
        wait([job], 'generating_fp32')
        selected_gpus_empty()
        job = launch('scripts/audit_qwen_author_suite.py', [], '', 'audit',
            label='official-audit', state_file='report.json')
        wait([job], 'auditing')
        report = json.loads((output / 'report.json').read_text())
        assert report['status'] == 'verified_complete'
        assert report['audit']['all_21000_responses_verified'] and report['audit']['official_scorers_used']
        update(stage='complete', completed_at=time.time(), report_sha256=sha(output / 'report.json'))
    except BaseException:
        update(stage='failed', error=traceback.format_exc())
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()
        fcntl.flock(suite_lock.fileno(), fcntl.LOCK_UN); suite_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-preparation', action='store_true')
    args = parser.parse_args()
    if args.check_preparation:
        print(json.dumps(check_inputs(), allow_nan=False))
    else:
        assert args.output is not None
        run(args.output)
