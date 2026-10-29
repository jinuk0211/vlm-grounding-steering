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
GEMMA = ROOT / 'outputs/gemma-grains-fp16-v2-20261029-1793246507'
PLAN = ROOT / 'outputs/qwen25-vl7b-next-plan-20261028'
GPU2 = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3 = 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
PAIR = GPU2 + ',' + GPU3
METHODS = ['base', 'revis', 'vti', 'rudder_clone', 'grains_clone']
SOURCES = ['src/vlm_grounding/backend.py', 'src/vlm_grounding/baselines.py',
    'src/vlm_grounding/cli.py', 'src/vlm_grounding/evaluation.py',
    'scripts/author_clone_bridge.py', 'scripts/qwen_author_compat.py',
    'scripts/qwen_gpu23_adapter.py', 'scripts/qwen_five_profile_api.py', 'scripts/qwen_five_profile_stage.py', 'scripts/preflight_qwen_authors.py',
    'scripts/fit_qwen_retained.py', 'scripts/extract_qwen_author_grains.py',
    'scripts/run_qwen_five_profile_worker.py', 'scripts/run_qwen_five_profile_suite.py',
    'scripts/audit_qwen_five_profile_suite.py', 'scripts/audit_hallucination_suite.py',
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
    assert len(tasks) == len({t['id'] for t in tasks})
    for bench, rows in datasets.items():
        for method in METHODS:
            selected = [t for t in tasks if t['benchmark'] == bench and t['method'] == method]
            assert [i for t in selected for i in range(t['start'], t['stop'])] == list(range(len(rows)))
            assert all(t['dtype'] == ('float16' if method == 'grains_clone' else 'bfloat16') for t in selected)
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
    from qwen_five_profile_api import require_gemma_complete
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
        'model': config['model'], 'methods': METHODS, 'method_dtypes': {m: 'float16' if m == 'grains_clone' else 'bfloat16' for m in METHODS}, 'base_fp32_enabled': False, 'mixed_precision_comparisons': ['grains_clone'], 'gpu_indices': [2, 3], 'other_gpu_fallback': False,
        'records_per_method': {'pope_adversarial': 3000, 'chair_coco_val2014': 500},
        'generation': parent['generation'], 'method_parameters': {**parent['method_parameters'], 'grains_clone': {**parent['method_parameters']['grains_clone'], 'dtype': 'float16', 'extraction_dtype': 'float32', 'steering_vector_dtype': 'float32'}},
        'comparisons': parent['comparisons'], 'dataset_provenance': parent['dataset_provenance'],
        'chair_scoring': parent['chair_scoring'], 'deferred': parent['deferred'],
        'source_commit': '1683e86223cdc491a2223d61ec63403bf37f7513', 'source_hashes': sources, 'author_sources': clones,
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


def available_gpus():
    rows=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    parsed=[tuple(x.strip() for x in row.split(',')) for row in rows]
    assert {(i,u) for i,u,m in parsed}=={('2',GPU2),('3',GPU3)}
    busy=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
    return [('gpu'+i,u) for i,u,m in parsed if u not in busy and int(m)>=21*1024]


def run(output):
    assert output.resolve().parent==ROOT/'outputs' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'outputs/.qwen-author-suite-controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir();state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'started_at':time.time(),
        'gpu_indices':[2,3],'base_fp32_enabled':False,'grains_inference_dtype':'float16','qwen_gpu_models_launched':False}
    jobs=[];preflight_jobs=[];children=[]
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'status.json',state)
    def free(required,phase):
        while True:
            available=available_gpus()
            if len(available)>=required:return available
            update(stage='waiting_for_available_gpu',next_phase=phase,required_empty_gpus=required,
                available_gpu_uuids=[u for label,u in available]);time.sleep(20)
    def launch(script,args,visible,kind,label,state_file,device_proof=None,child_output=output,**metadata):
        log=child_output/(label+'.log')
        with log.open('xb') as stream:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),script,'--output',str(child_output),*args],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record={'pid':child.pid,'start_ticks':ticks(child.pid),'script':script,'output':str(child_output),'kind':kind,
            'visible_devices':visible,'log':str(log),'state_file':state_file,'device_proof':device_proof,**metadata}
        (preflight_jobs if kind=='preflight' else jobs).append(record);children.append(child)
        save(output/('preflight-jobs.json' if kind=='preflight' else 'jobs.json'),preflight_jobs if kind=='preflight' else jobs)
        if visible:update(qwen_gpu_models_launched=True)
        return child,record
    def wait(group,phase):
        while True:
            complete=True;progress=[]
            for child,job in group:
                code=child.poll();complete &= code is not None
                file=Path(job['output'])/job['state_file'];s=json.loads(file.read_text()) if file.exists() else {}
                assert code in [None,0] and s.get('stage')!='failed' and s.get('status')!='failed',{'job':job,'code':code,'state':s}
                if code==0:
                    assert s.get('stage',s.get('status')) in ['complete','passed','verified_complete'],s
                    job.update(exit_code=0,state_sha256=sha(file))
                    if job['device_proof']:job['device_proof_sha256']=sha(Path(job['output'])/job['device_proof'])
                progress.append({'pid':job['pid'],'kind':job['kind'],'stage':s.get('stage',s.get('status')),
                    'completed':s.get('completed_records',s.get('completed')),'returncode':code})
            save(output/'jobs.json',jobs);save(output/'preflight-jobs.json',preflight_jobs)
            update(stage=phase,jobs=progress)
            if complete:return
            time.sleep(15)
    try:
        update(stage='cpu_preparation');checked=check_inputs();save(output/'cpu-preparation.json',checked)
        save(ROOT/'outputs/qwen-author-suite-current.json',{'output':str(output),'controller_pid':os.getpid(),
            'controller_start_ticks':state['controller_start_ticks'],'script':'scripts/run_qwen_five_profile_suite.py',
            'gpu_indices':[2,3],'base_fp32_enabled':False,'grains_inference_dtype':'float16'})
        while True:
            previous=json.loads((GEMMA/'status.json').read_text())
            assert previous['stage']!='failed',previous
            if previous['stage']=='complete':
                from qwen_five_profile_api import require_gemma_complete
                require_gemma_complete();break
            update(stage='waiting_for_gemma',gemma_stage=previous['stage'],qwen_gpu_models_launched=False);time.sleep(30)
        preflight=output.with_name(output.name+'-preflight');preflight.mkdir()
        for phase in ['bf16','fp32']:
            available=free(1 if phase=='bf16' else 2,'preflight_'+phase)
            visible=available[0][1] if phase=='bf16' else PAIR
            job=launch('scripts/qwen_five_profile_stage.py',['--stage','preflight_'+phase],visible,'preflight',phase,
                phase+'-preflight.json',phase+'-device-map.json',child_output=preflight,dtype='bfloat16' if phase=='bf16' else 'float32')
            wait([job],'preflight_'+phase)
        protocol=freeze_protocol(output,preflight);update(protocol_sha256=protocol['sha256'])
        pending=['revis','vti']
        while pending:
            group=[]
            for label,visible in free(1,'fit_retained'):
                if not pending:break
                method=pending.pop(0)
                group.append(launch('scripts/qwen_five_profile_stage.py',['--stage','fit_'+method],visible,'fit','fit-'+method,
                    'fit-'+method+'-status.json','fit-'+method+'-device-map.json',dtype='bfloat16'))
            wait(group,'fitting_retained')
        free(2,'extract_grains_fp32')
        job=launch('scripts/qwen_five_profile_stage.py',['--stage','extract'],PAIR,'extraction','grains-extraction',
            'extraction-status.json','grains-extraction-device-map.json',dtype='float32')
        wait([job],'fitting_grains_fp32')
        seals={}
        for method,file in [('revis',output/'artifacts/revis.pt'),('vti',output/'artifacts/vti.pt'),('grains_clone',output/'grains-author.pt')]:
            seal=json.loads((output/(('grains' if method=='grains_clone' else method)+'-artifact.json')).read_text())
            assert seal['status']=='verified_complete' and seal['source_protocol_sha256']==protocol['sha256']
            assert sha(file)==seal['artifact_sha256'];seals[method]={'sha256':sha(file)}
        save(output/'artifacts-ready.json',{'status':'verified_complete','source_protocol_sha256':protocol['sha256'],
            'methods':seals,'gemma_vectors_reused':False,'created_at':time.time()})
        available=free(1,'preflight_fp16')
        job=launch('scripts/qwen_five_profile_stage.py',['--stage','preflight_fp16'],available[0][1],'preflight_fp16',
            'fp16-preflight','fp16-preflight.json','fp16-preflight-device-map.json',dtype='float16')
        wait([job],'preflight_fp16')
        tag=str(int(time.time()))
        for dtype in ['bfloat16','float16']:
            group=[]
            for label,visible in free(1,'generate_'+dtype):
                worker=label+'-'+dtype+'-'+tag
                group.append(launch('scripts/run_qwen_five_profile_worker.py',['--worker',worker,'--dtype',dtype,
                    '--lease-epoch',dtype+'-'+tag],visible,'generation',worker,'worker-'+worker+'.json',
                    worker+'-device-map.json',dtype=dtype,worker=worker))
            wait(group,'generating_'+dtype)
        # Audit stays CPU-only and does not require unrelated GPUs to be empty.
        log=output/'official-audit.log';update(stage='auditing')
        with log.open('xb') as stream:
            result=subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/audit_qwen_five_profile_suite.py','--output',str(output)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/'report.json').read_text());assert report['status']=='verified_complete'
        assert report['audit']['all_17500_responses_verified'] and report['audit']['no_base_fp32_generated_or_included']
        update(stage='complete',completed_at=time.time(),report_sha256=sha(output/'report.json'))
    except BaseException:
        for child in children:
            if child.poll() is None:child.terminate()
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--check-preparation',action='store_true')
    args=parser.parse_args()
    if args.check_preparation:print(json.dumps(check_inputs()))
    else:assert args.output is not None;run(args.output)

