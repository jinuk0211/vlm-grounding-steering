"""Audit the five-method suite after the user-stopped author-VTI run was completed on GPU 1.

audit_qwen_vti_replacement.py is reused unchanged for parent assets, attention,
official scoring and bootstrap. Its VTI lifecycle and CHAIR-join checks assume
the original GPU0/2/3 jobs exited normally; here they were stopped by the user
(user-stop-proof-20261029.json), so the preserved prefixes are verified against
that proof and joined with the disjoint GPU 1 continuation in memory. No file
in the VTI subrun folder is rewritten; merged rows go to the new report folder.
"""
import argparse,json,math,os,time
from pathlib import Path

import audit_qwen_vti_replacement as base
from audit_qwen_vti_replacement import METHODS,VTI_COMMIT,obj,read,sealed,dead,device_proof
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_five_profile_api import CHECKPOINT,REVISION,require_gemma_complete

GPU1='GPU-5cf44493-6a32-f174-b98a-c8bc5702ee73'
GPU23={2:'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde',3:'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'}
STOP='user-stop-proof-20261029.json'
# Preserved prefixes at the user stop, in benchmark index order: (start,end,file,offset).
CHAIR_PREFIX=[(0,45,'chair-parallel/000-167.jsonl',0),(60,115,'chair-gpu23/gpu2.jsonl',0),
              (167,211,'chair-parallel/167-334.jsonl',0),(227,281,'chair-gpu23/gpu3.jsonl',0),
              (334,378,'chair-parallel/334-500.jsonl',0)]


def parent_lifecycle(parent,vti):
    """Parent donor-worker checks of base.lifecycle, unchanged up to the VTI resume controller."""
    transition=obj(vti/'transition.json');assert transition['all_selected_instances_exited']
    stopped={(j['pid'],j['start_ticks']):j for j in transition['stopped_jobs']};assert len(stopped)==2
    for j in stopped.values():dead(j)
    sc=transition['stopped_controller'];dead({'pid':sc['controller_pid'],'start_ticks':sc['controller_start_ticks']})
    continuation=obj(parent/'gpu0-continuation-amendment.json')
    failed_pre={(j['pid'],j['start_ticks']):j for j in continuation['failed_pre_model_jobs']}
    hardware=obj(parent/'gpu0-hardware-amendment.json');stop=obj(Path(hardware['stop_proof']))
    migrated={(j['pid'],j['start_ticks']) for j in stop['stopped_instances']}
    workers={};jobs=[];seen=set();history=[]
    for name in ['jobs.json','supplemental-jobs.json','fp16-stable-jobs.json']:
        for j in obj(parent/name):
            identity=(j['pid'],j['start_ticks']);assert identity not in seen;seen.add(identity);jobs.append(j);dead(j)
            state=obj(parent/j['state_file'])
            if identity in stopped:
                assert j==stopped[identity] and j['kind']=='generation'
                history.append({'pid':j['pid'],'outcome':'explicitly_superseded_supplemental_replica'})
            elif j.get('gpu0_pre_model_failure'):
                assert j==failed_pre[identity] and state['stage']=='failed' and state['completed']==38
                assert sha(parent/j['state_file'])==j['failed_state_sha256']
                history.append({'pid':j['pid'],'outcome':'preserved_pre_model_failure'});continue
            elif j.get('numerical_preflight_failed'):
                numerical=obj(parent/'gpu0-bf16-priority-amendment.json')
                assert state['stage']==state['status']=='failed' and state['pid']==j['pid']
                assert sha(parent/j['state_file'])==j['failed_state_sha256']==numerical['failed_preflight_state_sha256']
                history.append({'pid':j['pid'],'outcome':'preserved_native_fp16_preflight_failure'});continue
            elif j.get('gpu0_migration_stopped'):
                assert identity in migrated and j['kind']=='extraction' and state['completed']==stop['completed_ig_records']
                history.append({'pid':j['pid'],'outcome':'historical_extraction_stopped_for_gpu0_migration'});continue
            else:
                assert state['pid']==j['pid'] and state['stage']=='complete'
                assert sha(parent/j['state_file'])==j['state_sha256']
                if j.get('direct_launch'):
                    assert {'pid':j['pid'],'start_ticks':j['start_ticks']}==continuation['adopted_extraction_identity']
                    assert j['adopted_completion_observed'] and not j['exit_code_observed'] and state['completed']==50
                    history.append({'pid':j['pid'],'outcome':'adopted_extraction_complete_exit_code_unavailable'})
                else:assert j['exit_code']==0
            if j['kind']=='generation':
                device_proof(parent/j['device_proof'],j['dtype']);assert j['visible_devices']==GPU0
                if identity not in stopped:assert sha(parent/j['device_proof'])==j['device_proof_sha256']
                workers[j['worker']]=j
    extra_jobs,extra_state,amendment=(obj(vti/n) for n in ['grains-extra-jobs.json','grains-extra-status.json','grains-extra-amendment.json'])
    assert len(extra_jobs)==2 and amendment['gpu_indices']==[0]
    assert amendment['parent_protocol_sha256']==obj(parent/'protocol.json')['sha256']
    assert amendment['vti_protocol_sha256']==obj(vti/'protocol.json')['sha256']
    previous_finalizer=obj(vti/'grains-extra-finalizer-transition.json')
    assert previous_finalizer['all_selected_instances_exited'];dead(previous_finalizer['previous_finalizer'])
    assert sha(vti/'grains-extra-finalizer-transition.json')==amendment['previous_finalizer_stop_proof_sha256']
    primary=[j for j in jobs if j['kind']=='generation' and j['dtype']=='float16']
    assert len(primary)==1 and amendment['same_lease_epoch']==primary[0]['worker']
    worker_source='scripts/qwen_fp16_stable_stage.py'
    snapshot=(vti/amendment['source_snapshot_dir']).resolve();assert snapshot.is_relative_to(vti.resolve())
    for path,digest in amendment['source_hashes'].items():
        assert sha(snapshot/path)==digest,path
        if path not in ('scripts/audit_qwen_vti_replacement.py','scripts/finalize_qwen_vti_replacement.py'):
            assert sha(ROOT/path)==digest,path
    assert amendment['source_hashes'][worker_source]==obj(parent/'fp16-attention-amendment.json')['source_hashes'][worker_source]
    assert extra_state['stage']=='complete'
    dead({'pid':extra_state['controller_pid'],'start_ticks':extra_state['controller_start_ticks']})
    for j in extra_jobs:
        identity=(j['pid'],j['start_ticks']);assert identity not in seen;seen.add(identity);dead(j)
        assert j['kind']=='generation' and j['dtype']=='float16' and j['visible_devices']==GPU0
        assert j['source_file']==j['script']==worker_source and j['source_sha256']==amendment['source_hashes'][worker_source]
        assert j['lease_epoch']==amendment['same_lease_epoch'] and j['output']==str(parent) and j['worker'] not in workers
        s=obj(parent/j['state_file'])
        assert s['pid']==j['pid'] and s['worker']==j['worker'] and s['stage']=='complete'
        assert s['dtype']=='float16' and s['gpu_visible_devices']==GPU0
        assert j['exit_code']==0 and sha(parent/j['state_file'])==j['state_sha256']
        d=device_proof(parent/j['device_proof'],'float16');assert sha(parent/j['device_proof'])==j['device_proof_sha256']
        assert d['attention_score_dtype']=='torch.float32' and d['attention_output_dtype']=='torch.float16'
        assert d['fp16_attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
        workers[j['worker']]=j
        history.append({'pid':j['pid'],'start_ticks':j['start_ticks'],'worker':j['worker'],
            'outcome':'extra_original_grains_generation_complete','amendment_sha256':sha(vti/'grains-extra-amendment.json')})
    for j in obj(parent/'preflight-jobs.json'):
        dead(j);folder=Path(j['output']);assert obj(folder/j['state_file'])['status']=='passed' and j['exit_code']==0
        assert sha(folder/j['state_file'])==j['state_sha256'] and sha(folder/j['device_proof'])==j['device_proof_sha256']
    for folder,name in [(parent,'status.json'),(parent,'fp16-stable-status.json')]:
        s=obj(folder/name);dead({'pid':s['controller_pid'],'start_ticks':s['controller_start_ticks']})
        assert s['stage'] in ['complete','bf16_generation_complete_fp16_pending']
    for j in obj(parent/'vti-author-old-watchers-superseded.json')['stopped']:dead(j)
    return workers,history


def vti_lifecycle(parent,vti,workers,history):
    """VTI jobs: every selected instance was stopped by the user; the donor recovery had completed."""
    stop=obj(vti/STOP);assert stop['all_selected_instances_exited'] and all(stop['process_exited'].values())
    for s in stop['selected']:dead(s)
    for name,info in stop['files'].items():assert sha(vti/name)==info['sha256'],name
    resumed=obj(vti/'resumed-jobs.json');ctrl=obj(vti/'resume-controller-status.json')
    author=resumed['author_job'];dead(author);dead(ctrl)
    selected={s['name']:(s['pid'],s['start_ticks']) for s in stop['selected']}
    assert selected['original_vti']==(author['pid'],author['start_ticks'])
    assert selected['resume_controller']==(ctrl['pid'],ctrl['start_ticks'])
    state=obj(vti/'status.json');assert state==stop['status_snapshots']['original']
    assert state['pid']==author['pid'] and state['benchmark']=='pope' and state['benchmark_completed']==2851
    assert author['source_sha256']==sha(ROOT/'scripts/run_qwen_original_vti.py')
    assert ctrl['recovery_stage']=='complete' and not ctrl['recovery_live']
    recovery=resumed['adopted_recovery_job'];dead(recovery)
    rs=obj(parent/recovery['state_file']);assert rs['pid']==recovery['pid'] and rs['stage']=='complete'
    assert rs['completed_records']==ctrl['recovery_completed']
    assert recovery['source_sha256']==sha(ROOT/recovery['args'][0])
    name=recovery['args'][recovery['args'].index('--worker')+1]
    assert recovery['args'][recovery['args'].index('--dtype')+1]=='bfloat16'
    assert rs['dtype']=='bfloat16' and rs['gpu_visible_devices']==GPU0
    device_proof(parent/(name+'-device-map.json'),'bfloat16')
    workers[name]={**recovery,'dtype':'bfloat16','visible_devices':GPU0,'worker':name}
    previous=resumed['failed_author_job'];dead(previous)
    failed=obj(vti/('failed-start-'+str(previous['pid']))/'status.json')
    assert failed['pid']==previous['pid'] and failed['stage']=='failed' and failed['completed_records']==0
    history.extend([{'pid':previous['pid'],'outcome':'failed_vti_start_before_predictions'},
        {'pid':recovery['pid'],'outcome':'adopted_recovery_complete_exit_code_unavailable'},
        {'pid':author['pid'],'outcome':'user_stopped_after_2851_pope_rows_prefix_preserved'},
        {'stop_proof_sha256':sha(vti/STOP),'outcome':'user_stopped_gpu0_gpu2_gpu3_chair_workers_prefixes_preserved'}])
    return workers,history


def vti_assets(parent,vti,protocol,datasets,development_hashes):
    """base.vti_assets without the completed-run field source_hashes_after (run was user-stopped)."""
    from run_qwen_original_vti import verify_author
    state=obj(vti/'status.json');assert 'source_hashes_after' not in state
    patched=dict(state,source_hashes_after=verify_author())
    original=base.obj
    base.obj=lambda p:patched if Path(p)==vti/'status.json' else original(p)
    try:return base.vti_assets(parent,vti,protocol,datasets,development_hashes)
    finally:base.obj=original


def merged_vti_rows(parent,vti,protocol,vti_protocol,digest,datasets):
    stop=obj(vti/STOP);amendment=obj(vti/'gpu1-amendment.json')
    assert amendment['parent_protocol_sha256']==protocol['sha256'] and amendment['vti_protocol_sha256']==vti_protocol['sha256']
    assert amendment['artifact_sha256']==digest and amendment['gpu_uuid']==GPU1 and amendment['gpu_index']==1
    assert amendment['stop_proof_sha256']==sha(vti/STOP)
    assert amendment['preserved_prefix_files']=={n:i['sha256'] for n,i in stop['files'].items()}
    snapshot=(vti/amendment['source_snapshot_dir']).resolve();assert snapshot.is_relative_to(vti.resolve())
    for source,d in amendment['source_hashes'].items():assert sha(snapshot/source)==d==sha(ROOT/source),source
    amendment_sha=sha(vti/'gpu1-amendment.json')
    # Preserved GPU0 / GPU2 / GPU3 worker identities and per-row provenance.
    for label in ['000-167','167-334','334-500']:
        w=obj(vti/'chair-parallel'/(label+'-status.json'));dead(w)
        assert w['gpu_uuid']==GPU0 and w['artifact_sha256']==digest
        assert w['original_worker_sha256']==sha(ROOT/'scripts/run_qwen_original_vti.py')
        device_proof(vti/'chair-parallel'/(label+'-device-map.json'),'float16')
    gpu23=obj(vti/'chair-gpu23-amendment.json')
    assert gpu23['artifact_sha256']==digest and gpu23['gpu_uuids']=={str(k):v for k,v in GPU23.items()}
    for index in (2,3):
        w=obj(vti/'chair-gpu23'/f'gpu{index}-status.json');dead(w)
        assert w==stop['status_snapshots'][f'gpu{index}'] and w['gpu_uuid']==GPU23[index]
        assert w['amendment_sha256']==sha(vti/'chair-gpu23-amendment.json')
        assert w['source_sha256']==gpu23['source_hashes']['scripts/qwen_vti_chair_gpu23.py']
        p=obj(vti/'chair-gpu23'/f'gpu{index}-device-map.json')
        assert p['physical_gpu_index']==index and p['visible_devices']==GPU23[index] and p['dtype']=='float16'
        assert p['parameter_devices']==['cuda:0'] and p['checkpoint']==CHECKPOINT and p['revision']==REVISION
        assert not p['quantization'] and not p['cpu_or_disk_offload']
    # GPU 1 shards: complete, exited, hashed, device-proven.
    jobs=obj(vti/'gpu1/jobs.json');shards={}
    assert sorted(j['shard'] for j in jobs)==sorted(int(k) for k in amendment['shards'])
    for j in jobs:
        dead(j);k=j['shard'];s=obj(vti/'gpu1'/f'shard{k}-status.json')
        assert s['pid']==j['pid'] and s['stage']=='complete' and s['gpu_uuid']==GPU1==j['visible_devices']
        assert s['amendment_sha256']==amendment_sha and s['artifact_sha256']==digest
        assert s['source_sha256']==amendment['source_hashes']['scripts/qwen_vti_gpu1.py']==j['source_sha256']
        assert s['tasks']==amendment['shards'][str(k)]
        path=vti/'gpu1'/f'shard{k}.jsonl';assert s['output_sha256']==sha(path)
        assert s['device_proof_sha256']==sha(vti/'gpu1'/f'shard{k}-device-map.json')
        p=obj(vti/'gpu1'/f'shard{k}-device-map.json')
        assert p['physical_gpu_index']==1 and p['visible_devices']==GPU1 and p['dtype']=='float16'
        assert p['parameter_devices']==['cuda:0'] and p['parameter_dtypes']==['torch.float16']
        assert p['checkpoint']==CHECKPOINT and p['revision']==REVISION and not p['quantization'] and not p['cpu_or_disk_offload']
        rows=read(path);expected=[(b,i) for b,a,e in s['tasks'] for i in range(a,e)]
        assert len(rows)==len(expected)==s['completed_records']
        for row,(b,i) in zip(rows,expected):
            assert row['id']==datasets[b][i]['id'];shards.setdefault(b,{})[i]=row
    result={};sources={}
    pope_prefix=read(vti/'pope.jsonl');assert len(pope_prefix)==2851
    result['pope']=pope_prefix+[shards['pope'][i] for i in range(2851,3000)]
    sources['pope']=[[0,2851,'gpu0','pope.jsonl'],[2851,3000,'gpu1','gpu1/shard0.jsonl']]
    chair=dict(shards['chair']);sources['chair']=[]
    for a,e,name,offset in CHAIR_PREFIX:
        part=read(vti/name)[offset:offset+e-a];assert len(part)==e-a
        for i,row in zip(range(a,e),part):assert i not in chair;chair[i]=row
        sources['chair'].append([a,e,'gpu0' if name.startswith('chair-parallel') else Path(name).stem,name])
    assert sorted(chair)==list(range(500));result['chair']=[chair[i] for i in range(500)]
    counts={}
    for b,rows in result.items():
        assert len(rows)==len(datasets[b])
        for row,ref in zip(rows,datasets[b]):assert all(row.get(k)==v for k,v in ref.items())
        counts[b]={g:sum(r['gpu_uuid']==u for r in rows) for g,u in [('gpu0',GPU0),('gpu1',GPU1),('gpu2',GPU23[2]),('gpu3',GPU23[3])]}
        assert sum(counts[b].values())==len(rows)
    assert counts['pope']=={'gpu0':2851,'gpu1':149,'gpu2':0,'gpu3':0}
    assert counts['chair']=={'gpu0':133,'gpu1':258,'gpu2':55,'gpu3':54}
    names={'gpu0':obj(vti/'chair-parallel/000-167-device-map.json').get('gpu_name'),
           'gpu1':obj(vti/'gpu1/shard0-device-map.json')['gpu_name'],
           'gpu2':obj(vti/'chair-gpu23/gpu2-device-map.json')['gpu_name'],
           'gpu3':obj(vti/'chair-gpu23/gpu3-device-map.json')['gpu_name']}
    provenance={'status':'verified','stop_proof_sha256':sha(vti/STOP),'gpu1_amendment_sha256':amendment_sha,
        'records_by_gpu':counts,'gpu_names':names,'sources':sources,
        'gpu_uuids':{'gpu0':GPU0,'gpu1':GPU1,'gpu2':GPU23[2],'gpu3':GPU23[3]}}
    return result,provenance


def rows(parent,vti,protocol,vti_protocol,datasets,seals,workers,vti_digest,vti_rows):
    """base.rows with merged VTI rows and the GPU 0/1/2/3 per-row VTI provenance allow-list."""
    gpu1_sha=sha(vti/'gpu1-amendment.json');gpu23_sha=sha(vti/'chair-gpu23-amendment.json')
    grouped={b:{m:{} for m in METHODS} for b in datasets};input_hashes={}
    tasks=obj(parent/'tasks.json');assert sha(parent/'tasks.json')==protocol['tasks_sha256']
    assert all(t['method']!='base_fp32' for t in tasks)
    def validate(row,ref,method,bench):
        assert all(row.get(k)==v for k,v in ref.items())
        assert row['method']==method and row['checkpoint']==CHECKPOINT and row['revision']==REVISION
        assert isinstance(row['prediction'],str)
        if method=='vti_author' and row['gpu_uuid']==GPU1:assert row['gpu1_hardware_amendment_sha256']==gpu1_sha
        elif method=='vti_author' and row['gpu_uuid'] in GPU23.values():
            assert bench=='chair' and row['gpu23_hardware_amendment_sha256']==gpu23_sha
        else:assert row['gpu_uuid']==GPU0
        assert type(row['generated_tokens']) is int and 0<row['generated_tokens']<=ref['max_new_tokens']
        assert ref['max_new_tokens']==(128 if bench=='pope' else 512)
        assert math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0
        assert row['dtype']==('float16' if method in ['grains_clone','vti_author'] else 'bfloat16')
        if method=='vti_author':
            assert row['protocol_sha256']==vti_protocol['sha256'] and row['parent_protocol_sha256']==protocol['sha256']
            assert row['author_commit']==VTI_COMMIT and row['artifact_sha256']==vti_digest
            assert row['attention_score_dtype']=='float32' and row['attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
            assert row['original_hook_file'].endswith('VTI/vti_utils/llm_layers.py')
            for kind in ['vision','text']:
                d=row['original_vti_diagnostics'][kind]
                assert d['calls']>0 and d['changed_elements']>0 and math.isfinite(d['max_abs_delta']) and d['max_abs_delta']>0
        else:
            assert row['protocol_sha256']==protocol['sha256'] and row['model_family']=='qwen2_5_vl'
            assert row['worker'] in workers and row['artifact_sha256']==seals['methods'].get(method,{}).get('sha256')
            worker=workers[row['worker']]
            assert row['dtype']==worker['dtype'] and row['gpu_uuid']==worker['visible_devices']
            assert 'adopted_cache_sha256' not in row
            if method=='grains_clone':
                assert row['attention_score_dtype']=='torch.float32'
                assert row['fp16_attention_amendment_sha256']==sha(parent/'fp16-attention-amendment.json')
            if method in ['rudder_clone','grains_clone']:
                d=row['author_diagnostics'];assert d['calls']>0
                if method=='rudder_clone':
                    assert d['active_positions']>0 and d['card_norm']>0
                    assert d['layer']==(1 if bench=='pope' else 25) and d['alpha']==(8.0 if bench=='pope' else 5.0)
                    assert d['hook_source'].endswith('RUDDER/Qwen/qwen_methods.py')
                else:
                    assert d['layer']==27 and d['alpha']==10 and d['hook_source'].endswith('GrAInS/src/utils/steering.py')
        bucket=grouped[bench][method];assert row['id'] not in bucket;bucket[row['id']]=row
    for task in tasks:
        if task['method']=='vti':continue
        assert task['method'] in METHODS[:-1]
        path=parent/'predictions'/(task['id']+'.jsonl');complete=obj(parent/'claims'/task['id']/'complete.json')
        assert sha(path)==complete['sha256'];part=read(path);refs=datasets[task['benchmark']][task['start']:task['stop']]
        assert len(part)==len(refs)==complete['records']
        for row,ref in zip(part,refs):validate(row,ref,task['method'],task['benchmark'])
        input_hashes[str(path)]=sha(path)
    for bench,part in vti_rows.items():
        assert len(part)==len(datasets[bench])
        for row,ref in zip(part,datasets[bench]):validate(row,ref,'vti_author',bench)
    for bench,methods in grouped.items():
        expected=[r['id'] for r in datasets[bench]]
        for method,bucket in methods.items():
            assert set(bucket)==set(expected);methods[method]=[bucket[i] for i in expected]
    assert sum(len(r) for methods in grouped.values() for r in methods.values())==17500
    return grouped,input_hashes


def audit(parent,vti,output,precheck=False):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='','Auditor is CPU-only'
    parent=parent.resolve();vti=vti.resolve();output=output.resolve()
    assert parent.parent==vti.parent==ROOT/'outputs' and output.parent==parent
    assert not output.exists(),'Use a new combined-report folder; do not overwrite a report'
    require_gemma_complete();protocol=verify_protocol(parent);sealed(protocol)
    vti_protocol=obj(vti/'protocol.json');sealed(vti_protocol)
    current=obj(parent/'original-vti-current.json')
    assert current['protocol_sha256']==vti_protocol['sha256'] and Path(current['output'])==vti
    workers,history=parent_lifecycle(parent,vti)
    workers,history=vti_lifecycle(parent,vti,workers,history)
    datasets,seals,development_hashes=base.parent_assets(parent,protocol)
    attention=base.validate_attention(parent)
    digest=vti_assets(parent,vti,vti_protocol,datasets,development_hashes)
    if precheck:
        print(json.dumps({'status':'precheck_passed','workers':len(workers),'artifact_sha256':digest}));return
    vti_rows,provenance=merged_vti_rows(parent,vti,protocol,vti_protocol,digest,datasets)
    grouped,input_hashes=rows(parent,vti,protocol,vti_protocol,datasets,seals,workers,digest,vti_rows)
    assert sum(len(r) for m in grouped.values() for r in m.values())==17500
    output.mkdir()
    merged=output/'vti-author-merged';merged.mkdir()
    for b,part in vti_rows.items():
        with (merged/(b+'.jsonl')).open('x') as sink:
            for row in part:sink.write(json.dumps(row,allow_nan=False)+'\n')
        input_hashes[str(merged/(b+'.jsonl'))]=sha(merged/(b+'.jsonl'))
    metrics,uncertainty=base.score(output,protocol,grouped)
    gpus=[0,1,2,3]
    report={'status':'verified_complete','scope':{'methods':METHODS,'pope':'COCO adversarial 3000 questions','chair':'fixed 500 COCO val2014 images','total_responses':17500},
        'methods':METHODS,'protocol':protocol,'vti_author_protocol':vti_protocol,
        'vti_author':{'author_commit':VTI_COMMIT,'artifact_sha256':digest,'source_output':str(vti),
            'original_function_calls_and_finite_nonzero_directions_verified':True,'independent_pca_reconstruction_available':False,
            'pca_reconstruction_limitation':'VTI extraction did not persist all raw fit hidden states; original sources, directions, calls and provenance are verified.'},
        'metrics':metrics,'uncertainty':uncertainty,'fp16_attention_amendment':attention,
        'vti_gpu_provenance':provenance,
        'source_output':str(parent),'output':str(output),'created_at':time.time(),
        'input_prediction_sha256':input_hashes,'execution_history':history,
        'analysis_hashes':{'auditor':sha(__file__),'base_auditor':sha(base.__file__),
            'official_metric_helper':sha(ROOT/'scripts/audit_hallucination_suite.py')},
        'audit':{'all_17500_responses_verified':True,'official_scorers_used':True,'actual_workers_exited':True,
            'original_author_sources_unmodified':True,'source_input_artifact_hashes_verified':True,
            'fit50_raw_ig_hidden_and_pca_independently_verified':True,'only_gpu0_used_for_benchmark_responses':False,
            'benchmark_response_gpu_indices':gpus,'vti_records_by_gpu':provenance['records_by_gpu'],
            'vti_gpu_names':provenance['gpu_names'],'user_stop_prefixes_preserved_byte_identical':True,
            'no_base_fp32_generated_or_included':True,'no_gemma_responses_or_vectors_reused':True,
            'development_test_image_hash_overlap':0,'vti_demo_test_image_hash_overlap':0,
            'matched_precision_baselines':False,'mixed_precision_comparisons':['grains_clone','vti_author'],
            'excluded_historical_local_vti':True,'historical_parent_protocol_and_gpu23_ig_provenance_retained':True}}
    labels={'base':'Base','revis':'REVIS 자체 구현','rudder_clone':'RUDDER 원본 함수 연결','grains_clone':'GrAInS 원본 함수 연결','vti_author':'VTI 원저자 구현 연결'}
    c=provenance['records_by_gpu'];n=provenance['gpu_names']
    md=['# Qwen2.5-VL-7B · POPE adversarial / CHAIR','','각 방법 POPE 3,000문항·CHAIR 500장, 전체 17,500응답. 수치는 %.','',
        '| 방법 | 모델 정밀도 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |','|---|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        p,q=metrics['pope'][m],metrics['chair'][m];dtype='FP16' if m in ['grains_clone','vti_author'] else 'BF16'
        md.append(f"| {labels[m]} | {dtype} | {100*p['accuracy']:.2f} | {100*p['f1']:.2f} | {100*q['CHAIR_s']:.2f} | {100*q['CHAIR_i']:.2f} | {100*q['object_recall']:.2f} |")
    md.extend(['',f"VTI 응답 생성 GPU: POPE는 GPU0 {c['pope']['gpu0']}·GPU1 {c['pope']['gpu1']}문항, CHAIR는 GPU0 {c['chair']['gpu0']}·GPU1 {c['chair']['gpu1']}·GPU2 {c['chair']['gpu2']}·GPU3 {c['chair']['gpu3']}장이다. "
        f"GPU 종류: GPU0 {n['gpu0']}, GPU1 {n['gpu1']}, GPU2 {n['gpu2']}, GPU3 {n['gpu3']}. 가중치·수식·데이터·벡터·정밀도는 동일하지만 FP16 수치는 GPU 종류에 따라 미세하게 다를 수 있다. 각 row에 gpu_uuid를 기록했다.",
        '','원래 작업은 사용자 요청으로 중지되었고(user-stop-proof-20261029.json), 중지 시점의 출력은 바이트 단위로 보존해 검증했다. 남은 POPE 149문항과 CHAIR 258장은 GPU1에서 겹치지 않게 이어 생성했다.',
        '','VTI 자체 구현의 이전 점수는 제외했다. VTI는 고정 저자 commit의 PCA·방향 생성·MLP 개입·정규화 함수를 직접 호출하고 Qwen2.5 입출력과 ViT 배치 차원을 연결했다. 저자 데모 70개·seed42·마스크99%·각50회·rank1·시각/텍스트 강도0.9를 사용했다. 저자 저장소는 Qwen2.5 POPE/CHAIR 실행 경로를 제공하지 않으므로 논문 전체 환경 재현으로 표현하지 않는다.',
        '','GrAInS·VTI 모델 가중치와 attention 출력은 FP16, QK·mask·softmax 점수는 overflow 방지를 위해 FP32다. 다른 세 방법은 BF16이므로 방법 비교에는 정밀도 영향도 포함된다. REVIS는 자체 구현이다. RUDDER·GrAInS는 고정 저자 원본 함수를 호출한다.',
        '','공식 POPE evaluate.py 및 원본 CHAIR Python3 포트로 채점했다. 이미지 단위 paired bootstrap 10,000회의 95% 구간은 report.json에 있다.',
        '','VTI는 원시 fit hidden states를 모두 보존하지 않아 PCA 자체의 독립 재구성은 할 수 없다. 종료 코드를 직접 관측하지 못한 인계 작업은 그 한계를 execution_history에 표시했다.'])
    (output/'REPORT_KO.md').write_text('\n'.join(md)+'\n')
    save(output/'report.json',report)
    print(json.dumps({'status':'verified_complete','output':str(output),'report_sha256':sha(output/'report.json'),'metrics':metrics},allow_nan=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--vti',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--precheck',action='store_true')
    a=p.parse_args();audit(a.parent,a.vti,a.output,a.precheck)
