"""CPU watcher and official interim scoring for complete Base/REVIS/RUDDER.

This is a 10,500-response interim cohort. It neither consumes old local VTI
nor declares the complete five-method evaluation finished.
"""
import argparse,fcntl,json,math,os,time,traceback
from pathlib import Path
import torch
from audit_qwen_vti_replacement import obj,read,sealed,live,device_proof,score
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_five_profile_api import CHECKPOINT,REVISION
from author_clone_bridge import verify_clone_sources
from vlm_grounding.cli import _validate_artifact

METHODS=['base','revis','rudder_clone']
PARENT=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
VTI=ROOT/'outputs/qwen-original-vti-1793260357'


def worker_catalog(parent,vti):
    workers={}
    for name in ['jobs.json','supplemental-jobs.json']:
        for j in obj(parent/name):
            if j['kind']!='generation' or j['dtype']!='bfloat16':continue
            assert j['worker'] not in workers and j['visible_devices']==GPU0
            workers[j['worker']]={**j,'device_proof_path':str(parent/j['device_proof'])}
    recovery=obj(vti/'resumed-jobs.json')['adopted_recovery_job'];args=recovery['args']
    assert args[args.index('--output')+1]==str(parent)
    assert args[args.index('--dtype')+1]=='bfloat16'
    assert recovery['gpu_uuid']==GPU0 and recovery['source_sha256']==sha(ROOT/args[0])
    name=args[args.index('--worker')+1];assert name not in workers
    workers[name]={**recovery,'worker':name,'dtype':'bfloat16','visible_devices':GPU0,
        'device_proof_path':str(parent/(name+'-device-map.json'))}
    assert workers
    return workers


def validate_assets(parent):
    protocol=verify_protocol(parent);sealed(protocol)
    from audit_qwen_gpu0_final import verify_final_sources
    verify_final_sources(parent)
    for name,digest in protocol['source_hashes'].items():
        assert sha(ROOT/name)==digest==sha(parent/'source-snapshot'/name)
    assert verify_clone_sources()==protocol['author_sources']
    for name,info in protocol['author_sources'].items():
        for path,digest in info['files'].items():assert sha(parent/'author-snapshot'/name/path)==digest
    for name,digest in protocol['manifest_hashes'].items():assert sha(parent/'data'/name)==digest
    assert sha(parent/'tasks.json')==protocol['tasks_sha256']
    weight=parent/'proofs/weight-cache-verification.json'
    assert sha(weight)==protocol['cached_weight_verification_sha256'] and obj(weight)['all_hf_lfs_sha256_verified']
    data={b:read(parent/'data'/(b+'.jsonl')) for b in ['pope','chair']}
    fit=read(parent/'data/fit.jsonl');cal=read(parent/'data/calibration.jsonl')
    assert len(fit)==len(cal)==32 and len({r['group_id'] for r in fit+cal})==64
    dev_hashes={sha(r['image']) for r in fit+cal};assert len(dev_hashes)==64
    for bench,rows in data.items():
        assert len(rows)==len({r['id'] for r in rows})==(3000 if bench=='pope' else 500)
        assert not dev_hashes&{r['image_sha256'] for r in rows}
        for path,digest in {(r['image'],r['image_sha256']) for r in rows}:assert sha(path)==digest
        assert all(r['max_new_tokens']==(128 if bench=='pope' else 512) for r in rows)
    config=obj(parent/'config.json');seals=obj(parent/'artifacts-ready.json')
    assert seals['status']=='verified_complete' and seals['source_protocol_sha256']==protocol['sha256']
    path=parent/'artifacts/revis.pt';assert sha(path)==seals['methods']['revis']['sha256']
    side=obj(parent/'revis-artifact.json')
    assert side['status']=='verified_complete' and side['source_protocol_sha256']==protocol['sha256']
    assert side['model']==config['model'] and not side['gemma_vectors_reused']
    assert side['records']==[r['id'] for r in fit+cal]
    for p,digest in side['input_hashes'].items():assert sha(p)==digest
    artifact=torch.load(path,map_location='cpu',weights_only=True);_validate_artifact(artifact,config['model'])
    assert all(torch.isfinite(v).all() for v in artifact['vectors'].values())
    assert artifact['vectors'][artifact['layer']].norm()>0
    return protocol,data,seals


def canonical(parent,protocol,data,seals,workers,tasks):
    grouped={b:{m:{} for m in METHODS} for b in data};inputs={};proofs={}
    for name,j in workers.items():
        d=device_proof(j['device_proof_path'],'bfloat16')
        assert d['hardware_amendment_sha256']==sha(parent/'gpu0-hardware-amendment.json')
        proofs[name]={'pid':j['pid'],'start_ticks':j['start_ticks'],'dtype':j['dtype'],
            'device_proof':j['device_proof_path'],'device_proof_sha256':sha(j['device_proof_path'])}
    for task in tasks:
        assert task['method'] in METHODS and task['dtype']=='bfloat16'
        path=parent/'predictions'/(task['id']+'.jsonl');marker=parent/'claims'/task['id']/'complete.json'
        complete=obj(marker);assert complete['sha256']==sha(path)
        part=read(path);refs=data[task['benchmark']][task['start']:task['stop']]
        assert len(part)==len(refs)==complete['records']
        for row,ref in zip(part,refs):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['method']==task['method'] and row['dtype']=='bfloat16'
            assert row['protocol_sha256']==protocol['sha256'] and row['model_family']=='qwen2_5_vl'
            assert row['checkpoint']==CHECKPOINT and row['revision']==REVISION
            assert row['worker'] in workers and row['gpu_uuid']==GPU0
            worker=workers[row['worker']];assert row['dtype']==worker['dtype'] and row['gpu_uuid']==worker['visible_devices']
            assert row['artifact_sha256']==seals['methods'].get(row['method'],{}).get('sha256')
            assert 'adopted_cache_sha256' not in row and isinstance(row['prediction'],str)
            assert type(row['generated_tokens']) is int and 0<row['generated_tokens']<=ref['max_new_tokens']
            assert math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0
            if row['method']=='rudder_clone':
                d=row['author_diagnostics'];assert d['calls']>0 and d['active_positions']>0 and d['card_norm']>0
                assert d['layer']==(1 if task['benchmark']=='pope' else 25)
                assert d['alpha']==(8.0 if task['benchmark']=='pope' else 5.0)
                assert d['hook_source'].endswith('RUDDER/Qwen/qwen_methods.py')
            bucket=grouped[task['benchmark']][task['method']];assert row['id'] not in bucket;bucket[row['id']]=row
        inputs[task['id']]={'prediction_sha256':sha(path),'complete_marker_sha256':sha(marker),'records':len(part)}
    for bench,methods in grouped.items():
        ids=[r['id'] for r in data[bench]]
        for method,bucket in methods.items():
            assert set(bucket)==set(ids);methods[method]=[bucket[i] for i in ids]
    assert sum(len(rows) for methods in grouped.values() for rows in methods.values())==10500
    return grouped,inputs,proofs


def main(parent,vti,output,poll_seconds=20):
    assert os.environ.get('CUDA_VISIBLE_DEVICES','')=='','Interim scorer is CPU-only'
    parent=parent.resolve();vti=vti.resolve();output=output.resolve()
    assert parent.parent==vti.parent==ROOT/'outputs' and output.parent==parent
    output.mkdir(exist_ok=True);assert not (output/'summary.json').exists(),'Interim summary already exists'
    assert not any((output/b).exists() for b in ['pope','chair']),'Use a new output after an interrupted scoring attempt'
    lock=(output/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script_digest=sha(__file__);scoring_digest=sha(ROOT/'scripts/audit_qwen_vti_replacement.py')
    fields=(Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()
    state={'pid':os.getpid(),'start_ticks':int(fields[19]),'started_at':time.time(),'methods':METHODS,'scope':'interim_three_bf16_profiles','goal_complete':False}
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'watcher-status.json',state)
    try:
        protocol=verify_protocol(parent);sealed(protocol)
        tasks=[t for t in obj(parent/'tasks.json') if t['method'] in METHODS]
        assert tasks and {t['method'] for t in tasks}==set(METHODS)
        assert sum(t['stop']-t['start'] for t in tasks)==10500
        while True:
            workers=worker_catalog(parent,vti)
            pending=[t['id'] for t in tasks if not (parent/'claims'/t['id']/'complete.json').is_file()]
            handles=[{'worker':n,'pid':j['pid'],'start_ticks':j['start_ticks'],'live':live(j['pid'],j['start_ticks'])} for n,j in workers.items()]
            update(stage='waiting_required_complete_claims',pending_tasks=len(pending),pending_task_ids=pending,
                observed_worker_handles=handles,completed_required_tasks=len(tasks)-len(pending),required_tasks=len(tasks))
            if not pending:break
            assert any(h['live'] for h in handles),{'error':'All known BF16 workers exited with required task claims missing','missing':pending}
            time.sleep(poll_seconds)
        update(stage='validating_complete_three_method_cohort')
        protocol,data,seals=validate_assets(parent)
        grouped,inputs,proofs=canonical(parent,protocol,data,seals,workers,tasks)
        update(stage='official_scoring_and_paired_bootstrap',selected_responses=10500)
        assert sha(__file__)==script_digest and sha(ROOT/'scripts/audit_qwen_vti_replacement.py')==scoring_digest
        metrics,uncertainty=score(output,protocol,grouped)
        paired={}
        for bench,methods in grouped.items():
            base=methods['base'];paired[bench]={}
            for method in METHODS[1:]:
                paired[bench][method]={'n':len(base),'different_prediction_strings':sum(r['prediction']!=b['prediction'] for r,b in zip(methods[method],base))}
        summary={'status':'interim_three_bf16_officially_scored','scope':{'methods':METHODS,'pope_per_method':3000,'chair_per_method':500,'selected_responses':10500},
            'full_five_method_suite_complete':False,'goal_complete':False,'remaining_methods':['grains_clone','vti_author'],
            'historical_local_vti_excluded':True,'protocol':protocol,'gpu_indices':[0],'model_dtype':'bfloat16',
            'metrics':metrics,'uncertainty':uncertainty,'paired_against_base':paired,
            'all_10500_selected_responses_verified':True,'official_pope_and_chair_scorers_used':True,
            'paired_image_bootstrap_samples':10000,'input_tasks':inputs,'worker_device_proofs':proofs,
            'auditor_sha256':script_digest,'scoring_helper_sha256':scoring_digest,
            'output':str(output),'created_at':time.time()}
        labels={'base':'Base','revis':'REVIS 자체 구현','rudder_clone':'RUDDER 원본 함수 연결'}
        md=['# Qwen BF16 세 방법 · 완료된 코호트 중간 보고','','Base·REVIS·RUDDER의 POPE adversarial 각3,000문항과 CHAIR 각500장, 총10,500응답을 공식 채점했다. 모든 평가 응답은 GPU 0에서 BF16으로 생성했다. 수치는 %.','',
            '| 방법 | POPE Acc ↑ | POPE F1 ↑ | CHAIR_s ↓ | CHAIR_i ↓ | 객체 recall ↑ |','|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            p,c=metrics['pope'][method],metrics['chair'][method]
            md.append(f"| {labels[method]} | {100*p['accuracy']:.2f} | {100*p['f1']:.2f} | {100*c['CHAIR_s']:.2f} | {100*c['CHAIR_i']:.2f} | {100*c['object_recall']:.2f} |")
        md.extend(['','GrAInS와 원저자 VTI는 이 표에 포함하지 않았다. 과거 VTI 자체 구현 결과도 제외했다. 다섯 방법 전체17,500응답의 평가·검증 목표는 아직 진행 중이다.',
            '', '공식 POPE evaluate.py와 원본 CHAIR Python3 포트로 채점하고 이미지 단위 paired bootstrap 10,000회를 수행했다. 전체 문항·이미지 ID, 응답 hash, 고정 입력, 모델 revision, GPU0, BF16, 생성 길이, REVIS artifact와 RUDDER 원본 호출을 검증했다. 상세 수치와95% 구간은 summary.json에 있다.'])
        (output/'REPORT_KO.md').write_text('\n'.join(md)+'\n')
        (output/'scorer-source.py').write_bytes(Path(__file__).read_bytes())
        save(output/'summary.json',summary)
        update(stage='interim_scored',summary_sha256=sha(output/'summary.json'),completed_at=time.time(),selected_responses=10500)
        print(json.dumps({'status':summary['status'],'output':str(output),'summary_sha256':sha(output/'summary.json'),'metrics':metrics}))
    except BaseException:update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--vti',type=Path,default=VTI)
    p.add_argument('--output',type=Path);p.add_argument('--poll-seconds',type=int,default=20)
    args=p.parse_args();assert 1<=args.poll_seconds<=60
    main(args.parent,args.vti,args.output or args.parent/'bf16-three-complete',args.poll_seconds)
