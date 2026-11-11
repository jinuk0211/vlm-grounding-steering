"""Continue preserved Qwen fitting, then all five profiles on physical GPU0."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from qwen_gpu0_migration_api import ROOT,GPU0,verify_protocol,sha,save
from qwen_five_profile_api import verify_run_protocol as verify_parent
from run_qwen_five_profile_suite import ticks

SOURCES=['scripts/qwen_gpu0_migration_api.py','scripts/qwen_gpu0_stage_entry.py',
         'scripts/run_qwen_gpu0_suite.py','scripts/audit_qwen_gpu0_suite.py']

def gpu0_free(required_gib):
    row=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==GPU0
    return int(row[1])>=required_gib*1024,int(row[1])

def run(output):
    assert output.resolve().parent==ROOT/'outputs' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'outputs/.qwen-author-suite-controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=verify_parent(output)
    migration_record=json.loads((ROOT/'outputs/qwen-gpu0-migration-current.json').read_text())
    assert migration_record['output']==str(output);migration=Path(migration_record['migration'])
    stop_path=migration/'stop-proof.json';stop=json.loads(stop_path.read_text());assert stop['all_our_selected_instances_exited']
    journal=(output/'extraction.jsonl').read_bytes()
    assert len(journal)==stop['journal_bytes']==stop['committed_journal_bytes']
    assert sha(output/'extraction.jsonl')==stop['journal_prefix_sha256']
    assert not list((output/'predictions').glob('*.jsonl')),'Benchmark rows already exist; inspect before moving'
    assert not (output/'gpu0-hardware-amendment.json').exists()
    amendment={'parent_protocol_sha256':parent['sha256'],'user_requested_gpu_indices':[0],'gpu_uuid':GPU0,
        'request':'User: stop GPU2/3 and run all remaining work on GPU0',
        'source_hashes':{name:sha(ROOT/name) for name in SOURCES},'stop_proof':str(stop_path),'stop_proof_sha256':sha(stop_path),
        'migration':str(migration),'preserved_ig_records':stop['completed_ig_records'],
        'remaining_ig_records':50-stop['completed_ig_records'],'replicas_per_generation_phase':2,
        'model_revision_dtype_original_author_math_and_generation_unchanged':True,
        'original_frozen_sources_and_protocol_preserved':True,'created_at':time.time()}
    save(output/'gpu0-hardware-amendment.json',amendment)
    for name in SOURCES:
        path=migration/'source-snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    jobs=json.loads((output/'jobs.json').read_text());stopped={(j['pid'],j['start_ticks']) for j in stop['stopped_instances']}
    for job in jobs:
        if (job['pid'],job['start_ticks']) in stopped:
            job['gpu0_migration_stopped']=True;job['stop_proof_sha256']=sha(stop_path)
    save(output/'jobs.json',jobs)
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'started_at':time.time(),
        'gpu_indices':[0],'parent_protocol_gpu_indices':parent['gpu_indices'],'hardware_amendment_sha256':sha(output/'gpu0-hardware-amendment.json'),
        'base_fp32_enabled':False,'grains_inference_dtype':'float16','qwen_gpu_models_launched':True}
    children=[]
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'status.json',state)
    def free(gib,phase):
        while True:
            available,mib=gpu0_free(gib)
            if available:return
            update(stage='waiting_for_gpu0_memory',next_phase=phase,required_free_gib=gib,free_mib=mib);time.sleep(15)
    def launch(stage,label,kind,dtype,state_file,device_proof,extra=None):
        script='scripts/qwen_gpu0_stage_entry.py';log=output/(label+'.log')
        args=[str(ROOT/'.venv/bin/python'),script,'--output',str(output),'--stage',stage]
        if extra:args+=extra
        with log.open('xb') as stream:
            child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        job={'pid':child.pid,'start_ticks':ticks(child.pid),'script':script,'output':str(output),'kind':kind,
            'visible_devices':GPU0,'log':str(log),'state_file':state_file,'device_proof':device_proof,
            'dtype':dtype,'gpu0_migration':True,'worker':label if kind=='generation' else None}
        jobs.append(job);children.append(child);save(output/'jobs.json',jobs);return child,job
    def wait(group,phase):
        while True:
            complete=True;progress=[]
            for child,job in group:
                code=child.poll();complete &= code is not None
                file=output/job['state_file'];s=json.loads(file.read_text()) if file.exists() else {}
                assert code in [None,0] and s.get('stage')!='failed' and s.get('status')!='failed',{'job':job,'code':code,'state':s}
                if code==0:
                    assert s.get('stage',s.get('status')) in ['complete','passed'],s
                    job.update(exit_code=0,state_sha256=sha(file),device_proof_sha256=sha(output/job['device_proof']))
                progress.append({'pid':job['pid'],'kind':job['kind'],'stage':s.get('stage',s.get('status')),
                    'completed':s.get('completed_records',s.get('completed')),'returncode':code})
            save(output/'jobs.json',jobs);update(stage=phase,jobs=progress)
            if complete:return
            time.sleep(10)
    old_launch=ROOT/'outputs/qwen-five-profile-launch-20261029.json'
    shutil.copy2(old_launch,migration/'previous-launch.json')
    receipt={'pid':os.getpid(),'start_ticks':state['controller_start_ticks'],'script':'scripts/run_qwen_gpu0_suite.py','output':str(output),
        'log':str(output/'gpu0-controller.log'),'base_fp32_enabled':False,'grains_inference_dtype':'float16','gpu_indices':[0]}
    save(old_launch,receipt);save(ROOT/'outputs/qwen-author-suite-current.json',receipt)
    try:
        verify_protocol(output);update(stage='starting_gpu0_migration')
        free(50,'resume_grains_ig');job=launch('extract','gpu0-grains-extraction','extraction','float32',
            'extraction-gpu0-status.json','grains-extraction-gpu0-device-map.json');wait([job],'fitting_grains_gpu0_fp32')
        assert not (output/'artifacts-ready.json').exists()
        methods={}
        for method,path in [('revis',output/'artifacts/revis.pt'),('vti',output/'artifacts/vti.pt'),('grains_clone',output/'grains-author.pt')]:
            sidecar=json.loads((output/(('grains' if method=='grains_clone' else method)+'-artifact.json')).read_text())
            assert sidecar['status']=='verified_complete' and sidecar['source_protocol_sha256']==parent['sha256']
            assert sha(path)==sidecar['artifact_sha256'];methods[method]={'sha256':sha(path)}
        save(output/'artifacts-ready.json',{'status':'verified_complete','source_protocol_sha256':parent['sha256'],
            'methods':methods,'gemma_vectors_reused':False,'hardware_amendment_sha256':sha(output/'gpu0-hardware-amendment.json')})
        free(30,'fp16_preflight');job=launch('preflight_fp16','gpu0-fp16-preflight','preflight_fp16','float16',
            'fp16-preflight.json','fp16-preflight-device-map.json');wait([job],'preflight_gpu0_fp16')
        tag=str(int(time.time()))
        for dtype in ['bfloat16','float16']:
            free(45,'generate_'+dtype);group=[]
            for index in range(2):
                name='gpu0-'+dtype+'-'+str(index)+'-'+tag
                group.append(launch('generation',name,'generation',dtype,'worker-'+name+'.json',name+'-device-map.json',
                    ['--worker',name,'--dtype',dtype,'--lease-epoch','gpu0-'+dtype+'-'+tag]))
            wait(group,'generating_gpu0_'+dtype)
        update(stage='auditing')
        with (output/'audit-gpu0.log').open('xb') as stream:
            subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/audit_qwen_gpu0_suite.py','--output',str(output)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/'report.json').read_text());assert report['status']=='verified_complete' and report['audit']['all_17500_responses_verified']
        update(stage='complete',report_sha256=sha(output/'report.json'),completed_at=time.time())
    except BaseException:
        for child in children:
            if child.poll() is None:child.terminate()
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
