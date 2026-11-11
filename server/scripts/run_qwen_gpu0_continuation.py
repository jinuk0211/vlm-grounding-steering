"""Adopt the running extraction; continue GPU0 stages with PID-bound status."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_gpu0_stage_v2 import verify_repair
from run_qwen_five_profile_suite import ticks
from audit_qwen_five_profile_suite import actual_job_live

SOURCES=['scripts/run_qwen_gpu0_continuation.py','scripts/audit_qwen_gpu0_continuation.py']

def run(output):
    assert output.resolve().parent==ROOT/'outputs' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'outputs/.qwen-author-suite-controller.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=verify_protocol(output);verify_repair(output)
    amendment=json.loads((output/'gpu0-hardware-amendment.json').read_text())
    migration=Path(amendment['migration'])
    jobs=json.loads((output/'jobs.json').read_text())
    adopted=json.loads((ROOT/'outputs/qwen-gpu0-direct-extraction.json').read_text())
    direct=next(j for j in jobs if (j['pid'],j['start_ticks'])==(adopted['pid'],adopted['start_ticks']))
    assert direct['direct_launch'] and direct['visible_devices']==GPU0
    failures=[j for j in jobs if j.get('gpu0_pre_model_failure')]
    assert len(failures)==2
    for j in failures:
        assert not actual_job_live(j,output) and sha(output/j['state_file'])==j['failed_state_sha256']
    manifest={'source_hashes':{p:sha(ROOT/p) for p in SOURCES},'repair_amendment_sha256':sha(output/'gpu0-repair-amendment.json'),
        'failed_pre_model_jobs':failures,'adopted_extraction_identity':{'pid':direct['pid'],'start_ticks':direct['start_ticks']},
        'request_gpu_indices':[0],'status_pid_must_match':True,'created_at':time.time()}
    assert not (output/'gpu0-continuation-amendment.json').exists()
    save(output/'gpu0-continuation-amendment.json',manifest)
    for p in SOURCES:
        dest=migration/'continuation-source-snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'started_at':time.time(),
        'gpu_indices':[0],'base_fp32_enabled':False,'grains_inference_dtype':'float16'}
    children=[]
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'status.json',state)
    def status(job):
        path=output/job['state_file'];s=json.loads(path.read_text()) if path.exists() else {}
        return s if s.get('pid')==job['pid'] else {}
    def free(gib,phase):
        while True:
            values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
            assert values[0].strip()==GPU0
            if int(values[1])>=gib*1024:return
            update(stage='waiting_for_gpu0_memory',next_phase=phase,free_mib=int(values[1]));time.sleep(15)
    def launch(stage,label,kind,dtype,state_file,device_proof,extra=None):
        script='scripts/qwen_gpu0_stage_v2.py';log=output/(label+'.log')
        args=[str(ROOT/'.venv/bin/python'),script,'--output',str(output),'--stage',stage]+(extra or [])
        with log.open('xb') as stream:
            child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        j={'pid':child.pid,'start_ticks':ticks(child.pid),'script':script,'output':str(output),'kind':kind,
            'visible_devices':GPU0,'log':str(log),'state_file':state_file,'device_proof':device_proof,'dtype':dtype,
            'gpu0_migration':True,'worker':label if kind=='generation' else None}
        jobs.append(j);children.append(child);save(output/'jobs.json',jobs);return child,j
    def wait(group,phase):
        while True:
            done=True;progress=[]
            for child,j in group:
                code=child.poll();s=status(j);done &= code is not None
                assert code in (None,0) and s.get('stage')!='failed' and s.get('status')!='failed',{'job':j,'code':code,'state':s}
                if code==0:
                    assert s.get('stage')=='complete',s
                    j.update(exit_code=0,state_sha256=sha(output/j['state_file']),device_proof_sha256=sha(output/j['device_proof']))
                progress.append({'pid':j['pid'],'stage':s.get('stage'),'completed':s.get('completed_records',s.get('completed'))})
            save(output/'jobs.json',jobs);update(stage=phase,jobs=progress)
            if done:return
            time.sleep(10)
    receipt={'pid':os.getpid(),'start_ticks':state['controller_start_ticks'],'script':'scripts/run_qwen_gpu0_continuation.py',
        'output':str(output),'log':str(output/'gpu0-continuation-controller.log'),'gpu_indices':[0],
        'base_fp32_enabled':False,'grains_inference_dtype':'float16'}
    save(ROOT/'outputs/qwen-five-profile-launch-20261029.json',receipt);save(ROOT/'outputs/qwen-author-suite-current.json',receipt)
    try:
        while actual_job_live(direct,output):
            s=status(direct);assert s.get('stage')!='failed',s
            update(stage='fitting_grains_gpu0_fp32',adopted_extraction_pid=direct['pid'],completed=s.get('completed'),total=50)
            time.sleep(5)
        s=status(direct);assert s.get('stage')=='complete' and s.get('completed')==50,s
        assert len((output/'extraction.jsonl').read_text().splitlines())==50
        direct.update(adopted_completion_observed=True,exit_code_observed=False,state_sha256=sha(output/direct['state_file']),
            device_proof_sha256=sha(output/direct['device_proof']),observed_exited_at=time.time())
        save(output/'jobs.json',jobs)
        methods={}
        for method,path in [('revis',output/'artifacts/revis.pt'),('vti',output/'artifacts/vti.pt'),('grains_clone',output/'grains-author.pt')]:
            sidecar=json.loads((output/(('grains' if method=='grains_clone' else method)+'-artifact.json')).read_text())
            assert sidecar['status']=='verified_complete' and sidecar['source_protocol_sha256']==protocol['sha256']
            assert sha(path)==sidecar['artifact_sha256'];methods[method]={'sha256':sha(path)}
        save(output/'artifacts-ready.json',{'status':'verified_complete','source_protocol_sha256':protocol['sha256'],
            'methods':methods,'gemma_vectors_reused':False,'hardware_amendment_sha256':sha(output/'gpu0-hardware-amendment.json')})
        free(25,'fp16_preflight')
        wait([launch('preflight_fp16','gpu0-continuation-fp16-preflight','preflight_fp16','float16',
            'fp16-preflight.json','fp16-preflight-device-map.json')],'preflight_gpu0_fp16')
        tag=str(int(time.time()))
        for dtype in ['bfloat16','float16']:
            free(45,'generate_'+dtype);group=[]
            for i in range(2):
                name='gpu0-'+dtype+'-'+str(i)+'-'+tag
                group.append(launch('generation',name,'generation',dtype,'worker-'+name+'.json',name+'-device-map.json',
                    ['--worker',name,'--dtype',dtype,'--lease-epoch','gpu0-'+dtype+'-'+tag]))
            wait(group,'generating_gpu0_'+dtype)
        update(stage='auditing')
        with (output/'audit-gpu0-continuation.log').open('xb') as stream:
            subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/audit_qwen_gpu0_continuation.py','--output',str(output)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/'report.json').read_text());assert report['status']=='verified_complete' and report['audit']['all_17500_responses_verified']
        update(stage='complete',report_sha256=sha(output/'report.json'),completed_at=time.time())
    except BaseException:
        for child in children:
            if child.poll() is None:child.terminate()
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
