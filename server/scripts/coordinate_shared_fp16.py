"""Adopt the live GPU3 worker and add a user-authorized limited GPU2 replica."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from resume_gpu23_suite_20261029 import ticks,same_instance
from grains_fp16_worker_v2 import ROOT,GPU2,GPU3,save,verified_protocol

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'outputs/.five-profile-gemma-controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=verified_protocol(output);previous=json.loads((output/'status.json').read_text())
    assert previous['stage']=='generating_grains_fp16'
    assert not same_instance({'pid':previous['controller_pid'],'start_ticks':previous['controller_start_ticks'],'output':str(output)})
    jobs=json.loads((output/'jobs.json').read_text())
    existing=[j for j in jobs if not j['preflight']]
    assert len(existing)==1 and existing[0]['visible_devices']==GPU3
    assert same_instance(dict(existing[0],output=str(output)))
    assert json.loads((output/('worker-'+existing[0]['worker']+'.json')).read_text())['stage']!='failed'
    row=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert row[0].strip()==GPU2 and int(row[1])>=19*1024
    assert not (output/'shared-gpu2-amendment.json').exists()
    sources=['scripts/grains_fp16_limited_entry.py','scripts/coordinate_shared_fp16.py']
    amendment={'request':'2026-10-29 user: GPU2 has spare VRAM, add our model',
        'allocator_fraction':0.70,'source_hashes':{f:sha(ROOT/f) for f in sources},
        'protocol_sha256':protocol['sha256'],'original_protocol_and_generation_unchanged':True,
        'adopted_gpu3_worker':existing[0],'prior_controller_pid':previous['controller_pid'],
        'prior_status_sha256':hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest(),
        'other_gpu2_processes_not_modified':True,'created_at':time.time()}
    save(output/'shared-gpu2-amendment.json',amendment);save(output/'status-before-shared-gpu2.json',previous)
    save(output/'jobs-before-shared-gpu2.json',jobs)
    name='gpu2-shared-fp16-'+str(int(time.time()));log=output/(name+'.log')
    with log.open('xb') as stream:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'scripts/grains_fp16_limited_entry.py','--output',str(output),'--worker',name],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU2),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    job={'pid':child.pid,'start_ticks':ticks(child.pid),'worker':name,'visible_devices':GPU2,'dtype':'float16',
        'preflight':False,'log':str(log),'allocator_fraction':0.70,'entry_source_sha256':sha(ROOT/'scripts/grains_fp16_limited_entry.py')}
    jobs.append(job);save(output/'jobs.json',jobs)
    state={**previous,'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'shared_gpu2':True,
        'allocator_fraction_gpu2':0.70,'base_fp32_enabled':False,'grains_inference_dtype':'float16'}
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'status.json',state)
    save(ROOT/'outputs/five-profile-gemma-current.json',{'output':str(output),'controller_pid':os.getpid(),'controller_start_ticks':state['controller_start_ticks']})
    receipt={'pid':os.getpid(),'start_ticks':state['controller_start_ticks'],'output':str(output),'script':'scripts/coordinate_shared_fp16.py',
        'log':str(output/'shared-gpu2-controller.log'),'base_fp32_enabled':False,'grains_inference_dtype':'float16'}
    save(ROOT/'outputs/five-profile-gemma-launch.json',receipt)
    try:
        while True:
            progress=[];done=True
            for j in jobs:
                if j['preflight']:continue
                active=same_instance(dict(j,output=str(output)));done &= not active
                f=output/('worker-'+j['worker']+'.json');ws=json.loads(f.read_text()) if f.exists() else {}
                assert ws.get('stage')!='failed',{'job':j,'state':ws}
                if not active:assert ws.get('stage')=='complete',{'job':j,'state':ws}
                progress.append({'pid':j['pid'],'worker':j['worker'],'actual_live':active,'state':ws})
            update(stage='generating_grains_fp16',workers=progress)
            if done:break
            time.sleep(10)
        assert child.wait()==0
        for f,digest in amendment['source_hashes'].items():assert sha(ROOT/f)==digest
        update(stage='auditing')
        with (output/'audit-shared-gpu2.log').open('xb') as stream:
            subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/audit_five_profile_suite_v2.py','--output',str(output)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,check=True)
        report=json.loads((output/'report.json').read_text());assert report['status']=='verified_complete'
        resource=json.loads((output/(name+'-resource-limit.json')).read_text())
        assert resource['allocator_fraction']==0.70 and resource['generation_and_author_math_unchanged']
        report['shared_gpu2_amendment']={'sha256':sha(output/'shared-gpu2-amendment.json'),'allocator_fraction':0.70,
            'generation_and_original_author_math_unchanged':True,'other_users_processes_untouched':True}
        save(output/'report.json',report)
        update(stage='complete',report_sha256=sha(output/'report.json'),completed_at=time.time())
    except BaseException:
        # Keep useful GPU3 progress. The goal controller handles any actual failed worker separately.
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
