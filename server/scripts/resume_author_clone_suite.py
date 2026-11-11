"""Recover metadata-only worker startup failure without restarting live extraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from author_clone_bridge import ROOT
from run_author_clone_suite import GPU, alive, save


def main(output):
    original_jobs=json.loads((output/'jobs.json').read_text())
    extraction=next(j for j in original_jobs if j['extract'])
    previous_status=json.loads((output/'status.json').read_text())
    assert previous_status['stage']=='failed' and not alive(previous_status['controller_pid'])
    assert alive(extraction['pid']) or json.loads((output/'extraction-status.json').read_text())['stage']=='complete'
    for job in original_jobs:
        if job['extract']:continue
        assert not alive(job['pid'])
        state=json.loads((output/f"worker-{job['worker']}.json").read_text())
        assert state['stage']=='failed' and state['completed_records']==0
    assert not list((output/'predictions').glob('*.jsonl')) and not list((output/'claims').iterdir())
    save(output/'initial-startup-jobs.json',original_jobs)
    save(output/'initial-startup-failure.json',previous_status)
    files=['scripts/author_clone_worker_entry.py','scripts/resume_author_clone_suite.py']
    save(output/'execution-amendment.json',{'reason':'HF model.config lost revision metadata; validated pinned cache snapshot and restored metadata only',
        'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        'original_protocol_unchanged':True,'original_author_functions_unchanged':True,'live_extraction_preserved':extraction['pid']})
    jobs=[extraction];save(output/'jobs.json',jobs);started=time.time()
    def status(stage,**kw):save(output/'status.json',{'stage':stage,'controller_pid':os.getpid(),'started_at':started,'updated_at':time.time(),**kw})
    def launch(worker,dtype):
        log=output/f'{worker}.log'
        with log.open('xb') as sink:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'scripts/author_clone_worker_entry.py','--output',str(output),'--worker',worker,'--dtype',dtype],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU),stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
        job={'worker':worker,'pid':p.pid,'dtype':dtype,'extract':False,'log':str(log)};jobs.append(job);save(output/'jobs.json',jobs);return job
    try:
        bf16=[launch(f'bf16-recovered-{i}','bfloat16') for i in range(3)];fp32=[];extraction_complete=False
        while True:
            states={}
            for job in jobs:
                f=output/('extraction-status.json' if job['extract'] else f"worker-{job['worker']}.json")
                s=json.loads(f.read_text()) if f.exists() else {'stage':'starting'}
                active=alive(job['pid']);states[job['worker']]={**s,'actual_live':active}
                if s['stage']=='failed' or (not active and s['stage']!='complete'):raise RuntimeError(json.dumps({'job':job,'state':s}))
            if not extraction_complete and not alive(extraction['pid']):
                assert states['grains-extraction']['stage']=='complete';extraction_complete=True;fp32.append(launch('fp32-0','float32'))
            if extraction_complete and len(fp32)==1 and sum(alive(j['pid']) for j in bf16)<=1 and alive(fp32[0]['pid']):fp32.append(launch('fp32-1','float32'))
            status('generating' if extraction_complete else 'extracting_and_generating',workers=states,
                   completed_records=sum(s.get('completed_records',0) for k,s in states.items() if k!='grains-extraction'),
                   total_records=21000,extracted_fit_records=states['grains-extraction'].get('completed',0))
            if extraction_complete and all(not alive(j['pid']) for j in jobs):break
            time.sleep(15)
        status('auditing')
        with (output/'audit.log').open('ab') as sink:
            subprocess.run([str(ROOT/'.venv/bin/python'),'scripts/audit_author_clone_suite.py','--output',str(output)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=sink,stderr=subprocess.STDOUT,check=True)
        assert json.loads((output/'report.json').read_text())['status']=='verified_complete';status('complete',completed_at=time.time())
    except BaseException:
        status('failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
