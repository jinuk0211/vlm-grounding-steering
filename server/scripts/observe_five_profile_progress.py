"""Read-only bounded observation of exact live Gemma/Qwen task instances."""
import argparse
import collections
import json
import os
from pathlib import Path
import time

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
ALLOWED={'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde','GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'}

def read(path):return json.loads(Path(path).read_text())

def instance(job,script=None,output=None,cpu=False):
    pid=job.get('pid',job.get('controller_pid'));expected=job.get('start_ticks',job.get('controller_start_ticks'))
    assert pid is not None and expected is not None
    p=Path('/proc')/str(pid);proof={'pid':pid,'start_ticks':expected,'live':False}
    try:
        fields=(p/'stat').read_text().split(') ',1)[1].split()
        if fields[0] in ['Z','X'] or int(fields[19])!=expected:return proof
        assert p.stat().st_uid==os.getuid() and os.readlink(p/'cwd')==str(ROOT)
        args=(p/'cmdline').read_bytes().decode().split('\0')
        if script:assert script in args
        if output:assert str(output) in args
        env=(p/'environ').read_bytes().split(b'\0')
        visible=next((v.decode().split('=',1)[1] for v in env if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
        if cpu:assert visible==''
        elif job.get('visible_devices'):
            assert visible==job['visible_devices'] and set(visible.split(','))<=ALLOWED
        proof.update(live=True,state=fields[0],visible_devices=visible)
    except FileNotFoundError:pass
    return proof

def snapshot(count_rows=False):
    result={'checked_at':time.time()}
    for model,file in [('gemma','five-profile-gemma-launch.json'),('qwen','qwen-five-profile-launch-20261029.json')]:
        receipt=read(ROOT/'outputs'/file);out=Path(receipt['output']);status=read(out/'status.json')
        controller=instance(receipt,receipt['script'],out,True)
        assert status['stage']!='failed',{'model':model,'status':status}
        assert controller['live'] or status['stage']=='complete',{'model':model,'controller_missing':receipt,'stage':status['stage']}
        entry={'output':str(out),'stage':status['stage'],'controller':controller,'workers':[],
            'base_fp32_enabled':status.get('base_fp32_enabled'),'grains_inference_dtype':status.get('grains_inference_dtype')}
        jobs=out/'jobs.json'
        if jobs.exists():
            for job in read(jobs):
                if job.get('preflight'):continue
                child_out=Path(job.get('output',out))
                child=instance(job,job.get('script'),child_out)
                file=child_out/job.get('state_file','worker-'+job.get('worker','')+'.json')
                ws=read(file) if file.exists() else {}
                assert ws.get('stage')!='failed' and ws.get('status')!='failed',{'model':model,'job':job,'state':ws}
                entry['workers'].append({**child,'kind':job.get('kind','generation'),'worker':job.get('worker'),
                    'stage':ws.get('stage',ws.get('status')),'completed':ws.get('completed_records',ws.get('completed')),
                    'benchmark':ws.get('benchmark'),'last_seconds':ws.get('last_seconds')})
        if count_rows:
            counts=collections.Counter();tasks=out/'tasks.json'
            if tasks.exists():
                for task in read(tasks):
                    file=out/'predictions'/(task['id']+'.jsonl')
                    if file.exists():
                        raw=file.read_bytes();raw=raw[:raw.rfind(b'\n')+1]
                        counts[task['benchmark']+':'+task['method']]+=raw.count(b'\n')
            entry['counts']=dict(counts)
        result[model]=entry
    return result

def main(seconds):
    assert 0<=seconds<=45 and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    before=snapshot(True);deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        time.sleep(min(5,max(0,deadline-time.monotonic())));current=snapshot()
        if any(current[m]['stage']!=before[m]['stage'] for m in ['gemma','qwen']):break
    after=snapshot(True);after['observed_seconds']=seconds
    target=ROOT/'outputs/five-profile-observation-latest.json';tmp=target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(after,indent=2)+'\n');tmp.replace(target)
    print(json.dumps(after))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=45);main(p.parse_args().seconds)
