"""Run independent BF16 profiles while retaining the failed FP16 preflight."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_gpu0_stage_v2 import verify_repair
from run_qwen_five_profile_suite import ticks

def run(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    lock=(ROOT/'outputs/.qwen-author-suite-controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_protocol(output);verify_repair(output)
    prior=json.loads((output/'status.json').read_text());assert prior['stage']=='failed'
    failure=json.loads((output/'fp16-preflight.json').read_text());assert failure['stage']=='failed'
    jobs=json.loads((output/'jobs.json').read_text())
    assert not any(j['kind']=='generation' for j in jobs)
    stamp=str(int(time.time()));archive=output/('fp16-native-preflight-failure-'+stamp);archive.mkdir()
    import shutil
    for name in ['status.json','jobs.json','fp16-preflight.json','fp16-preflight-device-map.json']:
        if (output/name).exists():shutil.copy2(output/name,archive/name)
    for j in jobs:
        if j['pid']==failure['pid']:
            j['numerical_preflight_failed']=True;j['state_file']=str((archive/'fp16-preflight.json').relative_to(output))
            j['failed_state_sha256']=sha(archive/'fp16-preflight.json')
    save(output/'jobs.json',jobs)
    save(output/'gpu0-bf16-priority-amendment.json',{'source_sha256':sha(__file__),'source_file':'scripts/run_qwen_gpu0_bf16_controller.py',
        'reason':'Native Qwen FP16 activation nonfinite before steering; run independent BF16 methods while investigating',
        'failed_preflight_state_sha256':sha(archive/'fp16-preflight.json'),'archive':str(archive),'gpu_indices':[0],
        'dtype_unchanged':'bfloat16','grains_still_requires_fp16_preflight':True,'created_at':time.time()})
    shutil.copy2(__file__,archive/Path(__file__).name)
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'gpu_indices':[0],
        'base_fp32_enabled':False,'grains_inference_dtype':'float16','fp16_preflight_pending_numerical_resolution':True}
    def update(**kw):state.update(kw,updated_at=time.time());save(output/'status.json',state)
    receipt={'pid':os.getpid(),'start_ticks':state['controller_start_ticks'],'script':'scripts/run_qwen_gpu0_bf16_controller.py',
        'output':str(output),'gpu_indices':[0],'log':str(output/'gpu0-bf16-priority-controller.log')}
    save(ROOT/'outputs/qwen-five-profile-launch-20261029.json',receipt);save(ROOT/'outputs/qwen-author-suite-current.json',receipt)
    children=[]
    try:
        for i in range(2):
            name='gpu0-bfloat16-priority-'+str(i)+'-'+stamp;script='scripts/qwen_gpu0_stage_v2.py';log=output/(name+'.log')
            args=[str(ROOT/'.venv/bin/python'),'-u','-c',"import torch; torch.cuda.init(); import sys,runpy; sys.path.insert(0,'scripts'); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",script,
                '--output',str(output),'--stage','generation','--worker',name,'--dtype','bfloat16','--lease-epoch','gpu0-bf16-priority-'+stamp]
            with log.open('xb') as stream:
                p=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            j={'pid':p.pid,'start_ticks':ticks(p.pid),'script':script,'output':str(output),'kind':'generation','visible_devices':GPU0,
                'log':str(log),'state_file':'worker-'+name+'.json','device_proof':name+'-device-map.json','dtype':'bfloat16',
                'gpu0_migration':True,'worker':name}
            children.append((p,j));jobs.append(j);save(output/'jobs.json',jobs)
        while True:
            done=True;progress=[]
            for p,j in children:
                code=p.poll();done &= code is not None;path=output/j['state_file']
                s=json.loads(path.read_text()) if path.exists() else {}
                if s.get('pid')!=j['pid']:s={}
                assert code in (None,0) and s.get('stage')!='failed',{'job':j,'code':code,'state':s}
                if code==0:
                    assert s.get('stage')=='complete'
                    j.update(exit_code=0,state_sha256=sha(path),device_proof_sha256=sha(output/j['device_proof']))
                progress.append({'pid':j['pid'],'stage':s.get('stage'),'completed':s.get('completed_records'),'task':s.get('task')})
            save(output/'jobs.json',jobs);update(stage='generating_gpu0_bfloat16',jobs=progress)
            if done:break
            time.sleep(10)
        update(stage='bf16_generation_complete_fp16_pending',completed_at=time.time())
    except BaseException:
        for p,j in children:
            if p.poll() is None:p.terminate()
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
