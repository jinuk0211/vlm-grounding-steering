"""Real preflight and one FP16 worker alongside four BF16 GPU0 replicas."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save
from qwen_fp16_stable_stage import verify_attention_amendment
from run_qwen_five_profile_suite import ticks

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
    lock=(out/'fp16-stable-controller.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_attention_amendment(out)
    assert not (out/'fp16-stable-jobs.json').exists()
    import shutil
    amendment={'controller_source_sha256':sha(__file__),'attention_amendment_sha256':sha(out/'fp16-attention-amendment.json'),
        'replicas':1,'gpu_indices':[0],'created_at':time.time()}
    save(out/'fp16-stable-controller-amendment.json',amendment);shutil.copy2(__file__,out/'fp16-stable-controller-source.py')
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'gpu_indices':[0]}
    jobs=[];children=[]
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'fp16-stable-status.json',state)
    def free():
        while True:
            mib=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if mib>=22*1024:return
            update(stage='waiting_gpu0_free_memory',free_mib=mib,required_mib=22*1024);time.sleep(15)
    def launch(stage,name,state_file,proof,extra):
        free();script='scripts/qwen_fp16_stable_stage.py';log=out/(name+'.log')
        args=[str(ROOT/'.venv/bin/python'),'-u',script,'--output',str(out),'--stage',stage]+extra
        with log.open('xb') as f:
            child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        job={'pid':child.pid,'start_ticks':ticks(child.pid),'script':script,'output':str(out),
            'kind':'generation' if stage=='generation' else 'preflight_fp16','visible_devices':GPU0,'log':str(log),
            'state_file':state_file,'device_proof':proof,'dtype':'float16','gpu0_migration':True,
            'worker':name if stage=='generation' else None,'fp16_stable_attention':True}
        children.append(child);jobs.append(job);save(out/'fp16-stable-jobs.json',jobs)
        while True:
            code=child.poll();path=out/state_file;s=json.loads(path.read_text()) if path.exists() else {}
            if s.get('pid')!=job['pid']:s={}
            assert code in (None,0) and s.get('stage')!='failed',{'job':job,'code':code,'state':s}
            update(stage='preflight_fp16' if stage!='generation' else 'generating_fp16',job_pid=job['pid'],
                completed=s.get('completed_records'),task=s.get('task'))
            if code is not None:
                assert s.get('stage')=='complete'
                job.update(exit_code=code,state_sha256=sha(path),device_proof_sha256=sha(out/proof))
                save(out/'fp16-stable-jobs.json',jobs);return
            time.sleep(5)
    try:
        launch('preflight_fp16','fp16-stable-preflight','fp16-stable-preflight.json','fp16-stable-preflight-device-map.json',[])
        stamp=str(int(time.time()));name='gpu0-stable-fp16-'+stamp
        launch('generation',name,'worker-'+name+'.json',name+'-device-map.json',
            ['--worker',name,'--dtype','float16','--lease-epoch','gpu0-stable-fp16-'+stamp])
        update(stage='complete',completed_at=time.time())
    except BaseException:
        for child in children:
            if child.poll() is None:child.terminate()
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
