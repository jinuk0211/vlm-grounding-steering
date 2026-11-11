"""Add two GPU0 replicas using the SAME atomic task lease namespace."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_gpu0_stage_v2 import verify_repair
from run_qwen_five_profile_suite import ticks

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
    lock=(out/'extra-replicas.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_protocol(out);verify_repair(out)
    original=json.loads((out/'jobs.json').read_text())
    existing=[j for j in original if j['kind']=='generation' and j['dtype']=='bfloat16'];assert len(existing)==2
    epochs=set()
    for j in existing:
        p=Path('/proc')/str(j['pid']);assert ticks(j['pid'])==j['start_ticks'] and p.stat().st_uid==os.getuid()
        args=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
        assert args[args.index('--output')+1]==str(out)
        epochs.add(args[args.index('--lease-epoch')+1])
    assert len(epochs)==1;epoch=epochs.pop()
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=45*1024,free
    assert not (out/'supplemental-jobs.json').exists()
    amendment={'user_request':'Use remaining GPU0 VRAM to speed up work','gpu_indices':[0],
        'bf16_replicas_before':2,'bf16_replicas_after':4,'same_lease_epoch':epoch,'batch_size':1,
        'source_sha256':sha(__file__),'created_at':time.time(),'initial_free_mib':free,
        'original_protocol_math_precision_data_unchanged':True,'separate_job_file_to_avoid_controller_write_races':True}
    save(out/'gpu0-extra-replicas-amendment.json',amendment)
    import shutil
    shutil.copy2(__file__,out/'gpu0-extra-replicas-source.py')
    children=[];jobs=[];stamp=str(int(time.time()))
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'gpu_indices':[0]}
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'supplemental-status.json',state)
    try:
        for i in range(2):
            name='gpu0-extra-bf16-'+str(i)+'-'+stamp;script='scripts/qwen_gpu0_stage_v2.py';log=out/(name+'.log')
            args=[str(ROOT/'.venv/bin/python'),'-u',script,'--output',str(out),'--stage','generation','--worker',name,
                '--dtype','bfloat16','--lease-epoch',epoch]
            with log.open('xb') as f:
                child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            j={'pid':child.pid,'start_ticks':ticks(child.pid),'script':script,'output':str(out),'kind':'generation',
                'visible_devices':GPU0,'log':str(log),'state_file':'worker-'+name+'.json','device_proof':name+'-device-map.json',
                'dtype':'bfloat16','gpu0_migration':True,'worker':name,'supplemental_replica':True}
            children.append((child,j));jobs.append(j);save(out/'supplemental-jobs.json',jobs)
        while True:
            done=True;progress=[]
            for child,j in children:
                code=child.poll();done &= code is not None;path=out/j['state_file']
                s=json.loads(path.read_text()) if path.exists() else {}
                if s.get('pid')!=j['pid']:s={}
                assert code in (None,0) and s.get('stage')!='failed',{'job':j,'code':code,'state':s}
                if code==0:
                    assert s['stage']=='complete'
                    j.update(exit_code=0,state_sha256=sha(path),device_proof_sha256=sha(out/j['device_proof']))
                progress.append({'pid':j['pid'],'stage':s.get('stage'),'completed':s.get('completed_records'),'task':s.get('task')})
            save(out/'supplemental-jobs.json',jobs);update(stage='generating_gpu0_bf16_supplemental',jobs=progress)
            if done:break
            time.sleep(10)
        update(stage='complete',completed_at=time.time())
    except BaseException:
        for child,j in children:
            if child.poll() is None:child.terminate()
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
