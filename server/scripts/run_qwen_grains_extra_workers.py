"""Two more GPU0 replicas of the unchanged original GrAInS worker."""
import fcntl,json,os,shutil,subprocess,time,traceback
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save,verify_protocol
from qwen_fp16_stable_stage import verify_attention_amendment
from run_qwen_five_profile_suite import ticks

PARENT=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'

def obj(p):return json.loads(Path(p).read_text())
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    vti=Path(obj(PARENT/'original-vti-current.json')['output'])
    lock=(vti/'grains-extra-controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (vti/'grains-extra-jobs.json').exists(),'Extra workers already launched'
    protocol=verify_protocol(PARENT);attention=verify_attention_amendment(PARENT)
    original=next(j for j in obj(PARENT/'fp16-stable-jobs.json') if j['kind']=='generation')
    proc=Path('/proc')/str(original['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    assert int(fields[19])==original['start_ticks'] and fields[0] not in ['Z','X'] and proc.stat().st_uid==os.getuid()
    args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    assert args[args.index('--output')+1]==str(PARENT) and args[args.index('--dtype')+1]=='float16'
    epoch=args[args.index('--lease-epoch')+1];assert epoch==original['worker']
    worker='scripts/qwen_fp16_stable_stage.py'
    assert sha(ROOT/worker)==attention['source_hashes'][worker]
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=45*1024,free
    sources=[worker,'scripts/run_qwen_grains_extra_workers.py','scripts/audit_qwen_vti_replacement.py','scripts/finalize_qwen_vti_replacement.py']
    snapshot=vti/'grains-extra-source-snapshot';snapshot.mkdir()
    hashes={}
    for source in sources:
        target=snapshot/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,target);hashes[source]=sha(target)
    shutil.copy2(__file__,vti/'grains-extra-controller-source.py')
    save(vti/'grains-extra-amendment.json',{'user_request':'Use remaining GPU0 VRAM for remaining experiments',
        'parent_protocol_sha256':protocol['sha256'],'vti_protocol_sha256':obj(vti/'protocol.json')['sha256'],
        'gpu_indices':[0],'same_lease_epoch':epoch,'replicas':2,'initial_free_mib':free,
        'controller_source_file':'scripts/run_qwen_grains_extra_workers.py','source_hashes':hashes,
        'source_snapshot_dir':'grains-extra-source-snapshot','original_worker_identity':{'pid':original['pid'],'start_ticks':original['start_ticks']},
        'previous_finalizer_stop_proof_sha256':sha(vti/'grains-extra-finalizer-transition.json'),
        'model_math_data_vector_precision_unchanged':True,'created_at':time.time()})
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'gpu_indices':[0],
        'source_sha256':sha(__file__),'started_at':time.time(),'same_lease_epoch':epoch}
    def update(**kw):state.update(kw,updated_at=time.time());save(vti/'grains-extra-status.json',state)
    children=[];jobs=[]
    try:
        update(stage='launching_extra_grains_gpu0')
        stamp=str(int(time.time()))
        for i in range(2):
            name=f'gpu0-extra-grains-{i}-{stamp}';log=vti/(name+'.log')
            args=[str(ROOT/'.venv/bin/python'),'-u',worker,'--output',str(PARENT),'--stage','generation',
                '--worker',name,'--dtype','float16','--lease-epoch',epoch]
            with log.open('xb') as stream:
                child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0,PYTHONDONTWRITEBYTECODE='1'),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            job={'pid':child.pid,'start_ticks':ticks(child.pid),'script':worker,'source_file':worker,'source_sha256':hashes[worker],
                'output':str(PARENT),'kind':'generation','visible_devices':GPU0,'dtype':'float16',
                'worker':name,'state_file':'worker-'+name+'.json','device_proof':name+'-device-map.json',
                'log':str(log),'lease_epoch':epoch,'fp16_stable_attention':True,'grains_extra_replica':True}
            children.append((child,job));jobs.append(job);save(vti/'grains-extra-jobs.json',jobs)
        while True:
            done=True;progress=[]
            for child,j in children:
                code=child.poll();done &= code is not None;path=PARENT/j['state_file']
                s=obj(path) if path.exists() else {}
                if s.get('pid')!=j['pid']:s={}
                assert code in [None,0] and s.get('stage')!='failed',{'job':j,'code':code,'state':s}
                if code==0:
                    assert s['stage']=='complete'
                    j.update(exit_code=0,state_sha256=sha(path),device_proof_sha256=sha(PARENT/j['device_proof']))
                progress.append({'pid':j['pid'],'worker':j['worker'],'stage':s.get('stage'),
                    'completed_records':s.get('completed_records'),'task':s.get('task')})
            save(vti/'grains-extra-jobs.json',jobs);update(stage='generating_extra_grains_gpu0',jobs=progress)
            if done:break
            time.sleep(10)
        update(stage='complete',completed_at=time.time())
    except BaseException:update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
