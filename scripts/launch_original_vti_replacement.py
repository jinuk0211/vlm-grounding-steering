"""Replace two supplemental replicas with one recovery and one author VTI job."""
import fcntl,json,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,GPU0,sha,save
from run_qwen_five_profile_suite import ticks
from run_qwen_original_vti import PARENT

def live(pid,start):
    p=Path('/proc')/str(pid)
    if not p.exists():return False
    fields=(p/'stat').read_text().split(') ',1)[1].split()
    return fields[0] not in ['Z','X'] and int(fields[19])==start

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    info=json.loads((PARENT/'original-vti-current.json').read_text());out=Path(info['output'])
    assert json.loads((out/'cpu-adapter-check.json').read_text())['status']=='passed'
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'started_at':time.time(),'gpu_indices':[0]}
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'controller-status.json',state)
    assert not (out/'replacement-jobs.json').exists()
    epoch=json.loads((PARENT/'vti-author-replacement-request.json').read_text())['lease_epoch'];released=[]
    if (out/'transition.json').exists():
        prior=json.loads((out/'transition.json').read_text())
        assert all(not live(j['pid'],j['start_ticks']) for j in prior['stopped_jobs'])
        c=prior['stopped_controller'];assert not live(c['controller_pid'],c['controller_start_ticks'])
    else:
        source=json.loads((PARENT/'supplemental-jobs.json').read_text());assert len(source)==2
        controller=json.loads((PARENT/'supplemental-status.json').read_text());cpid=controller['controller_pid']
        assert live(cpid,controller['controller_start_ticks'])
        for j in source:
            p=Path('/proc')/str(j['pid']);args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            assert live(j['pid'],j['start_ticks']) and p.stat().st_uid==os.getuid()
            assert args[args.index('--output')+1]==str(PARENT) and j['visible_devices']==GPU0
        archive=out/'superseded-supplemental-replicas';archive.mkdir()
        for n in ['supplemental-jobs.json','supplemental-status.json']:shutil.copy2(PARENT/n,archive/n)
        os.kill(cpid,signal.SIGTERM)
        for j in source:os.kill(j['pid'],signal.SIGTERM)
        deadline=time.monotonic()+15
        while any(live(j['pid'],j['start_ticks']) for j in source) or live(cpid,controller['controller_start_ticks']):
            assert time.monotonic()<deadline,'Selected replicas did not stop';time.sleep(.2)
        for t in json.loads((PARENT/'tasks.json').read_text()):
            claim=PARENT/'claims'/t['id'];owner=claim/'owner.json'
            if (claim/'complete.json').exists() or not owner.exists():continue
            d=json.loads(owner.read_text())
            if d.get('pid') not in [j['pid'] for j in source]:continue
            path=PARENT/'predictions'/(t['id']+'.jsonl')
            if path.exists():
                shutil.copy2(path,archive/path.name);raw=path.read_bytes();prefix=[]
                for line in raw.splitlines(keepends=True):
                    try:json.loads(line)
                    except json.JSONDecodeError:break
                    if not line.endswith(b'\n'):break
                    prefix.append(line)
                committed=b''.join(prefix)
                if committed!=raw:path.write_bytes(committed)
            lease=PARENT/'leases'/epoch/t['id']
            if lease.exists():lease.rename(archive/('lease-'+t['id']))
            released.append({'task':t['id'],'prefix_sha256':sha(path) if path.exists() else None})
        save(out/'transition.json',{'user_requested_original_vti':True,'stopped_jobs':source,
            'stopped_controller':controller,'all_selected_instances_exited':True,'released_incomplete_tasks':released,
            'retained_all_completed_predictions':True,'created_at':time.time()})
    children=[];jobs=[]
    try:
        update(stage='launching_gpu0_replacement')
        deadline=time.monotonic()+120
        while True:
            free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=43*1024:break
            assert time.monotonic()<deadline,free
            update(stage='waiting_cuda_memory_release',free_mib=free);time.sleep(2)
        name='gpu0-vti-author-recovery-'+str(int(time.time()))
        for kind,args,folder,statefile in [
            ('donor_bf16_recovery',['scripts/qwen_gpu0_stage_v2.py','--output',str(PARENT),'--stage','generation','--worker',name,'--dtype','bfloat16','--lease-epoch',epoch],PARENT,'worker-'+name+'.json'),
            ('original_vti',['scripts/run_qwen_original_vti.py','--output',str(out)],out,'status.json')]:
            log=out/(kind+'.log')
            with log.open('xb') as stream:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',*args],cwd=ROOT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            job={'pid':child.pid,'start_ticks':ticks(child.pid),'kind':kind,'args':args,'folder':str(folder),
                'state_file':statefile,'gpu_uuid':GPU0,'log':str(log),'source_sha256':sha(ROOT/args[0])}
            children.append((child,job));jobs.append(job);save(out/'replacement-jobs.json',jobs)
        while True:
            progress=[];done=True
            for child,j in children:
                code=child.poll();done&=code is not None;p=Path(j['folder'])/j['state_file']
                s=json.loads(p.read_text()) if p.exists() else {}
                if s.get('pid')!=j['pid']:s={}
                assert code in [None,0] and s.get('stage')!='failed',{'job':j,'code':code,'state':s}
                if code==0:j.update(exit_code=0,state_sha256=sha(p),completed_at=time.time())
                progress.append({'kind':j['kind'],'pid':j['pid'],'stage':s.get('stage'),'completed':s.get('completed_records')})
            save(out/'replacement-jobs.json',jobs);update(stage='running_gpu0_replacement',jobs=progress)
            if done:break
            time.sleep(10)
        update(stage='complete',completed_at=time.time())
    except BaseException:
        # Each benchmark remains resumable. Preserve a surviving healthy job.
        update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
