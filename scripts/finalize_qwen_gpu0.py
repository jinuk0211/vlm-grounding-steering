"""Wait on the five actual generation workers, then run the complete CPU audit."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from qwen_gpu0_migration_api import ROOT,sha,save,verify_protocol
from run_qwen_five_profile_suite import ticks

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
    lock=(out/'finalization.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=verify_protocol(out)
    sources=['scripts/audit_qwen_gpu0_final.py','scripts/finalize_qwen_gpu0.py']
    assert not (out/'gpu0-finalization-amendment.json').exists()
    dependencies=['gpu0-hardware-amendment.json','gpu0-repair-amendment.json','gpu0-continuation-amendment.json',
        'gpu0-bf16-priority-amendment.json','gpu0-extra-replicas-amendment.json','fp16-attention-amendment.json',
        'fp16-stable-controller-amendment.json','fp16-cpu-diagnostic.json']
    save(out/'gpu0-finalization-amendment.json',{'protocol_sha256':protocol['sha256'],'source_hashes':{p:sha(ROOT/p) for p in sources},
        'amendment_hashes':{p:sha(out/p) for p in dependencies},
        'purpose':'Full five-method 17500-response audit after all five GPU0 generation workers actually exit','created_at':time.time()})
    import shutil
    for p in sources:
        dest=out/'finalization-source-snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    state={'controller_pid':os.getpid(),'controller_start_ticks':ticks(os.getpid()),'started_at':time.time(),'gpu_indices':[0]}
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'finalization-status.json',state)
    receipt={'pid':os.getpid(),'start_ticks':state['controller_start_ticks'],'script':'scripts/finalize_qwen_gpu0.py','output':str(out),
        'gpu_indices':[0],'log':str(out/'finalization-controller.log')}
    save(ROOT/'outputs/qwen-finalization-current.json',receipt)
    expected={'status.json':'bf16_generation_complete_fp16_pending','supplemental-status.json':'complete','fp16-stable-status.json':'complete'}
    def live(pid,start):
        proc=Path('/proc')/str(pid)
        try:
            s=(proc/'stat').read_text().split(') ',1)[1].split()
            return s[0] not in ('Z','X') and int(s[19])==start and proc.stat().st_uid==os.getuid()
        except FileNotFoundError:return False
    try:
        while True:
            progress=[];done=True
            for name,stage in expected.items():
                s=json.loads((out/name).read_text());active=live(s['controller_pid'],s['controller_start_ticks'])
                assert s['stage']!='failed',s
                if not active:assert s['stage']==stage,{'name':name,'state':s}
                done &= not active
                progress.append({'file':name,'pid':s['controller_pid'],'live':active,'stage':s['stage']})
            for jf in ['jobs.json','supplemental-jobs.json','fp16-stable-jobs.json']:
                for j in json.loads((out/jf).read_text()):
                    if j['kind']=='generation':
                        active=live(j['pid'],j['start_ticks']);done &= not active
                        if active:
                            command=(Path('/proc')/str(j['pid'])/'cmdline').read_bytes().split(b'\0')
                            assert j['script'].encode() in command and str(out).encode() in command
                        p=out/j['state_file'];s=json.loads(p.read_text()) if p.exists() else {}
                        assert not s or s.get('pid')==j['pid']
                        assert s.get('stage')!='failed',s
                        progress.append({'pid':j['pid'],'live':active,'stage':s.get('stage'),'completed':s.get('completed_records')})
            update(stage='waiting_for_five_gpu0_workers',progress=progress)
            if done:break
            time.sleep(30)
        update(stage='auditing_all_17500_responses')
        with (out/'audit-gpu0-final.log').open('xb') as stream:
            p=subprocess.run([str(ROOT/'.venv/bin/python'),'-u','scripts/audit_qwen_gpu0_final.py','--output',str(out)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT)
        assert p.returncode==0,p.returncode
        report=json.loads((out/'report.json').read_text());assert report['status']=='verified_complete' and report['audit']['all_17500_responses_verified']
        update(stage='server_verified_complete_pending_local_export',report_sha256=sha(out/'report.json'),completed_at=time.time())
    except BaseException:update(stage='failed',error=traceback.format_exc());raise
    finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
