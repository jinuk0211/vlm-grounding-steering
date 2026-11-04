"""Resume an observed failed pre-generation author job, adopting healthy recovery."""
import json,os,subprocess,time,traceback
from pathlib import Path
from run_qwen_original_vti import ROOT,PARENT,GPU0,sha,save
from run_qwen_five_profile_suite import ticks
from launch_original_vti_replacement import live

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=Path(json.loads((PARENT/'original-vti-current.json').read_text())['output'])
    old=json.loads((out/'replacement-jobs.json').read_text());recovery=next(j for j in old if j['kind']=='donor_bf16_recovery')
    failed=next(j for j in old if j['kind']=='original_vti')
    assert not live(failed['pid'],failed['start_ticks'])
    previous=json.loads((out/'status.json').read_text());assert previous['stage']=='failed' and previous['completed_records']==0
    assert not any((out/f).exists() for f in ['pope.jsonl','chair.jsonl','vti-author.pt'])
    assert live(recovery['pid'],recovery['start_ticks'])
    archive=out/('failed-start-'+str(failed['pid']));archive.mkdir(exist_ok=True)
    for n in ['replacement-jobs.json','status.json','controller-status.json','protocol.json','worker-source.py']:
        if (out/n).exists():(archive/n).write_bytes((out/n).read_bytes())
    log=out/('original-vti-resume-'+str(int(time.time()))+'.log')
    with log.open('xb') as stream:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','scripts/run_qwen_original_vti.py','--output',str(out)],
            cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU0,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    job={'pid':child.pid,'start_ticks':ticks(child.pid),'kind':'original_vti','log':str(log),'output':str(out),
        'gpu_uuid':GPU0,'source_sha256':sha(ROOT/'scripts/run_qwen_original_vti.py')}
    state={'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'started_at':time.time(),'recovery_adopted':recovery,'original_vti':job}
    save(out/'resumed-jobs.json',{'author_job':job,'adopted_recovery_job':recovery,'failed_author_job':failed})
    def update(**kw):state.update(kw,updated_at=time.time());save(out/'resume-controller-status.json',state)
    try:
        while True:
            code=child.poll();s=json.loads((out/'status.json').read_text())
            if s.get('pid')!=child.pid:s={}
            assert code in [None,0] and s.get('stage')!='failed',{'code':code,'state':s}
            rp=PARENT/recovery['state_file'];rs=json.loads(rp.read_text());rlive=live(recovery['pid'],recovery['start_ticks'])
            if not rlive:assert rs['stage']=='complete'
            if code==0:
                assert s['stage']=='complete';job.update(exit_code=0,state_sha256=sha(out/'status.json'))
                save(out/'resumed-jobs.json',{'author_job':job,'adopted_recovery_job':recovery,'failed_author_job':failed})
            update(stage='running',author_stage=s.get('stage'),author_completed=s.get('completed_records'),
                recovery_live=rlive,recovery_stage=rs.get('stage'),recovery_completed=rs.get('completed_records'))
            if code==0 and not rlive:
                update(stage='complete',recovery_exit_observed=True,recovery_exit_code_unavailable=True,
                    recovery_state_sha256=sha(rp),completed_at=time.time());break
            time.sleep(10)
    except BaseException:update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':main()
