"""Join preserved GPU0 prefixes and disjoint GPU2/3 VTI CHAIR continuations."""
import json
import os
import signal
import time
import traceback
from pathlib import Path

import run_qwen_original_vti as original
from qwen_gpu0_migration_api import ROOT,save,sha

V=ROOT/'outputs/qwen-original-vti-1793260357'
ORIGINAL=(4107032,331187361)
OLD_CONTROLLER=(1440572,None)
CUTOFFS=[60,227,394]
GPU0_RANGES=[(0,167),(167,334),(334,500)]
JOIN=[(0,60,'gpu0',0),(60,167,'gpu2',None),(167,227,'gpu0',1),
      (227,334,'gpu3',None),(334,394,'gpu0',2),(394,447,'gpu2',None),(447,500,'gpu3',None)]

def proc(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists():return None
    fields=p.read_text().split()
    return fields[2],int(fields[21])

def live(pid,ticks):
    p=proc(pid)
    return p is not None and p[0]!='Z' and p[1]==ticks

def main():
    folder=V/'chair-mixed';folder.mkdir(exist_ok=False)
    previous=json.loads((V/'chair-parallel/controller-status.json').read_text())
    assert previous['stage']=='generating' and previous['pid']==OLD_CONTROLLER[0]
    assert live(previous['pid'],previous['start_ticks'])
    assert len(previous['jobs'])==3
    for job in previous['jobs']:assert live(job['pid'],job['start_ticks'])
    assert not (V/'chair.jsonl').exists()
    state={'pid':os.getpid(),'start_ticks':proc(os.getpid())[1],
           'source_sha256':sha(__file__),'stage':'taking_over',
           'old_controller':{'pid':previous['pid'],'start_ticks':previous['start_ticks']},
           'gpu0_jobs':previous['jobs'],'gpu0_prefix_cutoffs':CUTOFFS,
           'gpu23_amendment_sha256':sha(V/'chair-gpu23-amendment.json'),
           'gpu0_stops':[],'original_paused':False}
    def update(**kwargs):state.update(kwargs,updated_at=time.time());save(folder/'status.json',state)
    update()
    try:
        os.kill(previous['pid'],signal.SIGTERM)
        for _ in range(100):
            if not live(previous['pid'],previous['start_ticks']):break
            time.sleep(.1)
        assert not live(previous['pid'],previous['start_ticks'])
        save(folder/'old-controller-stop.json',{'pid':previous['pid'],
             'start_ticks':previous['start_ticks'],'old_status_sha256':sha(V/'chair-parallel/controller-status.json'),
             'stopped_at':time.time(),'verified_exited':True})
        update(stage='generating_disjoint_ranges',old_controller_stop_sha256=sha(folder/'old-controller-stop.json'))
        stopped=set()
        while len(stopped)<3 or any(not (V/'chair-gpu23'/f'gpu{i}-status.json').exists() or
                                     json.loads((V/'chair-gpu23'/f'gpu{i}-status.json').read_text()).get('stage')!='complete'
                                     for i in (2,3)):
            progress=json.loads((V/'status.json').read_text())
            if (not state['original_paused'] and progress.get('benchmark')=='pope' and
                progress.get('benchmark_completed',0)>=2850):
                assert progress['pid']==ORIGINAL[0] and live(*ORIGINAL)
                os.kill(ORIGINAL[0],signal.SIGSTOP)
                update(original_paused=True,original_pope_at_pause=progress['benchmark_completed'])
            for i,(start,end) in enumerate(GPU0_RANGES):
                if i in stopped:continue
                job=previous['jobs'][i];label=f'{start:03d}-{end:03d}'
                rows=original.read(V/'chair-parallel'/(label+'.jsonl'))
                if len(rows)>=CUTOFFS[i]-start:
                    assert live(job['pid'],job['start_ticks'])
                    os.kill(job['pid'],signal.SIGTERM)
                    for _ in range(100):
                        if not live(job['pid'],job['start_ticks']):break
                        time.sleep(.1)
                    assert not live(job['pid'],job['start_ticks'])
                    state['gpu0_stops'].append({'pid':job['pid'],'start_ticks':job['start_ticks'],
                        'range':[start,end],'prefix_kept':CUTOFFS[i]-start,
                        'rows_present':len(original.read(V/'chair-parallel'/(label+'.jsonl'))),
                        'file_sha256':sha(V/'chair-parallel'/(label+'.jsonl')),'stopped_at':time.time()})
                    stopped.add(i);update()
            for gpu in (2,3):
                path=V/'chair-gpu23'/f'gpu{gpu}-status.json'
                if path.exists():
                    s=json.loads(path.read_text())
                    assert s['stage']!='failed',s.get('error')
            if len(stopped)<3 or any(not (V/'chair-gpu23'/f'gpu{i}-status.json').exists() or
                                     json.loads((V/'chair-gpu23'/f'gpu{i}-status.json').read_text()).get('stage')!='complete'
                                     for i in (2,3)):
                time.sleep(5)
        refs=original.read(original.PARENT/'data/chair.jsonl')
        gpu0=[original.read(V/'chair-parallel'/f'{a:03d}-{b:03d}.jsonl') for a,b in GPU0_RANGES]
        gpu23={i:original.read(V/'chair-gpu23'/f'gpu{i}.jsonl') for i in (2,3)}
        by_id={i:{r['id']:r for r in rows} for i,rows in gpu23.items()}
        result=[]
        for start,end,source,index in JOIN:
            if source=='gpu0':result.extend(gpu0[index][:end-start])
            else:result.extend(by_id[int(source[-1])][r['id']] for r in refs[start:end])
        assert len(result)==500 and not (V/'chair.jsonl').exists()
        for row,ref in zip(result,refs):
            assert all(row.get(k)==v for k,v in ref.items())
            assert row['method']=='vti_author'
        temp=V/'chair-mixed-merge.tmp'
        with temp.open('x') as sink:
            for row in result:sink.write(json.dumps(row,allow_nan=False)+'\n')
        temp.replace(V/'chair.jsonl')
        if state['original_paused']:
            assert live(*ORIGINAL);os.kill(ORIGINAL[0],signal.SIGCONT)
            state['original_paused']=False
        update(stage='complete',chair_sha256=sha(V/'chair.jsonl'),completed_records=500,
               completed_at=time.time())
    except BaseException:
        update(stage='failed',error=traceback.format_exc())
        raise
    finally:
        if state['original_paused'] and live(*ORIGINAL):os.kill(ORIGINAL[0],signal.SIGCONT)

if __name__=='__main__':main()
