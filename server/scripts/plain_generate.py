"""Base greedy generation (same adapter.generate call as the Base runs) for any task jsonl. Resumable shard."""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    refs=C.read(a.data);a.end=min(a.end,len(refs))
    name=f'{a.tag}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    prefix=C.read(rows_file) if rows_file.exists() else []
    for r,i in zip(prefix,range(a.start,a.end)):assert r['id']==refs[i]['id']
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),model=a.model,data=str(a.data),range=[a.start,a.end],resumed_from=len(prefix),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);update(stage='running',gpu_name=torch.cuda.get_device_name(0))
        with rows_file.open('a') as f:
            for i in range(a.start+len(prefix),a.end):
                ref=refs[i];t0=time.perf_counter()
                text=adapter.generate(adapter.prepare(ref['image'],ref['prompt']),max_new_tokens=ref['max_new_tokens'])
                f.write(json.dumps({**ref,'method':'base','prediction':text,'generated_tokens':adapter.last_generated_tokens,
                    'elapsed_seconds':time.perf_counter()-t0,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],
                    'dtype':config['dtype'],'checkpoint':config['checkpoint'],'revision':config['revision']})+'\n');f.flush()
                update(done=i-a.start+1)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(C.MODELS),required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=10**9)
    main(p.parse_args())
