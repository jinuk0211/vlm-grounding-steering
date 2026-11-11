"""Existing baselines on the CHAIR dev split (COCO train2014, 100 images), through exactly the functions and
fitted artifacts of the Qwen test run (run_qwen_five_profile_worker.py):
  revis, vti     fitted artifacts (artifacts/*.pt), build_baseline, BF16
  rudder_clone   author RUDDER function (layer 25, alpha 5 for captions), BF16
  grains_clone   author GrAInS function + extracted vector (grains-author.pt), FP16 as in the test run
Only the GPU-pair/lease guards of the original worker are dropped. Decoding: adapter.generate (greedy, the
model's generation config), max 512 tokens, seed 42 per image. Rows match score_dev.py.
"""
import argparse,contextlib,json,os,time,traceback
from pathlib import Path
import torch
from author_clone_bridge import load_authors,rudder_context
from qwen_author_compat import grains_on_qwen_layer_device
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed,_validate_artifact
from rbd_common import ROOT
Q=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True);assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    config=json.loads((Q/'config.json').read_text());m=a.method
    cfg=dict(config['model'],dtype='float16' if m=='grains_clone' else 'bfloat16')
    refs=read(a.data)[a.start:a.end];tag=f'qwen-{m}';name=f'{tag}-{a.start}-{a.end}'
    rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json';done=read(rows_file) if rows_file.exists() else []
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),method=m,dtype=cfg['dtype'],resumed_from=len(done))
    try:
        adapter=load_adapter(cfg);artifact=vector=authors=None
        if m in ('revis','vti'):
            artifact=torch.load(Q/'artifacts'/f'{m}.pt',map_location='cpu',weights_only=True)
            _validate_artifact(artifact,config['model'],adapter=adapter)
            for g in ('vectors','vision_vectors'):
                if g in artifact:artifact[g]={k:v.to(adapter.device) for k,v in artifact[g].items()}
        if m in ('rudder_clone','grains_clone'):authors=load_authors()
        if m=='grains_clone':
            art=torch.load(Q/'grains-author.pt',map_location='cpu',weights_only=True);vector=art['vector'].to(adapter.device)
            assert vector.dtype==torch.float32
        update(stage='running')
        with rows_file.open('a') as f:
            for i,ref in enumerate(refs[len(done):],start=len(done)):
                _seed(config);batch=adapter.prepare(ref['image'],ref['prompt']);diag={}
                if m=='rudder_clone':ctx=rudder_context(adapter,batch,'chair',authors,diag)
                elif m=='grains_clone':ctx=grains_on_qwen_layer_device(adapter,vector,authors,diag)
                else:ctx=build_baseline(m,adapter,artifact,config['methods'][m])
                t0=time.perf_counter()
                with ctx:text=adapter.generate(batch,max_new_tokens=int(ref['max_new_tokens']))
                if m in ('rudder_clone','grains_clone'):assert diag['calls']>0
                f.write(json.dumps({**ref,'method':tag,'prediction':text,'generated_tokens':adapter.last_generated_tokens,
                                    'elapsed_seconds':time.perf_counter()-t0,'dtype':cfg['dtype'],'author_diagnostics':{k:str(v) for k,v in diag.items()}})+'\n');f.flush()
                update(done=i+1)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['revis','vti','rudder_clone','grains_clone'],required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/chair-dev-train2014-100/chair.jsonl');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=100);main(p.parse_args())
