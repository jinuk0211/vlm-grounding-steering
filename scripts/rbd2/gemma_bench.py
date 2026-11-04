"""Gemma 4 E4B baselines on the caption sets (11-03 user: full THRONE 1000 on Gemma), through the functions, parameters
and artifacts of the verified Gemma CHAIR/POPE run (gemma-grains-fp16-v2-20261029-1793246507, whose BF16 profiles come
from author-clone-rerun-20261027-165446; run_author_clone_worker.py and grains_fp16_worker_v2.py):
  base          adapter.generate, BF16
  revis         own implementation: artifacts/revis.pt + build_baseline (alpha 1.6, risk quantile 0.8), BF16
  vti           own implementation: artifacts/vti.pt + build_baseline (alpha 0.5, vision alpha 0.5), BF16 (user 11-03)
  rudder_clone  author RUDDER, caption profile (layer 25, alpha 5), BF16
  grains_clone  author GrAInS + grains-author.pt (layer 27, alpha 10), FP16 weights, FP32 vector as in the v2 worker
Eager attention for every method, as in that run. Only its GPU/lease/protocol guards are dropped; artifact hashes are
checked against its protocols. Greedy, seed 42 per image, the set's prompt and max_new_tokens.
Rows: {output}/{method}-{start}-{end}.jsonl (resumable), same row format as baseline_bench.py.
"""
import argparse,contextlib,hashlib,json,os,time,traceback
from pathlib import Path
import torch
from author_clone_bridge import load_authors,rudder_context,grains_context
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed,_validate_artifact
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
G=ROOT/'outputs/gemma-grains-fp16-v2-20261029-1793246507';OLD=ROOT/'outputs/author-clone-rerun-20261027-165446'
FP16=('grains_clone',)

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup(m,adapter,config):
    """Return ctx(batch, diag): the method's intervention for one image; diag collects its diagnostics."""
    if m=='base':return lambda batch,diag:contextlib.nullcontext()
    if m in ('revis','vti'):
        p=OLD/'artifacts'/f'{m}.pt';assert sha(p)==json.loads((OLD/'protocol.json').read_text())['retained_artifact_hashes'][m]
        art=torch.load(p,map_location='cpu',weights_only=True);_validate_artifact(art,config['model'],adapter=adapter)
        for g in ('vectors','vision_vectors'):
            if g in art:art[g]={k:v.to(adapter.device) for k,v in art[g].items()}
        return lambda batch,diag:build_baseline(m,adapter,art,config['methods'][m])
    authors=load_authors()
    if m=='rudder_clone':return lambda batch,diag:rudder_context(adapter,batch,'chair',authors,diag)
    proto=json.loads((G/'protocol.json').read_text());assert sha(proto['grains_artifact'])==proto['grains_artifact_sha256']
    art=torch.load(proto['grains_artifact'],map_location='cpu',weights_only=True)
    assert dict(art['model'],dtype='float16')==config['model']
    vector=art['vector'].to(adapter.device);assert vector.dtype==torch.float32 and torch.isfinite(vector).all() and vector.norm()>0
    return lambda batch,diag:grains_context(adapter,vector,authors,diag)

def main(a):
    m=a.method;fp16=m in FP16;assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    config=json.loads((G/'config.json').read_text());config['model']['dtype']='float16' if fp16 else 'bfloat16'
    assert config['model']['attn_implementation']=='eager'
    refs=read(a.set)[a.start:a.end];out=a.output;out.mkdir(parents=True,exist_ok=True)
    name=f'{m}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    done=read(rows_file) if rows_file.exists() else []
    assert not done or rows_file.read_bytes().endswith(b'\n')
    for r,ref in zip(done,refs):assert r['id']==ref['id'] and r['method']==m
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),method=m,set=str(a.set),dtype=config['model']['dtype'],attn='eager',
           total=len(refs),done=len(done),resumed_from=len(done))
    try:
        _seed(config);adapter=load_adapter(config['model']);assert adapter.model_type=='gemma4' and len(adapter.layers)==42
        # offline loads leave _commit_hash unset; check the snapshot on disk is the run's revision, then record it, as the
        # v2 GrAInS worker and baseline_bench.py do (build_baseline's artifact check compares it)
        rev=config['model']['revision'];assert getattr(adapter.model.config,'_commit_hash',None) in (None,rev)
        from huggingface_hub import try_to_load_from_cache
        cached=try_to_load_from_cache(config['model']['checkpoint'],'config.json',revision=rev)
        assert isinstance(cached,str) and Path(cached).parent.name==rev,cached
        adapter.model.config._commit_hash=rev
        dt={p.dtype for p in adapter.model.parameters() if p.is_floating_point()}
        assert (torch.float16 in dt and torch.bfloat16 not in dt) if fp16 else torch.bfloat16 in dt,dt
        ctx=setup(m,adapter,config);update(stage='running',gpu=torch.cuda.get_device_name(0))
        with rows_file.open('a') as f:
            for i,ref in enumerate(refs[len(done):],start=len(done)):
                assert sha(ref['image'])==ref['image_sha256'],ref['image']
                _seed(config);batch=adapter.prepare(ref['image'],ref['prompt']);diag={}
                t0=time.perf_counter()
                with ctx(batch,diag):text=adapter.generate(batch,max_new_tokens=int(ref['max_new_tokens']))
                if m in ('rudder_clone','grains_clone'):assert diag['calls']>0
                f.write(json.dumps({**ref,'method':m,'prediction':text,'generated_tokens':adapter.last_generated_tokens,
                    'elapsed_seconds':time.perf_counter()-t0,'model':'gemma-4-E4B-it','dtype':config['model']['dtype'],
                    'attn_implementation':'eager','gpu':torch.cuda.get_device_name(0),
                    'author_diagnostics':{k:str(v) for k,v in diag.items()}})+'\n');f.flush()
                update(done=i+1)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['base','revis','vti','rudder_clone','grains_clone'],required=True)
    p.add_argument('--set',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=60);main(p.parse_args())
