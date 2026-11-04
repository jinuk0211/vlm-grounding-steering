"""Existing baselines on the caption test sets (chair_test60, amber_test60, throne_test60), through the functions
and fitted artifacts of the verified Qwen CHAIR/POPE test run (original-vti-gpu1-final):
  base           adapter.generate, BF16
  revis          own implementation: artifacts/revis.pt + build_baseline, BF16
  rudder_clone   author RUDDER, caption profile (layer 25, alpha 5), BF16
  grains_clone   author GrAInS + grains-author.pt (layer 27, alpha 10), FP16 weights, FP32 attention scores
  vti_author     author VTI + vti-author.pt (visual and text alpha 0.9), FP16 weights, FP32 attention scores
Only the GPU-UUID/lease/protocol guards of the original workers are dropped. Greedy, seed 42 per image, the set's
max_new_tokens. --attn sdpa is for BF16 methods only (the FP16 guard patches eager attention).
Rows: {output}/{method}-{start}-{end}.jsonl (resumable), same row format as the rbd2 runs.
"""
import argparse,contextlib,hashlib,json,os,time,traceback
from pathlib import Path
import torch
from author_clone_bridge import load_authors,rudder_context
from qwen_author_compat import grains_on_qwen_layer_device
from qwen_fp16_attention_guard import stable_fp16_attention
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed,_validate_artifact
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
Q=ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386'
VTI=ROOT/'outputs/qwen-original-vti-1793260357/vti-author.pt'
REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'
FP16=('grains_clone','vti_author')

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup(m,adapter,config):
    """Return ctx(batch, diag): the method's intervention for one image; diag collects its diagnostics."""
    if m=='base':return lambda batch,diag:contextlib.nullcontext()
    if m=='revis':
        art=torch.load(Q/'artifacts/revis.pt',map_location='cpu',weights_only=True)
        _validate_artifact(art,config['model'],adapter=adapter)
        for g in ('vectors','vision_vectors'):
            if g in art:art[g]={k:v.to(adapter.device) for k,v in art[g].items()}
        return lambda batch,diag:build_baseline(m,adapter,art,config['methods'][m])
    if m=='rudder_clone':
        authors=load_authors()
        return lambda batch,diag:rudder_context(adapter,batch,'chair',authors,diag)
    if m=='grains_clone':
        authors=load_authors();art=torch.load(Q/'grains-author.pt',map_location='cpu',weights_only=True)
        vector=art['vector'].to(adapter.device);assert vector.dtype==torch.float32,'Keep the original author vector dtype'
        return lambda batch,diag:grains_on_qwen_layer_device(adapter,vector,authors,diag)
    # vti_author: author VTILayers stay installed for the whole process, as in run_qwen_original_vti.py
    import run_qwen_original_vti as original
    _,layers=original.imports();art=torch.load(VTI,map_location='cpu',weights_only=True)
    assert art['author_commit']==original.COMMIT and art['author_source_hashes']==original.verify_author()
    original.install(adapter.vision_layers,art['visual_direction'][1:],.9,layers,vision=True)
    original.install(adapter.layers,art['textual_direction'][1:],.9,layers)
    current=[{}];hooks=0
    for name,module in adapter.model.named_modules():
        if isinstance(module,layers.VTILayer):
            kind='vision' if '.visual.' in name else 'text'
            def observe(_m,args,value,kind=kind):
                assert torch.isfinite(value).all()
                d=current[0].setdefault(kind,{'calls':0,'changed_elements':0,'max_abs_delta':0.0})
                diff=(value.float()-args[0].float()).abs();d['calls']+=1
                d['changed_elements']+=int((diff>0).sum());d['max_abs_delta']=max(d['max_abs_delta'],float(diff.max()))
            module.register_forward_hook(observe);hooks+=1
    assert hooks==len(adapter.layers)+len(adapter.vision_layers)
    def ctx(batch,diag):current[0]=diag;return contextlib.nullcontext()
    return ctx

def check(m,diag):
    if m in ('rudder_clone','grains_clone'):assert diag['calls']>0
    if m=='vti_author':assert all(diag.get(k,{}).get('changed_elements',0)>0 for k in ('text','vision'))

def main(a):
    m=a.method;fp16=m in FP16;assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    assert not (fp16 and a.attn!='eager'),'FP16 methods need eager attention for the FP32-score guard'
    config=json.loads((Q/'config.json').read_text())
    cfg=dict(config['model'],dtype='float16' if fp16 else 'bfloat16',attn_implementation=a.attn)
    refs=read(a.set)[a.start:a.end];out=a.output;out.mkdir(parents=True,exist_ok=True)
    name=f'{m}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    done=read(rows_file) if rows_file.exists() else []
    assert not done or rows_file.read_bytes().endswith(b'\n')
    for r,ref in zip(done,refs):assert r['id']==ref['id'] and r['method']==m
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),method=m,set=str(a.set),dtype=cfg['dtype'],attn=a.attn,
           total=len(refs),done=len(done),resumed_from=len(done))
    attention={}
    try:
        with stable_fp16_attention(attention) if fp16 else contextlib.nullcontext():
            adapter=load_adapter(cfg);assert adapter.model_type=='qwen2_5_vl' and len(adapter.layers)==28
            assert getattr(adapter.model.config,'_commit_hash',None) in (None,REVISION)
            adapter.model.config._commit_hash=REVISION
            ctx=setup(m,adapter,config);update(stage='running',gpu=torch.cuda.get_device_name(0))
            with rows_file.open('a') as f:
                for i,ref in enumerate(refs[len(done):],start=len(done)):
                    assert sha(ref['image'])==ref['image_sha256'],ref['image']
                    _seed(config);batch=adapter.prepare(ref['image'],ref['prompt']);diag={}
                    t0=time.perf_counter()
                    with ctx(batch,diag):text=adapter.generate(batch,max_new_tokens=int(ref['max_new_tokens']))
                    check(m,diag)
                    f.write(json.dumps({**ref,'method':m,'prediction':text,'generated_tokens':adapter.last_generated_tokens,
                        'elapsed_seconds':time.perf_counter()-t0,'dtype':cfg['dtype'],'attn_implementation':a.attn,
                        'attention_score_dtype':'float32' if fp16 else None,'gpu':torch.cuda.get_device_name(0),
                        'author_diagnostics':{k:str(v) for k,v in diag.items()}})+'\n');f.flush()
                    update(done=i+1)
        update(stage='complete',fp16_attention_calls=attention.get('calls'))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['base','revis','rudder_clone','grains_clone','vti_author'],required=True)
    p.add_argument('--set',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--attn',choices=['eager','sdpa'],default='eager')
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=60);main(p.parse_args())
