"""Qwen3-VL-8B CHAIR rows for the official REVIS and GrAInS, same preparation, generation and row format as qwen3_bench.py (adapter.prepare, adapter.generate, seed per image).
  revis_official  upstream/Revis MADRSteeringHook_visual_only, unmodified, with outputs/qwen3-vl8b/revis-official (layer, tau from calibration; alpha 1.6, gamma 1), BF16 sdpa
  grains_clone    author GrAInS through author_clone_bridge.grains_context + grains-author.pt (layer 27, alpha 10), FP16 weights, eager with FP32 attention scores as for Qwen2.5
  base            no intervention, BF16 sdpa
usage: qwen3_bl.py --method M --attn A --set SET --output DIR --start S --end E"""
import argparse,contextlib,hashlib,importlib.util,json,os,time,traceback
from contextlib import contextmanager
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
from vlm_grounding.cli import _seed
ROOT=Path('/home/project-b/LJW/vlm-grounding-steering');V=ROOT/'outputs/qwen3-vl8b'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@contextmanager
def stable_fp16_attention(diag):
    """scripts/qwen_fp16_attention_guard.py for the Qwen3-VL module"""
    import transformers.models.qwen3_vl.modeling_qwen3_vl as q
    original=q.eager_attention_forward;diag['calls']=0
    def forward(module,query,key,value,attention_mask,scaling,dropout=0.0,**kw):
        if query.dtype!=torch.float16:return original(module,query,key,value,attention_mask,scaling,dropout,**kw)
        keys=q.repeat_kv(key,module.num_key_value_groups);values=q.repeat_kv(value,module.num_key_value_groups)
        w=torch.matmul(query.float(),keys.float().transpose(2,3))*scaling
        if attention_mask is not None:w=w+attention_mask.float()
        p=torch.nn.functional.softmax(w,dim=-1,dtype=torch.float32).to(query.dtype);p=torch.nn.functional.dropout(p,p=dropout,training=module.training)
        o=torch.matmul(p,values).transpose(1,2).contiguous();assert torch.isfinite(o).all();diag['calls']+=1;return o,p
    q.eager_attention_forward=forward
    try:yield
    finally:q.eager_attention_forward=original
def setup(m,a):
    if m=='base':return lambda b,d:contextlib.nullcontext()
    if m=='grains_clone':
        from author_clone_bridge import load_authors,grains_context
        art=torch.load(V/'grains-author.pt',map_location='cpu',weights_only=True);vec=art['vector'].to(a.device);assert vec.dtype==torch.float32 and art['layer']==27;authors=load_authors()
        return lambda b,d:grains_context(a,vec,authors,d)
    assert m=='revis_official'
    s=importlib.util.spec_from_file_location('revis_llm_layers',ROOT/'upstream/Revis/steering_utils/llm_layers.py');L=importlib.util.module_from_spec(s);s.loader.exec_module(L)
    c=json.loads((V/'revis-official/calibration.json').read_text());v=torch.load(V/'revis-official/vectors.pt',map_location='cpu',weights_only=True);k=int(c['layer']);tau=float(c['tau']);cur=[{}]
    u=v[k+1]['visual']
    def count(_m,_i,out):
        h=out[0] if isinstance(out,tuple) else out;x=torch.nn.functional.normalize(h[:,-1,:].float(),dim=-1);w=torch.nn.functional.normalize(u.to(device=h.device).float(),dim=-1)
        d=cur[0];d['calls']=d.get('calls',0)+1;d['fired']=d.get('fired',0)+int((-torch.matmul(x,w)>tau).sum())
    a.layers[k].register_forward_hook(count)
    L.MADRSteeringHook_visual_only(vectors=v[k+1],layer_idx=k,alpha_vis=float(c['alpha']),tau_low=tau,risk_gamma=1.0).attach(a.layers[k])
    def ctx(b,d):cur[0]=d;return contextlib.nullcontext()
    return ctx
def main(x):
    m=x.method;fp16=m=='grains_clone';assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1;assert not fp16 or x.attn=='eager'
    config=json.loads((ROOT/'configs/qwen3_vl_8b.json').read_text());cfg=dict(config['model'],dtype='float16' if fp16 else 'bfloat16',attn_implementation=x.attn)
    refs=read(x.set)[x.start:x.end];out=x.output;out.mkdir(parents=True,exist_ok=True);name=f'{m}-{x.start}-{x.end}';rf=out/f'{name}.jsonl';st=out/f'{name}-status.json'
    done=read(rf) if rf.exists() else []
    for r,ref in zip(done,refs):assert r['id']==ref['id'] and r['method']==m
    def update(**k):
        s=json.loads(st.read_text()) if st.exists() else {};s.update(k,updated_at=time.time());st.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),method=m,dtype=cfg['dtype'],attn=x.attn,total=len(refs),done=len(done),resumed_from=len(done));att={}
    try:
        with stable_fp16_attention(att) if fp16 else contextlib.nullcontext():
            a=load_adapter(cfg);assert a.model_type=='qwen3_vl' and len(a.layers)==36;ctx=setup(m,a);update(stage='running',gpu=torch.cuda.get_device_name(0))
            with rf.open('a') as f:
                for i,ref in enumerate(refs[len(done):],start=len(done)):
                    assert sha(ref['image'])==ref['image_sha256'];_seed(config);b=a.prepare(ref['image'],ref['prompt']);diag={};t0=time.perf_counter()
                    with ctx(b,diag):text=a.generate(b,max_new_tokens=int(ref['max_new_tokens']))
                    if m!='base':assert diag.get('calls',0)>0
                    f.write(json.dumps({**ref,'method':m,'prediction':text,'generated_tokens':a.last_generated_tokens,'elapsed_seconds':time.perf_counter()-t0,'dtype':cfg['dtype'],'attn_implementation':x.attn,
                        'attention_score_dtype':'float32' if fp16 else None,'gpu':torch.cuda.get_device_name(0),'author_diagnostics':{k:str(v) for k,v in diag.items()}})+'\n');f.flush();update(done=i+1)
        update(stage='complete',fp16_attention_calls=att.get('calls'))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['base','revis_official','grains_clone'],required=True);p.add_argument('--attn',choices=['eager','sdpa'],required=True);p.add_argument('--set',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=500);main(p.parse_args())
