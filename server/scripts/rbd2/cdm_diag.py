"""Diagnose the M3ID masked row: (1) rotary position ids row 1 gets at prefill and decode, (2) bf16 noise floor
(text-only prompt batch-1 vs the same prompt twice in a batch of 2), (3) masked row 1 vs text-only, top-5 / top-50 / KL.
usage: cdm_diag.py MODEL ATTN SET.jsonl"""
import json,sys
import torch
from transformers import LogitsProcessor,LogitsProcessorList
import cdm_decode as D
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C

def cmp(a,b):
    a=a.float().log_softmax(-1);b=b.float().log_softmax(-1);i5=b.topk(5).indices;i50=b.topk(50).indices
    kl=float((b.exp()*(b-a)).sum())
    return f"top1 {'=' if a.argmax()==b.argmax() else 'X'} top5 {float((a[i5]-b[i5]).abs().max()):.3f} top50 {float((a[i50]-b[i50]).abs().max()):.3f} KL {kl:.4f}"

def main(model,attn,data):
    config=dict(json.loads((C.MODELS[model]/'config.json').read_text())['model'],attn_implementation=attn)
    adapter=load_adapter(config);m=adapter.model;ref=C.read(data)[0]
    rot=[mod for n,mod in m.named_modules() if n.endswith('rotary_emb')][0];cap=[]
    h=rot.register_forward_hook(lambda mod,args,kw_out=None:None)
    h.remove()
    def pre(mod,args,kwargs):
        p=kwargs.get('position_ids',args[1] if len(args)>1 else None);cap.append(None if p is None else p.detach().cpu())
    h=rot.register_forward_pre_hook(pre,with_kwargs=True)
    class A:method='m3id';vcd_noise=500
    batch,info=D.pair(adapter,ref,A);batch={k:v.to(adapter.device) for k,v in batch.items()};s,e=info['masked_span']
    class P(LogitsProcessor):
        def __call__(self,ids,sc):sc[1]=sc[0];return sc
    with torch.no_grad():o=m.generate(**batch,do_sample=False,max_new_tokens=3,logits_processor=LogitsProcessorList([P()]))
    h.remove()
    print('captured',len(cap),'rotary calls; shapes',[None if c is None else tuple(c.shape) for c in cap[:4]])
    def rows(c):
        c=c.view(-1,c.shape[-2],c.shape[-1]);return c[:,0],c[:,1 if c.shape[1]>1 else 0]
    p0=cap[0];keep=[i for i in range(p0.shape[-1]) if not s<=i<e];r0,r1=rows(p0)
    print(' row1 kept positions:',[x[keep].tolist() for x in r1])
    print(' row0 last 3 positions:',[x[-3:].tolist() for x in r0])
    n=len(cap)//3   # rotary calls per forward (several rotary modules share the first match only, so usually 1)
    for k,c in enumerate(cap[1:3],1):a,b=rows(c);print(f' decode step {k} positions row0/row1:',a.flatten().tolist(),b.flatten().tolist())
    tb=adapter.prepare(None,ref['prompt'])
    with torch.no_grad():
        t1=m(**tb,use_cache=False).logits[0,-1]
        t2=m(**{k:torch.cat([v,v],0) for k,v in tb.items()},use_cache=False).logits[0,-1]
        pr=m(**batch,use_cache=False).logits[1,-1]
    print('noise floor  text b1 vs b2  :',cmp(t2,t1))
    print('masked row 1 vs text b1     :',cmp(pr,t1))
    print('masked row 1 vs text b2     :',cmp(pr,t2))

if __name__=='__main__':main(*sys.argv[1:4])
