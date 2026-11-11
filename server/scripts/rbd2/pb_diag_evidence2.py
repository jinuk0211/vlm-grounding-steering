"""Stage B2 (.venv, GPU): same teacher-forced Base captions and gray null as pb_diag_evidence.py, but compare ways to
normalise the visual evidence d_t(o) before the belief (A = sum r d, belief = clip(A / sum r)):
  raw      log p_real(o) - log p_gray(o)                      (full-vocabulary log-softmax; the method as run)
  vobj     same, log-softmax renormalised over V_obj only      (removes the shift from how peaked each distribution is)
  median   raw - median over V_obj at that step                (removes the shift common to all objects at that step)
Each is read with and without the penalty (pen 0.5): adj = lam w belief [- pen w], w = 1 - r(t).
usage: pb_diag_evidence2.py MENTIONS_JSON OUT_JSON"""
import json,sys
from pathlib import Path
import numpy as np,torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from rbd_common import ROOT,r_curve

LAM,PEN,CLIP=1.0,0.5,5.0
items=json.load(open(sys.argv[1]));out=Path(sys.argv[2]);gray=str(out.parent/'nulls'/'gray448.png')
from PIL import Image
Image.new('RGB',(448,448),(128,128,128)).save(gray)
config=dict(json.loads((C.MODELS['qwen']/'config.json').read_text())['model'],attn_implementation='sdpa')
ad=load_adapter(config);m=ad.model;tok=ad.processor.tokenizer
vobj=json.loads((ROOT/'outputs/rbd-20261030/collect/qwen-vobj2.json').read_text());col={v:i for i,v in enumerate(vobj)}
vidx=torch.tensor(vobj,device=ad.device);rf=r_curve('qwen','measured')
rec=[]
with torch.no_grad():
    for n,it in enumerate(items):
        b0=ad.prepare(it['image'],it['prompt'],answer=it['caption']);lab=b0.pop('labels')[0];cs=int((lab!=-100).nonzero()[0])
        ids=b0['input_ids'][0];L=len(ids);tgt=ids[cs:].tolist();T=len(tgt)
        r=np.array([rf(t) for t in range(T)]);w=1-r;Rc=np.cumsum(r)
        words=[tok.decode([x]).strip().lower() for x in tgt];pos={}
        for mi,mn in enumerate(it['mentions']):
            cand=set(mn['words'])|{x+'s' for x in mn['words']}|{x+'es' for x in mn['words']}
            t=next((t for t,x in enumerate(words) if x in cand and tgt[t] in col),None)
            if t is not None:pos[mi]=t
        b1=ad.prepare(gray,it['prompt'],answer=it['caption']);b1.pop('labels');assert torch.equal(b1['input_ids'],b0['input_ids'])
        lg=m(**{k:torch.cat([b0[k],b1[k]],0) for k in b0},use_cache=False,return_dict=True).logits[:,cs-1:L-1].float()
        full=lg.log_softmax(-1)[:,:,vidx];sub=lg[:,:,vidx].log_softmax(-1)
        raw=(full[0]-full[1]).cpu().numpy()
        D={'raw':raw,'vobj':(sub[0]-sub[1]).cpu().numpy(),'median':raw-np.median(raw,1,keepdims=True)}
        for k,d in D.items():
            bel=np.clip(np.cumsum(r[:,None]*d,0)/np.maximum(Rc,1e-6)[:,None],-CLIP,CLIP)
            for mi,t in pos.items():
                j=col[tgt[t]];mn=it['mentions'][mi]
                rec.append({'norm':k,'id':it['id'],'object':mn['object'],'kind':mn['kind'],'stuff':mn['stuff'],'t':t,
                            'belief':float(bel[t,j]),'adj':float(LAM*w[t]*bel[t,j]-PEN*w[t]),'adj_nopen':float(LAM*w[t]*bel[t,j])})
        if n%20==0:print(n,len(rec),flush=True)
json.dump(rec,open(out,'w'))

def auroc(p,q):
    p=np.asarray(p);q=np.asarray(q)
    if not len(p) or not len(q):return float('nan')
    x=np.concatenate([p,q]);_,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    rk=(np.cumsum(cnt)-(cnt-1)/2)[inv];return (rk[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(q))
for k in ('raw','vobj','median'):
    R_=[x for x in rec if x['norm']==k]
    def g(kind,stuff=None):return [x for x in R_ if x['kind'].startswith(kind) and (stuff is None or x['stuff']==stuff)]
    def mean(xs,f):return f"{np.mean([x[f] for x in xs]):+.2f}" if xs else '  -  '
    def neg(xs,f):return f"{np.mean([x[f]<0 for x in xs]):.2f}" if xs else '-'
    print(f"[{k:6s}] AUROC truth>hal {auroc([x['belief'] for x in g('truth')],[x['belief'] for x in g('hallucinated')]):.3f}"
          f" (stuff truth>hal {auroc([x['belief'] for x in g('truth',True)],[x['belief'] for x in g('hallucinated')]):.3f})"
          f" | belief truth {mean(g('truth'),'belief')} stuff {mean(g('truth',True),'belief')} hal {mean(g('hallucinated'),'belief')}"
          f" | pushed down, with pen: truth {neg(g('truth'),'adj')} stuff {neg(g('truth',True),'adj')} hal {neg(g('hallucinated'),'adj')}"
          f" | without pen: truth {neg(g('truth'),'adj_nopen')} stuff {neg(g('truth',True),'adj_nopen')} hal {neg(g('hallucinated'),'adj_nopen')}")
