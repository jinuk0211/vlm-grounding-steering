"""Stage B (.venv, GPU): teacher-force the Base caption with the real image and a null image (batch of two, as
vdd_decode does), rebuild the visual-evidence belief exactly as main applies it (lam 1, pen 0.5, w = 1 - r(t), clip 5)
and read it at the first token of each mention from stage A. Null images, all made from the 448x448 input:
  gray     the method's null (RGB 128)
  blur     Gaussian blur, radius 24
  mean     solid image of the input's mean colour
  shuffle  16x16 patches permuted (seed 0)
usage: pb_diag_evidence.py MENTIONS_JSON OUT_JSON"""
import json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageFilter,ImageOps
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from rbd_common import ROOT,r_curve

LAM,PEN,CLIP=1.0,0.5,5.0
items=json.load(open(sys.argv[1]));out=Path(sys.argv[2]);nd=out.parent/'nulls';nd.mkdir(parents=True,exist_ok=True)
config=dict(json.loads((C.MODELS['qwen']/'config.json').read_text())['model'],attn_implementation='sdpa')
ad=load_adapter(config);m=ad.model;tok=ad.processor.tokenizer
vobj=json.loads((ROOT/'outputs/rbd-20261030/collect/qwen-vobj2.json').read_text());col={v:i for i,v in enumerate(vobj)}
vidx=torch.tensor(vobj,device=ad.device);rf=r_curve('qwen','measured')

def nulls(path,key):
    with Image.open(path) as s:im=ImageOps.exif_transpose(s).convert('RGB').resize((448,448),Image.Resampling.BICUBIC)
    a=np.asarray(im);P={'gray':Image.new('RGB',(448,448),(128,128,128)),'blur':im.filter(ImageFilter.GaussianBlur(24)),
                        'mean':Image.new('RGB',(448,448),tuple(int(x) for x in a.reshape(-1,3).mean(0)))}
    p=a.reshape(28,16,28,16,3).transpose(0,2,1,3,4).reshape(784,16,16,3)[np.random.default_rng(0).permutation(784)]
    P['shuffle']=Image.fromarray(np.ascontiguousarray(p.reshape(28,28,16,16,3).transpose(0,2,1,3,4).reshape(448,448,3)))
    paths={}
    for k,img in P.items():q=nd/f'{key}-{k}.png';img.save(q);paths[k]=str(q)
    return paths

rec=[]
with torch.no_grad():
    for n,it in enumerate(items):
        paths=nulls(it['image'],it['id'].replace(':','_'))
        b0=ad.prepare(it['image'],it['prompt'],answer=it['caption']);lab=b0.pop('labels')[0];cs=int((lab!=-100).nonzero()[0])
        ids=b0['input_ids'][0];L=len(ids);tgt=ids[cs:].tolist();T=len(tgt)
        r=np.array([rf(t) for t in range(T)]);w=1-r
        words=[tok.decode([x]).strip().lower() for x in tgt];pos={}
        for mi,mn in enumerate(it['mentions']):
            cand=set(mn['words'])|{x+'s' for x in mn['words']}|{x+'es' for x in mn['words']}
            t=next((t for t,x in enumerate(words) if x in cand and tgt[t] in col),None)
            if t is not None:pos[mi]=t
        for k,pth in paths.items():
            b1=ad.prepare(pth,it['prompt'],answer=it['caption']);b1.pop('labels');assert torch.equal(b1['input_ids'],b0['input_ids'])
            batch={kk:torch.cat([b0[kk],b1[kk]],0) for kk in b0}
            lp=m(**batch,use_cache=False,return_dict=True).logits[:,cs-1:L-1].float().log_softmax(-1)[:,:,vidx]
            d=(lp[0]-lp[1]).cpu().numpy();A=np.cumsum(r[:,None]*d,0);Rc=np.cumsum(r)
            bel=np.clip(A/np.maximum(Rc,1e-6)[:,None],-CLIP,CLIP)
            for mi,t in pos.items():
                j=col[tgt[t]];mn=it['mentions'][mi]
                rec.append({'null':k,'id':it['id'],'object':mn['object'],'kind':mn['kind'],'stuff':mn['stuff'],'t':t,'rel':t/T,
                            'inst':float(d[t,j]),'belief':float(bel[t,j]),'adj':float(LAM*w[t]*bel[t,j]-PEN*w[t])})
        if n%10==0:print(n,len(rec),flush=True)
json.dump(rec,open(out,'w'))

def auroc(p,q):
    p=np.asarray(p);q=np.asarray(q)
    if not len(p) or not len(q):return float('nan')
    x=np.concatenate([p,q]);_,inv,cnt=np.unique(x,return_inverse=True,return_counts=True)
    rk=(np.cumsum(cnt)-(cnt-1)/2)[inv];return (rk[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(q))
found=len({(x['id'],x['object']) for x in rec if x['null']=='gray'});total=sum(len(it['mentions']) for it in items)
print(f'mentions located in token space: {found}/{total}')
for k in ('gray','blur','mean','shuffle'):
    R_=[x for x in rec if x['null']==k]
    def g(kind,stuff=None):return [x for x in R_ if x['kind'].startswith(kind) and (stuff is None or x['stuff']==stuff)]
    def s(xs,f):return f"{np.mean([x[f] for x in xs]):+.2f}" if xs else '  -  '
    def neg(xs):return f"{np.mean([x['adj']<0 for x in xs]):.2f}" if xs else '-'
    print(f"[{k:7s}] belief: lost {s(g('truth_lost'),'belief')} (stuff {s(g('truth_lost',True),'belief')} / things {s(g('truth_lost',False),'belief')})"
          f" | kept {s(g('truth_kept'),'belief')} (stuff {s(g('truth_kept',True),'belief')} / things {s(g('truth_kept',False),'belief')})"
          f" | hallucinated {s(g('hallucinated'),'belief')}"
          f" || AUROC truth>hal {auroc([x['belief'] for x in g('truth')],[x['belief'] for x in g('hallucinated')]):.3f}"
          f" | negative push: truth {neg(g('truth'))}, stuff truth {neg(g('truth',True))}, hallucinated {neg(g('hallucinated'))}")
