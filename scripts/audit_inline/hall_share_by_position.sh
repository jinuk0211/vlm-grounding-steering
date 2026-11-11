#!/bin/bash
# Table 2 / App. A.6 support: share of hallucinated first mentions by caption position, t>=128 minus t<128, cohorts
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 09:11:01 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/hall_share_by_position.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= .venv/bin/python - <<'P'
import json,torch,numpy as np
O='outputs/logit-lens-20261104';B=[0,32,64,96,128,192,256,384,513];rng=np.random.default_rng(0)
base=[json.loads(l) for l in open('outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl')]
bad={r['id'] for r in base if 'addCriterion' in r['prediction']}
def ci(fn,g,n=1000):
    u=np.unique(g);idx={k:np.where(g==k)[0] for k in u};v=[]
    for _ in range(n):
        x=fn(np.concatenate([idx[k] for k in rng.choice(u,len(u))]))
        if x==x:v.append(x)
    return [round(float(np.quantile(v,.025)),3),round(float(np.quantile(v,.975)),3)]
def share(recs,first,lab):
    rows=[(r['image_id'],x['t'],x['kind']=='hallucinated') for r in recs for x in r['mentions'] if (x['first'] or not first)]
    g=np.array([a for a,_,_ in rows]);t=np.array([b for _,b,_ in rows]);h=np.array([c for _,_,c in rows])
    out=[]
    for a,b in zip(B,B[1:]):
        s=(t>=a)&(t<b)
        if s.sum()>=10:out.append(f'{a}-{b-1}: {100*h[s].mean():.1f}% (n={int(s.sum())})')
    # early vs late, same cut for all
    d=lambda ii:h[ii][t[ii]>=128].mean()-h[ii][t[ii]<128].mean()
    print(f'  {lab}: '+' | '.join(out));print(f'     t>=128 minus t<128: {100*(h[t>=128].mean()-h[t<128].mean()):+.1f} pts CI {[round(100*x,1) for x in ci(d,g)]}')
for m in ('qwen','gemma'):
    T=torch.load(f'{O}/{m}-text-0-500.pt',weights_only=False)['records'];print('=====',m,len(T),'captions')
    print(' share of object mentions that are hallucinated, by token position')
    share(T,True,'first mention of each object');share(T,False,'all mentions')
    if m=='qwen':share([r for r in T if r['id'] not in bad],True,'first mention, addCriterion captions excluded')
    # fixed cohort: captions reaching 192 tokens, so short captions do not drop out
    C=[r for r in T if r['T']>=192];print(f'  fixed cohort T>=192: {len(C)} captions');share(C,True,'first mention, cohort')
    # density: new hallucinated objects per 100 tokens of caption written, by position
    den=[];cor=[]
    for a,b in zip(B,B[1:]):
        tok=sum(max(0,min(r['T'],b)-a) for r in T);nh=sum(1 for r in T for x in r['mentions'] if x['first'] and a<=x['t']<b and x['kind']=='hallucinated');nc=sum(1 for r in T for x in r['mentions'] if x['first'] and a<=x['t']<b and x['kind']!='hallucinated')
        if tok>500:den.append(f'{a}-{b-1}: H {100*nh/tok:.2f} C {100*nc/tok:.2f}')
    print('  new objects per 100 tokens  '+' | '.join(den))
    # within caption: is the hallucinated object written later than the correct ones of the same caption?
    w=[];g=[]
    for r in T:
        f=[x for x in r['mentions'] if x['first']];h=[x['t']/r['T'] for x in f if x['kind']=='hallucinated'];c=[x['t']/r['T'] for x in f if x['kind']!='hallucinated']
        if h and c:w.append(np.mean(h)-np.mean(c))
    w=np.array(w);idx=rng.integers(0,len(w),(2000,len(w)));b=w[idx].mean(1);print(f'  same caption, relative position hallucinated minus correct: {w.mean():+.3f} CI [{np.quantile(b,.025):+.3f},{np.quantile(b,.975):+.3f}] ({len(w)} captions), later in {100*(w>0).mean():.0f}%')
    # order among distinct objects
    o={}
    for r in T:
        f=sorted([x for x in r['mentions'] if x['first']],key=lambda x:x['t'])
        for i,x in enumerate(f):o.setdefault(min(i,7),[]).append(x['kind']=='hallucinated')
    print('  by object order (1st, 2nd, ...): '+' '.join(f'{k+1}:{100*np.mean(v):.0f}%(n{len(v)})' for k,v in sorted(o.items())))
    # last-30% vs caption-start with captions having hall anywhere
    print('  captions with any hallucination by length: '+' | '.join(f'T{a}-{b-1}: {100*np.mean([any(x["kind"]=="hallucinated" for x in r["mentions"]) for r in T if a<=r["T"]<b]):.0f}% (n={sum(1 for r in T if a<=r["T"]<b)})' for a,b in ((0,128),(128,192),(192,256),(256,384),(384,513))))
P
