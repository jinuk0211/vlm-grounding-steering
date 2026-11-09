#!/bin/bash
# Table A.5 "Word in top-5 of a patch" rows (last layer and L21) and the caption top-1 / writing-step rates
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 08:45:14 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/lens_word_top5.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
HF_HOME=$R/../.cache/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=scripts OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= .venv/bin/python - <<'P'
import torch,numpy as np
from collections import Counter
from transformers import AutoTokenizer
from score_vti_alpha_sweep import chair_scorer
from logit_lens import object_tokens,object_class,CLASSES
O='/home/project-b/LJW/vlm-grounding-steering/outputs/logit-lens-20261104'
sc=chair_scorer();rng=np.random.default_rng(0)
def ci(fn,g,n=500):
    u=np.unique(g);idx={k:np.where(g==k)[0] for k in u};v=[]
    for _ in range(n):
        x=fn(np.concatenate([idx[k] for k in rng.choice(u,len(u))]))
        if x==x:v.append(x)
    return round(100*np.quantile(v,.025),1),round(100*np.quantile(v,.975),1)
def wd(x,c,nd,a,b,m=5):
    d=[x[(c==a)&(nd==k)].mean()-x[(c==b)&(nd==k)].mean() for k in np.unique(nd) if ((c==a)&(nd==k)).sum()>=m and ((c==b)&(nd==k)).sum()>=m]
    return (round(100*float(np.mean(d)),1) if d else None),len(d)
for m,ck,rev in (('qwen','Qwen/Qwen2.5-VL-7B-Instruct','cc594898137f460bfe9f0759e9844b3ce807cfb5'),('gemma','google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2')):
    tk=AutoTokenizer.from_pretrained(ck,revision=rev,local_files_only=True);allnodes,ids=object_tokens(tk,sc)
    d=torch.load(f'{O}/{m}-image-0-500.pt',weights_only=False);nodes=d['nodes'];assert all(ids[n] for n in nodes);depths=d['depths'];print(f'===== {m} depths {depths}')
    own={t:n for n in nodes for t in ids[n]}
    rows={l:{1:[],5:[]} for l in depths};cl=[];nd=[];g=[]
    for r in d['records']:
        for n in nodes:cl.append(object_class(n,r['labels']) or 'other');nd.append(n);g.append(r['image_id'])
        for l in depths:
            top=r['topk'][l].numpy()
            for k in (1,5):
                c_=Counter()
                for row in top[:,:k]:
                    for o in {own[t] for t in row.tolist() if t in own}:c_[o]+=1
                rows[l][k].append([c_[n] for n in nodes])
    cl=np.array(cl);nd=np.array(nd);g=np.array(g);print(' rows',{k:int((cl==k).sum()) for k in CLASSES})
    for k in (1,5):
        print(f' IMAGE TOKENS: object word in top-{k} of at least one patch')
        for l in depths:
            x=np.array(rows[l][k]).ravel();h=(x>0).astype(float)
            pairs=[('correct','hallucinated'),('missed','absent'),('hallucinated','absent'),('missed','hallucinated')]
            def dif(a,b):
                return lambda ii:h[ii][cl[ii]==a].mean()-h[ii][cl[ii]==b].mean()
            print(f'  L{l:2d} '+' '.join(f'{c_[:4]} {100*h[cl==c_].mean():5.1f}%' for c_ in CLASSES)+' | patches if found '+' '.join(f'{c_[:4]} {x[(cl==c_)&(x>0)].mean() if ((cl==c_)&(x>0)).any() else 0:.1f}' for c_ in CLASSES))
            print('      '+'; '.join(f'{a[:4]}-{b[:4]} {100*(h[cl==a].mean()-h[cl==b].mean()):+.1f} {ci(dif(a,b),g)} within {wd(h,cl,nd,a,b)}' for a,b in pairs))
    fp=np.array(rows[depths[-1]][5]).ravel()>0
    print(' most-often detected while absent, last layer top-5:',[(n,round(100*fp[(cl=="absent")&(nd==n)].mean())) for n,_ in Counter(nd[(cl=="absent")&fp]).most_common(8)])
    T=torch.load(f'{O}/{m}-text-0-500.pt',weights_only=False)['records'];lab={r['image_id']:r['labels'] for r in d['records']};NL=d['n_layers']
    show=sorted({round(f*NL) for f in (.25,.5,.75,.85,.9,.95,1.0)});R_=[]
    wr=[]
    for r in T:
        t1=r['top1'].numpy();rk=r['rank'].numpy();first={x['node']:x['t'] for x in reversed(r['mentions']) if x['first']}
        for n in nodes:
            c_=object_class(n,lab[r['image_id']])
            if c_ is None:continue
            end=first.get(n,r['T']);R_.append((c_,n,r['image_id'],np.isin(t1[:,:end],ids[n]).any(1) if end>0 else np.zeros(NL+1,bool)))
        for x in r['mentions']:
            if x['first'] and x['node'] in nodes:wr.append((x['kind'],x['node'],r['image_id'],rk[:,x['t']]))
    cc=np.array([x[0] for x in R_]);nn=np.array([x[1] for x in R_]);gg=np.array([x[2] for x in R_]);H=np.stack([x[3] for x in R_]).astype(float)
    print(' CAPTION: object word is lens top-1 at some position (correct/hall: before first mention; missed/absent: whole caption)')
    for l in show:print(f'  L{l:2d} '+' '.join(f'{c_[:4]} {100*H[cc==c_,l].mean():5.1f}%' for c_ in CLASSES)+f' | missed-absent within {wd(H[:,l],cc,nn,"missed","absent")}')
    k_=np.array([x[0] for x in wr]);rk=np.stack([x[3] for x in wr])
    print(' CAPTION: at the moment of writing, written object in top-k of the layer (first mention)')
    for l in show:print(f'  L{l:2d} '+' '.join(f'top{t} hall {100*(rk[k_=="hallucinated",l]<t).mean():5.1f}% corr {100*(rk[k_=="correct",l]<t).mean():5.1f}%' for t in (1,5,10)))
P
