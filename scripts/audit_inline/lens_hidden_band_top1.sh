#!/bin/bash
# App. A.8 "Caption positions" support: hidden-layer band (Qwen L22-27, Gemma L30-41) top-1 object words by position
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 09:35:03 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/lens_hidden_band_top1.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
HF_HOME=$R/../.cache/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=scripts OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= .venv/bin/python - <<'P'
import glob,torch,numpy as np
from collections import Counter
from transformers import AutoTokenizer
from score_vti_alpha_sweep import chair_scorer
from logit_lens import object_tokens
O='outputs/logit-lens-20261104';sc=chair_scorer();rng=np.random.default_rng(0)
B=[0,32,64,96,128,192,256,384,513];lab_b=[f'{a}-{b-1}' for a,b in zip(B,B[1:])]
for m,ck,rev,band in (('qwen','Qwen/Qwen2.5-VL-7B-Instruct','cc594898137f460bfe9f0759e9844b3ce807cfb5',range(22,28)),('gemma','google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2',range(30,42))):
    tk=AutoTokenizer.from_pretrained(ck,revision=rev,local_files_only=True);_,ids=object_tokens(tk,sc);own={t:n for n,v in ids.items() for t in v}
    caps={r['id']:r['caption_ids'] for f in sorted(glob.glob(f'outputs/visual-decay-20261030/{m}-[0-9]*-[0-9]*.pt')) if 'partial' not in f for r in torch.load(f,weights_only=False)}
    T=torch.load(f'{O}/{m}-text-0-500.pt',weights_only=False)['records'];D=torch.load(f'{O}/{m}-image-0-500.pt',weights_only=False);lab={r['image_id']:r['labels'] for r in D['records']}
    per=[];words=Counter();wh=Counter()
    for r in T:
        c=caps[r['id']];assert len(c)==r['T'];L=lab[r['image_id']];GT=set(L['gt']);H=set(L['hallucinated']);t1=r['top1'][list(band)].numpy()
        first={x['node']:x['t'] for x in reversed(r['mentions']) if x['first']}
        v=np.zeros((r['T'],5))  # any band layer at this position: nonGT, hall, hallBefore, GT, position counted
        for t in range(r['T']):
            prev=c[t-1] if t>0 else -1;seen=set()
            for w in t1[:,t].tolist():
                n=own.get(w)
                if n is None or w==prev or w==c[t]:continue  # echo of the input token / the token actually written
                seen.add((n,w))
            for n,w in seen:
                if n in GT:v[t,3]=1
                else:
                    v[t,0]=1;words[n]+=1
                    if n in H:
                        v[t,1]=1;wh[n]+=1
                        if t<=first.get(n,10**6):v[t,2]=1
        per.append((r['image_id'],r['T'],v))
    print(f'===== {m}: hidden layers L{band[0]}-L{band[-1]}, lens top-1 at any of them. Excludes the input token echo and the token written at that position. Per 100 positions.')
    print(' bin        captions | non-GT object word | later/earlier-hallucinated object | same, before it is written | GT object word | non-GT share')
    def rate(rows,a,b,k):
        x=[v[a:b,k].mean() for _,_T,v in rows if a<_T]
        return x
    for i,(a,b) in enumerate(zip(B,B[1:])):
        n=[r for r in per if r[1]>a]
        if len(n)<5:continue
        M=[100*np.mean(rate(n,a,b,k)) for k in (0,1,2,3)];print(f' {lab_b[i]:9s} {len(n):4d}     | {M[0]:6.2f} | {M[1]:6.2f} | {M[2]:6.2f} | {M[3]:6.2f} | {100*M[0]/max(M[0]+M[3],1e-9):5.1f}%')
    # within-caption paired: same captions, early vs late
    cut_e,cut_l=64,128;rows=[r for r in per if r[1]>=192]
    for k,nm in ((0,'non-GT word'),(1,'hallucinated object'),(3,'GT word')):
        d=np.array([100*(v[cut_l:,k].mean()-v[:cut_e,k].mean()) for _,_T,v in rows]);bs=d[rng.integers(0,len(d),(2000,len(d)))].mean(1);print(f'  same captions (T>=192, n={len(d)}), token 128+ minus token 0-63, {nm}: {d.mean():+.2f} [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}]')
    # by caption fraction, to remove length effects
    fr=[[],[],[],[]]
    for _,_T,v in per:
        for q in range(4):
            fr[q].append(v[int(q*_T/4):int((q+1)*_T/4),:4].mean(0)*100)
    print('  by caption quarter (non-GT / hallucinated / GT): '+' | '.join(f'Q{q+1} '+' '.join(f'{x:.2f}' for x in np.mean(fr[q],0)[[0,1,3]]) for q in range(4)))
    print('  top non-GT words',words.most_common(8));print('  top hallucinated-object words',wh.most_common(6))
P
