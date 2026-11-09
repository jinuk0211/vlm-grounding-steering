#!/bin/bash
# App. A.8 "Writing step": written word in the lens top-5 at the writing step, position-matched gaps (Qwen L26/L27, Gemma L34/L38)
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 09:13:54 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/lens_writing_step.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
date '+%m-%d %T'; nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | sed -n 3,6p; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | while IFS=, read u p m; do i=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$u" | cut -d, -f1); echo "gpu$i $m $(ps -o args= -p $p | cut -c1-70)"; done | grep -E "^gpu[2345] "; ps -eo args | grep -E "scripts/rbd2/.*\.sh" | grep -v grep | cut -c1-70 | sort | uniq -c
HF_HOME=$R/../.cache/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=scripts OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= .venv/bin/python - <<'P'
import json,torch,numpy as np
from transformers import AutoTokenizer
from score_vti_alpha_sweep import chair_scorer
from logit_lens import object_tokens,object_class
O='outputs/logit-lens-20261104';sc=chair_scorer();rng=np.random.default_rng(0)
B=[0,64,128,513]
for m,ck,rev in (('qwen','Qwen/Qwen2.5-VL-7B-Instruct','cc594898137f460bfe9f0759e9844b3ce807cfb5'),('gemma','google/gemma-4-E4B-it','ee0ef6023621cff504d758262d4e04895a5af4a2')):
    tk=AutoTokenizer.from_pretrained(ck,revision=rev,local_files_only=True);allnodes,ids=object_tokens(tk,sc);own={t:n for n,v in ids.items() for t in v}
    T=torch.load(f'{O}/{m}-text-0-500.pt',weights_only=False)['records'];D=torch.load(f'{O}/{m}-image-0-500.pt',weights_only=False);lab={r['image_id']:r['labels'] for r in D['records']};NL=D['n_layers']
    rk=[];hl=[];tt=[];gg=[]
    for r in T:
        for x in r['mentions']:
            if x['first']:rk.append(r['rank'][:,x['t']].numpy());hl.append(x['kind']=='hallucinated');tt.append(x['t']);gg.append(r['image_id'])
    rk=np.stack(rk);hl=np.array(hl);tt=np.array(tt);gg=np.array(gg);u=np.unique(gg);idx={k:np.where(gg==k)[0] for k in u};bins=np.digitize(tt,[32,64,96,128,192,256])
    def strat(v,ii):
        d=[];w=[]
        for b in np.unique(bins[ii]):
            s=ii[bins[ii]==b];a=v[s[hl[s]]];c=v[s[~hl[s]]]
            if len(a)>=3 and len(c)>=3:d.append(a.mean()-c.mean());w.append(len(a))
        return float(np.average(d,weights=w)) if d else float('nan')
    print(f'===== {m}: written object at the query that writes it, first mention, hall {hl.sum()} correct {(~hl).sum()}')
    print(' L | top1 hall corr | top5 hall corr | position-matched top5 gap [CI] | top5 by position hall/corr: <64, 64-127, 128+')
    show=[l for l in range(NL+1) if (rk[:,:][:,l]<5).mean()>=0.002]+[]
    for l in sorted(set(show)):
        v=(rk[:,l]<5).astype(float);gap=strat(v,np.arange(len(v)));bs=[strat(v,np.concatenate([idx[k] for k in rng.choice(u,len(u))])) for _ in range(300)]
        pos=' '.join(f'{100*v[hl&(tt>=a)&(tt<b)].mean():.0f}/{100*v[(~hl)&(tt>=a)&(tt<b)].mean():.0f}' for a,b in zip(B,B[1:]))
        print(f' L{l:2d} | {100*(rk[hl,l]==0).mean():5.1f} {100*(rk[~hl,l]==0).mean():5.1f} | {100*v[hl].mean():5.1f} {100*v[~hl].mean():5.1f} | {100*gap:+5.1f} [{100*np.nanquantile(bs,.025):+.1f},{100*np.nanquantile(bs,.975):+.1f}] | {pos}')
    # which object words surface as lens top-1 at each layer, over all caption positions, counting only tokens that were NOT the one written there
    cnt=np.zeros((NL+1,4));allc=np.zeros((NL+1,4));cl=['correct','hallucinated','missed','absent'];tok=0
    capt={}
    for r in T:
        t1=r['top1'].numpy();L=lab[r['image_id']];tok+=r['T'];ids_w=None
        rk0=r['rank'].numpy()
        for l in range(NL+1):
            for p in np.where(np.isin(t1[l],list(own)))[0]:
                c=object_class(own[int(t1[l,p])],L)
                if c is None:continue
                allc[l,cl.index(c)]+=1
                if rk0[l,p]!=0:cnt[l,cl.index(c)]+=1
    print(' object words surfacing as lens top-1 while NOT the written token, per 1000 caption tokens (correct / hall / missed / absent), share hall among non-absent')
    for l in range(NL+1):
        if cnt[l].sum()/tok*1000>=0.5:
            s=cnt[l];print(f' L{l:2d} {1000*s[0]/tok:6.2f} {1000*s[1]/tok:6.2f} {1000*s[2]/tok:6.2f} {1000*s[3]/tok:6.2f} | hall share {100*s[1]/max(s[:3].sum(),1):4.1f}%')
    s=json.load(open(f'{O}/{m}-lens-summary.json'))['image_lens']['tasks'];P=s['hallucinated']['max']['pooled'];Q=s['presence']['max']['pooled'];X=s['missed']['max']['pooled']
    print(' image-token max-prob AUROC per layer (correct vs hall | present vs absent | missed vs absent):');print('  '+' '.join(f'L{l}:{P[l]:.2f}|{Q[l]:.2f}|{X[l]:.2f}' for l in range(0,NL+1,1 if NL<30 else 2)))
P
