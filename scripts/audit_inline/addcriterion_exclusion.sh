#!/bin/bash
# App. A.1: Qwen captions with the spurious token addCriterion (77/500) excluded; gray KL, image share and lens by position
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 03:43:13 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/addcriterion_exclusion.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/logit-lens-20261104
cat $O/logs/gemma-image-done 2>/dev/null; python3 -c "import json;d=json.load(open('$O/gemma-image-0-500-status.json'));print('gemma image',d['stage'],d.get('done'),d.get('sec_per_image'),d.get('last_layer_max_prob_err'),(d.get('error') or '')[-200:])"
OMP_NUM_THREADS=6 .venv/bin/python - <<'P'
import json,glob,torch,numpy as np
R='/home/project-b/LJW/vlm-grounding-steering/outputs'
base=[json.loads(l) for l in open(R+'/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl')]
bad={r['id'] for r in base if 'addCriterion' in r['prediction']};print('qwen base captions with addCriterion:',len(bad),'of',len(base),' mean tokens bad/clean',np.mean([r['generated_tokens'] for r in base if r['id'] in bad]),np.mean([r['generated_tokens'] for r in base if r['id'] not in bad]))
B=[0,16,32,64,96,128,192,256,384,513]
def tab(recs,f):
    v=[[] for _ in B[:-1]]
    for r in recs:
        x=f(r)
        for i in range(len(B)-1):
            s=x[B[i]:B[i+1]]
            if len(s):v[i].append(float(np.mean(s)))
    return [(round(float(np.mean(a)),3),len(a)) if a else None for a in v]
vd=[r for f in sorted(glob.glob(R+'/visual-decay-20261030/qwen-[0-9]*-[0-9]*.pt')) if 'partial' not in f for r in torch.load(f,weights_only=False)]
print('visual decay records',len(vd))
for name,sel in (('all',vd),('clean',[r for r in vd if r['id'] not in bad]),('loop',[r for r in vd if r['id'] in bad])):
    print(' gray KL',name,len(sel),tab(sel,lambda r:r['causal']['gray']['kl'].float().numpy()))
    print(' img share',name,tab(sel,lambda r:(r['attn'][:,1].float()/(1-r['attn'][:,0].float())).mean(0).numpy()))
    print(' final lens real-gray',name,tab(sel,lambda r:(r['lens_real'][-1].float()-r['lens_gray'][-1].float()).numpy()))
t=torch.load(R+'/logit-lens-20261104/qwen-text-0-500.pt',weights_only=False)['records']
for name,sel in (('all',t),('clean',[r for r in t if r['id'] not in bad])):
    print(' text lens top1 L27',name,tab(sel,lambda r:(r['rank'][27]==0).float().numpy()));print(' text lens top1 L28',name,tab(sel,lambda r:(r['rank'][28]==0).float().numpy()))
d=torch.load(R+'/logit-lens-20261104/qwen-image-0-500.pt',weights_only=False);nodes=d['nodes']
def au(s,y):
    s=np.asarray(s,float);y=np.asarray(y,bool);p=y.sum();n=len(y)-p;_,i,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.cumsum(c)-(c-1)/2)[i];return float((r[y].sum()-p*(p+1)/2)/(p*n))
for name,ok in (('all',lambda r:True),('clean',lambda r:r['id'] not in bad)):
    S=[];c=[]
    for r in d['records']:
        if not ok(r):continue
        L=r['labels'];GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
        for j,n in enumerate(nodes):S.append(float(r['max'][-1,j]));c.append('hall' if n in H else 'correct' if n in M and n in GT else 'missed' if n in GT else 'absent' if n not in M else 'o')
    S=np.array(S);c=np.array(c);P=lambda a,b:round(au(S[(c==a)|(c==b)],c[(c==a)|(c==b)]==a),3)
    print(' image lens L28',name,{k:int((c==k).sum()) for k in ('correct','hall','missed')},'correct>hall',P('correct','hall'),'missed>absent',P('missed','absent'),'hall>absent',P('hall','absent'),'missed>hall',P('missed','hall'))
P
