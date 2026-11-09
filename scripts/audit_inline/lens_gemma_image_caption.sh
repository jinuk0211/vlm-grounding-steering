#!/bin/bash
# Table A.5 upper rows (Gemma) and the caption-position lens summary; wrote gemma-lens-summary.json
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 03:43:38 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/lens_gemma_image_caption.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/logit-lens-20261104
for i in $(seq 1 110); do [ -f $O/logs/gemma-image-done ] && break; sleep 5; done
cat $O/logs/gemma-image-done 2>/dev/null; python3 -c "import json;d=json.load(open('$O/gemma-image-0-500-status.json'));print('gemma image',d['stage'],d.get('done'),d.get('sec_per_image'),d.get('last_layer_max_prob_err'),d.get('objects_without_token'),(d.get('error') or '')[-300:])"
[ -f $O/logs/gemma-image-done ] || exit 0
grep -v -E "^    " $O/logs/gemma-analyze.log | tail -8
OMP_NUM_THREADS=6 .venv/bin/python - <<'P'
import json,torch,numpy as np
from collections import Counter
O='/home/project-b/LJW/vlm-grounding-steering/outputs/logit-lens-20261104'
s=json.load(open(f'{O}/gemma-lens-summary.json'));im=s['image_lens'];print(im['class_counts'])
for t,v in im['tasks'].items():
    m=v['max'];print(t,'pooled',[round(x,2) for x in m['pooled']][::6],'last',round(m['pooled'][-1],3),'within',[None if x!=x else round(x,2) for x in m['within_object']][::6],'best',m['best_layer'],round(m['best_pooled'],3),m['best_ci95'],'objs',m['objects_used_within'])
sp=im['spearman_with_question_margin']['max'];print('spearman',[round(x,2) for x in sp][::6],'max',round(np.nanmax(sp),3),int(np.nanargmax(sp)),'q margin',round(im['question_margin_auroc_hallucinated'],3))
d=torch.load(f'{O}/gemma-image-0-500.pt',weights_only=False);nodes=d['nodes']
def au(s,y):
    s=np.asarray(s,float);y=np.asarray(y,bool);p=y.sum();n=len(y)-p;_,i,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.cumsum(c)-(c-1)/2)[i];return float((r[y].sum()-p*(p+1)/2)/(p*n))
S=[];c=[];nd=[]
for r in d['records']:
    L=r['labels'];GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
    for j,n in enumerate(nodes):S.append(r['max'][:,j].float().numpy());nd.append(n);c.append('hall' if n in H else 'correct' if n in M and n in GT else 'missed' if n in GT else 'absent' if n not in M else 'o')
S=np.stack(S);c=np.array(c);nd=np.array(nd)
for l in (21,26,32,36,40,42):
    x=S[:,l];P=lambda a,b:round(au(x[(c==a)|(c==b)],c[(c==a)|(c==b)]==a),3)
    print(f' L{l} median',{k:float(f'{np.median(x[c==k]):.1e}') for k in ('correct','missed','hall','absent')},'correct>hall',P('correct','hall'),'missed>absent',P('missed','absent'),'hall>absent',P('hall','absent'),'missed>hall',P('missed','hall'))
x=S[:,-1];a=[au(x[((c=='hall')|(c=='absent'))&(nd==n)],c[((c=='hall')|(c=='absent'))&(nd==n)]=='hall') for n in np.unique(nd[c=='hall']) if ((c=='hall')&(nd==n)).sum()>=5];print(' hall vs absent within object, last layer',round(float(np.mean(a)),3),len(a))
r=d['records'][0];print(r['image_id'],r['labels']['gt'],Counter(w.strip()[:9] or '_' for w in r['top1_str'][32]).most_common(14))
P
