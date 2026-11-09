#!/bin/bash
# Table A.5 upper rows (Qwen): image-token lens AUROC per layer, within-object AUROC, Spearman with the question margin
# Restored from the transcript of the audit session (12a8f468 / dee91680), command run on the server on 2026-11-05 03:42:30 UTC. It was sent inline
# over ssh and never saved as a file on the server; this is the server-side part verbatim (the local ssh wrapper line is removed).
# Its printed output is outputs/audit-recheck-20261105/inline/lens_image_layers_qwen.out.txt. Reads outputs/logit-lens-20261104/*.pt and the other saved
# audit records (not in this branch: activation dumps). The analysis (python heredoc) runs on CPU; lines before it (GPU check, starting
# or waiting for a lens job) are the operational part of the same command, kept as sent.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/logit-lens-20261104
nvidia-smi --id=5 --query-gpu=memory.free,utilization.gpu --format=csv,noheader
F=$(nvidia-smi --id=5 --query-gpu=memory.free --format=csv,noheader,nounits)
if [ $F -ge 25000 ]; then
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts OMP_NUM_THREADS=8"
U=$(nvidia-smi --id=5 --query-gpu=uuid --format=csv,noheader)
nohup bash -c "env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/logit_lens.py image --model gemma --output $O > $O/logs/gemma-image.log 2>&1; env $ENV CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/logit_lens.py analyze --model gemma --output $O > $O/logs/gemma-analyze.log 2>&1; echo GEMMA_IMAGE_DONE > $O/logs/gemma-image-done" > /dev/null 2>&1 &
echo "started $!"
fi
OMP_NUM_THREADS=4 .venv/bin/python - <<'P'
import json,torch,numpy as np
O='/home/project-b/LJW/vlm-grounding-steering/outputs/logit-lens-20261104'
s=json.load(open(f'{O}/qwen-lens-summary.json'));im=s['image_lens'];tx=s['text_lens']
print('objects',im['objects'],'without',im['objects_without_token'],im['class_counts'])
for t,v in im['tasks'].items():
    m=v['max'];print(t,'pooled',[round(x,2) for x in m['pooled']][::4],'last',round(m['pooled'][-1],3),'within',[None if x!=x else round(x,2) for x in m['within_object']][::4],'objs',m['objects_used_within'],'top5 best',v['top5mean']['best_layer'],round(v['top5mean']['best_pooled'],3))
print('spearman',[round(x,2) for x in im['spearman_with_question_margin']['max']][::4])
for k in ('top1','kl'):
    b=tx['by_position'][k];print(k,'bins',tx['bins']);
    for l in (7,14,21,24,26,27,28):print('  L',l,[None if x is None else round(x,3) for x in b[l]])
mt=tx['mentions'];print('mention top1 by layer hall',[round(x,2) for x in mt['top1_rate_by_layer']['hallucinated']][14:],'corr',[round(x,2) for x in mt['top1_rate_by_layer']['correct']][14:])
print('logp hall',[round(x,2) for x in mt['logp_by_layer']['hallucinated']][20:],'corr',[round(x,2) for x in mt['logp_by_layer']['correct']][20:])
g=json.load(open(f'{O}/gemma-lens-summary.json'))['text_lens'];print('== gemma text')
for k in ('top1','kl'):
    b=g['by_position'][k]
    for l in (14,21,31,36,40,42):print(' ',k,'L',l,[None if x is None else round(x,3) for x in b[l]])
mt=g['mentions'];print(' top1 hall',[round(x,2) for x in mt['top1_rate_by_layer']['hallucinated']][28:],'corr',[round(x,2) for x in mt['top1_rate_by_layer']['correct']][28:]);print(' logp hall',[round(x,2) for x in mt['logp_by_layer']['hallucinated']][30:],'corr',[round(x,2) for x in mt['logp_by_layer']['correct']][30:])
# extra, from the saved image-lens records (no model): is a hallucinated object seen in the image tokens more than other absent objects?
d=torch.load(f'{O}/qwen-image-0-500.pt',weights_only=False);nodes=d['nodes'];NL=d['n_layers']
def au(s,y):
    s=np.asarray(s,float);y=np.asarray(y,bool);p=y.sum();n=len(y)-p;_,i,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.cumsum(c)-(c-1)/2)[i];return float((r[y].sum()-p*(p+1)/2)/(p*n))
S=[];cls=[];nd=[];g_=[]
for r in d['records']:
    L=r['labels'];GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
    for j,n in enumerate(nodes):
        S.append(r['max'][:,j].float().numpy());nd.append(n);g_.append(r['image_id'])
        cls.append('hall' if n in H else 'correct' if n in M and n in GT else 'missed' if n in GT else 'absent' if n not in M else 'other')
S=np.stack(S);cls=np.array(cls);nd=np.array(nd);print({c:int((cls==c).sum()) for c in np.unique(cls)})
for l in (14,21,24,26,27,28):
    x=S[:,l];print(f' L{l} median conf',{c:float(f'{np.median(x[cls==c]):.2e}') for c in ('correct','missed','hall','absent')},
      'hall>absent %.3f correct>missed %.3f missed>hall %.3f correct>hall %.3f'%(au(x[(cls=='hall')|(cls=='absent')],cls[(cls=='hall')|(cls=='absent')]=='hall'),au(x[(cls=='correct')|(cls=='missed')],cls[(cls=='correct')|(cls=='missed')]=='correct'),au(x[(cls=='missed')|(cls=='hall')],cls[(cls=='missed')|(cls=='hall')]=='missed'),au(x[(cls=='correct')|(cls=='hall')],cls[(cls=='correct')|(cls=='hall')]=='correct')))
# same objects only: hallucinated vs absent within object
for l in (21,28):
    a=[au(S[((cls=='hall')|(cls=='absent'))&(nd==n),l],cls[((cls=='hall')|(cls=='absent'))&(nd==n)]=='hall') for n in np.unique(nd[cls=='hall']) if ((cls=='hall')&(nd==n)).sum()>=5]
    print(f' L{l} hall vs absent within object: {np.mean(a):.3f} ({len(a)} objects)')
for r in d['records'][:2]:
    for l in (21,28):
        w=r['grid'][1];st=[x.strip()[:7] or '_' for x in r['top1_str'][l]];from collections import Counter;print(' image',r['image_id'],'L',l,'GT',r['labels']['gt'],'top patch words',Counter(st).most_common(14))
P
