"""Denial probes with fold-wise AUROC (mean over the 5 test folds), at the paper's layers, on CPU.
Same CV, PCA and L2 grid as verify_probe_analyze.py; only the AUROC aggregation changes. Reports pooled too."""
import glob,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'scripts')
import verify_probe_analyze as V
R=Path('/home/project-b/LJW/vlm-grounding-steering/outputs');O=R/'verify-probe-20261030';OUT=R/'audit-recheck-20261105';OUT.mkdir(exist_ok=True)
BASE={'qwen':R/'qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl','gemma':R/'gemma-grains-fp16-v2-20261029-1793246507/chair/base.jsonl'}
m=sys.argv[1];L=int(sys.argv[2]);f=glob.glob(str(O/f'{m}-n*-all.pt'))[0]
torch.set_num_threads(10);V.DEV='cpu'
d=torch.load(f,weights_only=False);keep=[i for i,x in enumerate(d['meta']) if x['first_mention']];M=[d['meta'][i] for i in keep]
g=np.array([x['image_id'] for x in M]);Y=np.array([[x['margin']<0,x['kind']=='hallucinated'] for x in M],float)
bad_ids=set()
for l in open(BASE[m]):
    r=json.loads(l)
    if 'addCriterion' in r['prediction']:bad_ids.add(r['metadata']['image_id'])
bad=np.array([x in bad_ids for x in g])
F3=torch.tensor([[x['logp_real'],x['object_order'],x['t']] for x in M]).float()[:,None]
H={'real':d['h_real'][keep][:,L,0].float()[:,None],'gray':d['h_gray'][keep][:,L,0].float()[:,None],'F3':F3}
fo=V.fold_ids(g,5,0);S={}
for k,X in H.items():
    s,_=V.cv(X,Y,g);S[k]=s[0];print(k,'fitted',flush=True)
def foldwise(s,y,mask=None):
    a=[V.auroc(s[(fo==f_)&(mask if mask is not None else True)],y[(fo==f_)&(mask if mask is not None else True)]) for f_ in range(5)];return float(np.nanmean(a))
out={'model':m,'layer':L,'n':len(M),'positives':{'T1':int(Y[:,0].sum()),'T2':int(Y[:,1].sum())}}
rng=np.random.default_rng(0);u=np.unique(g);idx={k:np.where(g==k)[0] for k in u}
for j,t in enumerate(('T1_denied','T2_hallucinated')):
    r={k:{'pooled':V.auroc(S[k][:,j],Y[:,j]),'foldwise':foldwise(S[k][:,j],Y[:,j]),'folds':[V.auroc(S[k][fo==f_,j],Y[fo==f_,j]) for f_ in range(5)],'foldwise_no_loops':foldwise(S[k][:,j],Y[:,j],~bad)} for k in S}
    bs=[]
    for _ in range(500):
        ii=np.concatenate([idx[k] for k in rng.choice(u,len(u))]);a=[];
        for f_ in range(5):
            s=ii[fo[ii]==f_]
            if Y[s,j].sum()>0 and (1-Y[s,j]).sum()>0:a.append(V.auroc(S['real'][s,j],Y[s,j])-V.auroc(S['gray'][s,j],Y[s,j]))
        bs.append(np.mean(a))
    r['real_minus_gray_foldwise']={'diff':r['real']['foldwise']-r['gray']['foldwise'],'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],'no_loops':r['real']['foldwise_no_loops']-r['gray']['foldwise_no_loops']}
    out[t]=r
(OUT/f'{m}-probe-foldwise.json').write_text(json.dumps(out,indent=1));print(json.dumps(out,indent=0)[:1500]);print('done')
