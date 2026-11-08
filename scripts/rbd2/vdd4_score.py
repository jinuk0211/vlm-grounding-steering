"""THRONE scoring for the colour-fixed-V_obj run (vdd4): official micro (_ALL) and class-wise (_CLS) metrics on existing
2-judge files (flan-t5 base+large, threshold 6), orange FP count, F0.5_ALL without class orange, paired image bootstrap
(2000 resamples, seed 0) vs the first tag. CPU only.
usage: vdd4_score.py SET TAG=COMBINED_DIR [TAG=COMBINED_DIR ...]      (first = reference; env BENCH, RBD2_SETS)
SET = a file sets/SET.jsonl (rows with metadata.image_id)."""
import glob,sys,contextlib,io,os
from pathlib import Path
import numpy as np
BENCH=Path(os.environ['BENCH']);T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json')
sys.path.insert(0,str(T));import throne_score_aqa as A,pycocotools.coco as coco,json
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()}
keys=[json.loads(l)['metadata']['image_id'] for l in open(Path(os.environ['RBD2_SETS'])/f'{sys.argv[1]}.jsonl') if l.strip()];N=len(keys)
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
OR=n2i['orange']
def mats(d):
    fs=sorted(glob.glob(str(Path(d)/'*.json')));assert len(fs)==2,fs
    pred=A.prepare_aqa_preds(fs,idx);assert set(keys)<=set(pred)
    col,dis=A.majority_vote_collapsing(pred,voting_threshold=6)
    P=np.zeros((N,80));C=np.zeros((N,80));G=np.zeros((N,80))
    for i,k in enumerate(keys):
        dq=set(dis[k])
        for g in gt[k]:
            if g not in dq:G[i,n2i[g]]=1
        for p in col[k]:
            if p not in dq:
                P[i,n2i[p]]=1
                if p in gt[k]:C[i,n2i[p]]=1
    return P,C,G
def f(p,r):
    with np.errstate(invalid='ignore',divide='ignore'):return 2*p*r/(p+r),1.25*p*r/(0.25*p+r)
def micro(W,P,C,G):
    np_,nc,ng=P.sum(1),C.sum(1),G.sum(1);ok=(np_>0)&(ng>0);p=np.where(ok,nc/np.maximum(np_,1),0.);r=np.where(ok,nc/np.maximum(ng,1),0.)
    f1,f5=f(p,r);m=~np.isnan(f5);s=W.sum(1);ws=W[:,m].sum(1)
    return {'P':(W@p)/s,'R':(W@r)/s,'F1':(W[:,m]@f1[m])/ws,'F0.5':(W[:,m]@f5[m])/ws}
def cls(W,P,C,G):
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):pr=c_/p_;rc=c_/g_
    f1,f5=f(pr,rc);return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F1':np.nanmean(f1,1),'F0.5':np.nanmean(f5,1)}
keep=np.arange(80)!=OR
rng=np.random.default_rng(0);W=np.stack([np.bincount(rng.integers(0,N,N),minlength=N) for _ in range(2000)]).astype(float);one=np.ones((1,N))
M={};tags=[]
for a in sys.argv[2:]:t,d=a.split('=',1);M[t]=mats(d);tags.append(t)
def val(t):
    P,C,G=M[t];A_=micro(one,P,C,G);C_=cls(one,P,C,G);nz=micro(one,P[:,keep],C[:,keep],G[:,keep])
    return A_,C_,nz
def boot(t):
    P,C,G=M[t];return micro(W,P,C,G),cls(W,P,C,G),micro(W,P[:,keep],C[:,keep],G[:,keep])
b0=boot(tags[0]);print('n images',N,'| reference',tags[0])
print('%-22s | %s | %s | orangeFP | F0.5_ALL w/o orange'%('tag','ALL  P    R   F1  F0.5','CLS  P    R   F1  F0.5'))
for t in tags:
    A_,C_,nz=val(t);P,C,G=M[t];ofp=int((P[:,OR]-C[:,OR]).sum())
    row='%-22s | %5.1f %5.1f %5.1f %5.1f | %5.1f %5.1f %5.1f %5.1f | %4d | %5.1f'%(t,*(100*A_[q][0] for q in ('P','R','F1','F0.5')),*(100*C_[q][0] for q in ('P','R','F1','F0.5')),ofp,100*nz['F0.5'][0])
    if t!=tags[0]:
        bb=boot(t);ci=lambda x,y:'[%+.1f,%+.1f]'%(100*np.quantile(x-y,.025),100*np.quantile(x-y,.975))
        row+='\n    CI vs %s: dP_ALL %s dR_ALL %s dF0.5_ALL %s | dP_CLS %s dR_CLS %s dF0.5_CLS %s | dF0.5_ALL w/o orange %s'%(tags[0],ci(bb[0]['P'],b0[0]['P']),ci(bb[0]['R'],b0[0]['R']),
            ci(bb[0]['F0.5'],b0[0]['F0.5']),ci(bb[1]['P'],b0[1]['P']),ci(bb[1]['R'],b0[1]['R']),ci(bb[1]['F0.5'],b0[1]['F0.5']),ci(bb[2]['F0.5'],b0[2]['F0.5']))
    print(row)
