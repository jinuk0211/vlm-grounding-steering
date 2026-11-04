L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R/scripts/rbd2
export BENCH=$L/bench RBD2_SETS=$O/sets PYTHONDONTWRITEBYTECODE=1 RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 CUDA_VISIBLE_DEVICES=
$L/venvs/throne/bin/python - <<'PY'
import glob,sys,contextlib,io
from pathlib import Path
import numpy as np
import score_throne as S
sys.path.insert(0,str(S.T));import throne_score_aqa as A,pycocotools.coco as coco
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(S.COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()}
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
keys=[r['metadata']['image_id'] for r in S.read(O/'sets/throne_full1000.jsonl')]
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def mats(d):
    fs=sorted(glob.glob(str(d/'*.json')));pred=A.prepare_aqa_preds(fs,idx);col,dis=A.majority_vote_collapsing(pred,voting_threshold=6)
    P=np.zeros((1000,80));C=np.zeros((1000,80));G=np.zeros((1000,80))
    for i,k in enumerate(keys):
        dq=set(dis[k])
        for g in gt[k]:
            if g not in dq:G[i,n2i[g]]=1
        for p in col[k]:
            if p not in dq:
                P[i,n2i[p]]=1
                if p in gt[k]:C[i,n2i[p]]=1
    return P,C,G
def f05(p,r):
    with np.errstate(invalid='ignore',divide='ignore'):return 1.25*p*r/(0.25*p+r)
def micro(W,P,C,G):   # per-image official micro F0.5 (NaN skipped), image weights W
    np_,nc,ng=P.sum(1),C.sum(1),G.sum(1);ok=(np_>0)&(ng>0);p=np.where(ok,nc/np.maximum(np_,1),0.);r=np.where(ok,nc/np.maximum(ng,1),0.)
    f=f05(p,r);m=~np.isnan(f);return {'P':(W@p)/W.sum(1),'R':(W@r)/W.sum(1),'F0.5':(W[:,m]@f[m])/W[:,m].sum(1)}
def cls(W,P,C,G):
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):pr=c_/p_;rc=c_/g_;return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F0.5':np.nanmean(f05(pr,rc),1)}
rng=np.random.default_rng(0);W=np.stack([np.bincount(rng.integers(0,1000,1000),minlength=1000) for _ in range(2000)]).astype(float);one=np.ones((1,1000))
keep=np.array([idx[c]!='orange' for c in range(80)])
for name,diag,ours in [('Qwen2.5-VL-7B','throne_full1000_2judges','s16a1bl2p1'),('Gemma-4-E4B','gemma_throne1000_2judges','s24a1bl2p1')]:
    Mb=mats(O/'diag'/diag/'base'/'combined');Mo=mats(O/'diag'/diag/ours/'combined')
    for lab,k in [('all 80 classes (check = official)',np.ones(80,bool)),('class orange removed from predictions and gt (79 classes)',keep)]:
        b=[m[:,k] for m in Mb];o=[m[:,k] for m in Mo]
        for met,fn in [('ALL',micro),('CLS',cls)]:
            vb,vo=fn(one,*b),fn(one,*o);bb,oo=fn(W,*b),fn(W,*o)
            print('%-14s %-58s F0.5_%s Base %5.1f ours %5.1f  dF0.5 [%+.1f,%+.1f]  dP [%+.1f,%+.1f]  (P_%s %5.1f -> %5.1f)'%(name,lab,met,100*vb['F0.5'][0],100*vo['F0.5'][0],
                  100*np.quantile(oo['F0.5']-bb['F0.5'],.025),100*np.quantile(oo['F0.5']-bb['F0.5'],.975),100*np.quantile(oo['P']-bb['P'],.025),100*np.quantile(oo['P']-bb['P'],.975),met,100*vb['P'][0],100*vo['P'][0]))
PY
