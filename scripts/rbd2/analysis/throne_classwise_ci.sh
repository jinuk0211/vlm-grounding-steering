L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R/scripts/rbd2
export BENCH=$L/bench RBD2_SETS=$O/sets PYTHONDONTWRITEBYTECODE=1 RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 CUDA_VISIBLE_DEVICES=
$L/venvs/throne/bin/python - <<'PY'
import glob,json,sys,contextlib,io
from pathlib import Path
import numpy as np
import score_throne as S
sys.path.insert(0,str(S.T));import throne_score_aqa as A,pycocotools.coco as coco
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(S.COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()}
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
keys=[r['metadata']['image_id'] for r in S.read(O/'sets/throne_full1000.jsonl')]
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def mats(files,thr):
    pred=A.prepare_aqa_preds(files,idx);assert set(pred)==set(keys)
    col,dis=A.majority_vote_collapsing(pred,voting_threshold=thr);P=np.zeros((1000,80));C=np.zeros((1000,80));G=np.zeros((1000,80))
    for i,k in enumerate(keys):
        dq=set(dis[k])
        for g in gt[k]:
            if g not in dq:G[i,n2i[g]]+=1
        for p in col[k]:
            if p not in dq:
                P[i,n2i[p]]+=1
                if p in gt[k]:C[i,n2i[p]]+=1
    return P,C,G
def cls(W,P,C,G):   # W: (b,1000) image weights -> official classwise P, R, F1, F0.5 (nanmean over classes)
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):
        pr=c_/p_;rc=c_/g_;f1=2*pr*rc/(pr+rc);f05=1.25*pr*rc/(0.25*pr+rc)
        return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F1':np.nanmean(f1,1),'F0.5':np.nanmean(f05,1)}
rng=np.random.default_rng(0);B=[rng.integers(0,1000,1000) for _ in range(2000)];W=np.stack([np.bincount(b,minlength=1000) for b in B]).astype(float)
one=np.ones((1,1000))
for name,run,diag,tags,thr,kind in [('Qwen2.5-VL-7B','throne_full1000','throne_full1000_2judges',['base','s16a1bl2p1','revis','rudder_clone','grains_clone','vti_author'],6,'2-judge'),
                                    ('Gemma-4-E4B','gemma_throne1000','gemma_throne1000_2judges',['base','s24a1bl2p1','revis','rudder_clone','grains_clone','vti'],6,'2-judge'),
                                    ('Qwen2.5-VL-7B','throne_full1000',None,['base','revis','rudder_clone','grains_clone','vti_author'],9,'3-judge xl')]:
    print(f'##### CLASS-WISE ({kind}, threshold {thr}) {name}; CI = paired image bootstrap vs bench Base, 2000 resamples, seed 0, 95% percentile')
    M={}
    for t in tags:
        d=O/'diag'/diag/t/'combined' if diag else O/'runs'/run/'eval'/t/'combined'
        fs=sorted(glob.glob(str(d/'*.json')));assert len(fs)==(2 if diag else 3),fs
        M[t]=mats(fs,thr)
    base=cls(W,*M['base'])
    for t in tags:
        v=cls(one,*M[t]);line=f"{t:13s} P_CLS {100*v['P'][0]:5.1f} R_CLS {100*v['R'][0]:5.1f} F1_CLS {100*v['F1'][0]:5.1f} F0.5_CLS {100*v['F0.5'][0]:5.1f}"
        if t!='base':
            b=cls(W,*M[t]);line+='  '+' '.join(f"d{q}_CLS[{100*np.quantile(b[q]-base[q],.025):+.1f},{100*np.quantile(b[q]-base[q],.975):+.1f}]" for q in ('P','R','F0.5'))
        print(line)
PY
