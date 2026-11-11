"""THRONE xl columns (user 11-08): THRONE's own setting, three judges (flan-t5 base, large, xl) at voting threshold 9 (unanimous, 3 judges x 3
question formats), on images 200-999 of throne_full1000, the rows of tab:throne. Below it the paper's two-judge setting (base, large; threshold 6)
from the same code, as a check against the table. ALL (per image) and CLS (class-wise) as in diag_throne_heldout.py; paired image bootstrap vs the
backbone's bench Base, 2000 resamples, seed 0. CPU only; a row without the judges' files is listed as missing.
usage: BENCH=/home/project-b/LJW/bench xl_score.py   (THRONE's venv)"""
import contextlib,glob,io,json,os,sys
from pathlib import Path
import numpy as np
BENCH=Path(os.environ['BENCH']);T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json')
sys.path.insert(0,str(T));import throne_score_aqa as A,pycocotools.coco as coco
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');SL=slice(200,1000)
ROWS={'Qwen2.5-VL':[('Base','throne_full1000','base'),('REVIS (= Base)','throne_full1000','revis'),('RUDDER','throne_full1000','rudder_clone'),
                    ('GrAInS','throne_full1000','grains_clone'),('VTI','throne_full1000','vti_author'),('RBD, first config.','throne_full1000','s16a1bl2p1'),
                    ('RBD','qwen_cf_test_all','cf_a3bl2p1')],
      'Gemma 4 E4B':[('Base','gemma_throne1000','base'),('REVIS','gemma_throne1000','revis'),('RUDDER','gemma_throne1000','rudder_clone'),
                     ('GrAInS','gemma_throne1000','grains_clone'),('VTI','gemma_throne1000','vti'),('RBD, first config.','gemma_throne1000','s24a1bl2p1'),
                     ('RBD','gemma_cf_test_all','cf_a3bl2p1')]}
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()}
keys=[json.loads(l)['metadata']['image_id'] for l in open(O/'sets/throne_full1000.jsonl') if l.strip()];assert len(keys)==1000;keys=keys[SL]
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def mats(d,judges,thr):
    fs=[f for j in judges for f in glob.glob(str(d/f'*flan-t5-{j}.json'))]
    if len(fs)!=len(judges):return None
    pred=A.prepare_aqa_preds(fs,idx);col,dis=A.majority_vote_collapsing(pred,voting_threshold=thr)
    P=np.zeros((len(keys),80));C=np.zeros_like(P);G=np.zeros_like(P)
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
    f1,f5=f(p,r);m1=~np.isnan(f1);m5=~np.isnan(f5);s=W.sum(1)
    return {'P':(W@p)/s,'R':(W@r)/s,'F1':(W[:,m1]@f1[m1])/W[:,m1].sum(1),'F0.5':(W[:,m5]@f5[m5])/W[:,m5].sum(1)}
def cls(W,P,C,G):
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):pr=c_/p_;rc=c_/g_;return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F0.5':np.nanmean(f(pr,rc)[1],1)}
Q=[('ALL','P'),('ALL','R'),('ALL','F1'),('ALL','F0.5'),('CLS','P'),('CLS','R'),('CLS','F0.5')]
n=len(keys);rng=np.random.default_rng(0);W=np.stack([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(2000)]).astype(float);one=np.ones((1,n))
def metr(Wt,m):
    with np.errstate(invalid='ignore'):a,c=micro(Wt,*m),cls(Wt,*m)
    return [100*(a if s=='ALL' else c)[q] for s,q in Q]
for judges,thr,label in ((('base','large','xl'),9,"THRONE's three judges (base, large, xl), threshold 9: the xl columns"),
                         (('base','large'),6,'two judges (base, large), threshold 6: the table as it stands (check of this code)')):
    print(f'##### {label}; images 200-999 (n {n})')
    for bb,rows in ROWS.items():
        M={name:mats(O/'runs'/run/'eval'/tag/'combined',judges,thr) for name,run,tag in rows};miss=[k for k,v in M.items() if v is None]
        print(f'== {bb}'+(f'  (missing the judges\' files: {miss})' if miss else ''))
        print('   '+' '*20+''.join(f'{s+"_"+q:>13s}' for s,q in Q))
        ref=M.get('Base');B0=metr(W,ref) if ref is not None else None
        for name,m in M.items():
            if m is None:continue
            print(f'   {name:20s}'+''.join(f'{x[0]:13.1f}' for x in metr(one,m)))
            if B0 is not None and name!='Base':
                print(f"   {'':20s}"+''.join(f"{'[%+.1f,%+.1f]'%(np.quantile(x-y,.025),np.quantile(x-y,.975)):>13s}" for x,y in zip(metr(W,m),B0)))
    print()
