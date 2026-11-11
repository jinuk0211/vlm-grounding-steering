"""THRONE part of the AMBER 'why' diagnostic (11-05, CPU only, existing 2-judge files, threshold 6): the old RBD config
s16a1bl2p1 on rows 150-999 of throne_full1000 (its setting was chosen on the first 150 images), vs bench Base, with and
without class orange; on rows 150-499 also vs vdd Base (vdd Base exists for the first 500 only). Micro (ALL) and class-wise (CLS).
Paired image bootstrap 2000, seed 0."""
import contextlib,glob,io,json,os,sys
from pathlib import Path
import numpy as np
BENCH=Path(os.environ['BENCH']);T=BENCH/'THRONE/throne';COCO=str(BENCH/'coco2017/annotations/instances_val2017.json')
sys.path.insert(0,str(T));import throne_score_aqa as A,pycocotools.coco as coco
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()};OR=n2i['orange']
keys=[json.loads(l)['metadata']['image_id'] for l in open(O/'sets/throne_full1000.jsonl') if l.strip()];assert len(keys)==1000
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def mats(d,ks):
    fs=sorted(glob.glob(str(Path(d)/'*.json')));assert len(fs)==2;pred=A.prepare_aqa_preds(fs,idx);col,dis=A.majority_vote_collapsing(pred,voting_threshold=6)
    P=np.zeros((len(ks),80));C=np.zeros((len(ks),80));G=np.zeros((len(ks),80))
    for i,k in enumerate(ks):
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
    f1,f5=f(p,r);m=~np.isnan(f5);s=W.sum(1);ws=W[:,m].sum(1);return {'P':(W@p)/s,'R':(W@r)/s,'F0.5':(W[:,m]@f5[m])/ws}
def cls(W,P,C,G):
    p_,c_,g_=W@P,W@C,W@G
    with np.errstate(invalid='ignore',divide='ignore'):pr=c_/p_;rc=c_/g_
    return {'P':np.nanmean(pr,1),'R':np.nanmean(rc,1),'F0.5':np.nanmean(f(pr,rc)[1],1)}
keep=np.arange(80)!=OR
B=O/'diag/throne_full1000_2judges'
bench=mats(B/'base/combined',keys);ours=mats(B/'s16a1bl2p1/combined',keys);vdd=mats(O/'runs/throne_test500_all/eval/base/combined',keys[:500])
def sub(m,sl):return tuple(x[sl] for x in m)
def val(m,fn,kp=None):
    m=tuple(x[:,kp] for x in m) if kp is not None else m;return fn(np.ones((1,m[0].shape[0])),*m)
def report(label,sl,items,ref):
    n=len(range(*sl.indices(1000)));rng=np.random.default_rng(0);W=np.stack([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(2000)]).astype(float)
    M={k:sub(v,sl) for k,v in items.items()};print(f'\n== {label}: n {n}; reference {ref}')
    print('   row            | ALL: P     R   F0.5  w/o orange | CLS: P     R   F0.5  w/o orange | orange FPs')
    for k,m in M.items():
        a,c_=val(m,micro),val(m,cls);an,cn=val(m,micro,keep),val(m,cls,keep);ofp=int((m[0][:,OR]-m[1][:,OR]).sum())
        print(f'   {k:14s} | {100*a["P"][0]:5.1f} {100*a["R"][0]:5.1f} {100*a["F0.5"][0]:5.1f} {100*an["F0.5"][0]:6.1f}      | {100*c_["P"][0]:5.1f} {100*c_["R"][0]:5.1f} {100*c_["F0.5"][0]:5.1f} {100*cn["F0.5"][0]:6.1f}      | {ofp:3d}')
    ci=lambda x,y:'[%+.1f,%+.1f]'%(100*np.quantile(x-y,.025),100*np.quantile(x-y,.975))
    for k,m in M.items():
        if k==ref:continue
        r0=M[ref];g=lambda m_,fn,kp=None:fn(W,*(tuple(x[:,kp] for x in m_) if kp is not None else m_))
        print(f'   CI {k} vs {ref}: dF0.5_ALL {ci(g(m,micro)["F0.5"],g(r0,micro)["F0.5"])} w/o orange {ci(g(m,micro,keep)["F0.5"],g(r0,micro,keep)["F0.5"])} dP_ALL {ci(g(m,micro)["P"],g(r0,micro)["P"])}'
              f' | dF0.5_CLS {ci(g(m,cls)["F0.5"],g(r0,cls)["F0.5"])} w/o orange {ci(g(m,cls,keep)["F0.5"],g(r0,cls,keep)["F0.5"])}')
print('##### 5b. THRONE, old RBD config s16a1bl2p1: held-out vs the 150 images its setting was chosen on')
report('rows 150-999 (850 images, NOT used to choose the setting)',slice(150,1000),{'bench_base':bench,'rbd_old':ours},'bench_base')
report('rows 0-149 (150 images, used to choose it)',slice(0,150),{'bench_base':bench,'rbd_old':ours},'bench_base')
vdd_pad=tuple(np.vstack([x,np.zeros((500,80))]) for x in vdd)   # vdd Base has the first 500 images; padded so rows 150-499 can be sliced
report('rows 150-499 (350 images; vdd Base exists here)',slice(150,500),{'bench_base':bench,'vdd_base':vdd_pad,'rbd_old':ours},'vdd_base')
