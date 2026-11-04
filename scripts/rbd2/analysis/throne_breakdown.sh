L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R/scripts/rbd2
export BENCH=$L/bench RBD2_SETS=$O/sets PYTHONDONTWRITEBYTECODE=1 RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 CUDA_VISIBLE_DEVICES=
$L/venvs/throne/bin/python - <<'PY'
import glob,sys,contextlib,io
from pathlib import Path
import numpy as np
import score_throne as S
sys.path.insert(0,str(S.T));import throne_score_aqa as A,pycocotools.coco as coco
with contextlib.redirect_stdout(io.StringIO()):ds=coco.COCO(S.COCO)
cats=sorted(ds.cats.values(),key=lambda x:x['id']);idx={i:c['name'] for i,c in enumerate(cats)};cid={c['id']:c['name'] for c in cats};n2i={v:k for k,v in idx.items()};names=[idx[i] for i in range(80)]
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
keys=[r['metadata']['image_id'] for r in S.read(O/'sets/throne_full1000.jsonl')]
gt={k:set(cid[a['category_id']] for a in ds.loadAnns(ds.getAnnIds(imgIds=[k]))) for k in keys}
def mats(d):
    fs=sorted(glob.glob(str(d/'*.json')));assert len(fs)==2
    pred=A.prepare_aqa_preds(fs,idx);col,dis=A.majority_vote_collapsing(pred,voting_threshold=6)
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
def per_img(P,C,G):   # official micro: P_i=R_i=0 if the pred or gt set is empty
    np_,nc,ng=P.sum(1),C.sum(1),G.sum(1);ok=(np_>0)&(ng>0)
    p=np.where(ok,nc/np.maximum(np_,1),0.);r=np.where(ok,nc/np.maximum(ng,1),0.);return p,r,np_
def f05(p,r):
    with np.errstate(invalid='ignore',divide='ignore'):return 1.25*p*r/(0.25*p+r)
def cls(P,C,G):
    p_,c_,g_=P.sum(0),C.sum(0),G.sum(0)
    with np.errstate(invalid='ignore',divide='ignore'):pc=c_/p_;rc=c_/g_
    return p_,c_,g_,pc,rc,f05(pc,rc)
fmt=lambda x:'  nan' if np.isnan(x) else '%5.1f'%(100*x)
rng=np.random.default_rng(0);B=[rng.integers(0,1000,1000) for _ in range(2000)]
for name,diag,ours in [('Qwen2.5-VL-7B','throne_full1000_2judges','s16a1bl2p1'),('Gemma-4-E4B','gemma_throne1000_2judges','s24a1bl2p1')]:
    Pb,Cb,Gb=mats(O/'diag'/diag/'base'/'combined');Po,Co,Go=mats(O/'diag'/diag/ours/'combined')
    print('\n################ %s: bench Base vs ours (%s), 2 judges, threshold 6 ################'%(name,ours))
    pb,cb,gb,Pcb,Rcb,Fb=cls(Pb,Cb,Gb);po,co,go,Pco,Rco,Fo=cls(Po,Co,Go)
    print('[1] PER CLASS. Counts are images. P_c=correct/predicted, R_c=correct/gt, F0.5_c from them; NaN if predicted=0 (P_c undefined) or gt=0, or correct=0 (P_c=R_c=0). Sorted by Base gt count.')
    print('%-16s | Base: pred corr   gt   P_c   R_c F0.5_c | ours: pred corr   gt   P_c   R_c F0.5_c | dF0.5_c'%'class')
    for c in np.argsort(-gb,kind='stable'):
        dF=(Fo[c]-Fb[c])*100
        print('%-16s |      %4d %4d %4d %s %s  %s |      %4d %4d %4d %s %s  %s | %s'%(names[c],pb[c],cb[c],gb[c],fmt(Pcb[c]),fmt(Rcb[c]),fmt(Fb[c]),po[c],co[c],go[c],fmt(Pco[c]),fmt(Rco[c]),fmt(Fo[c]),'  nan' if np.isnan(dF) else '%+5.1f'%dF))
    nb,no=np.isnan(Fb),np.isnan(Fo)
    print('F0.5_CLS = nanmean over classes: Base %.1f over %d classes, ours %.1f over %d classes'%(100*np.nanmean(Fb),(~nb).sum(),100*np.nanmean(Fo),(~no).sum()))
    print('classes never predicted: Base %d %s'%((pb==0).sum(),[names[c] for c in np.where(pb==0)[0]]))
    print('                         ours %d %s'%((po==0).sum(),[names[c] for c in np.where(po==0)[0]]))
    print('classes with F0.5_c NaN (left out of the mean): Base %d, ours %d; in the mean for ours only: %s; for Base only: %s'%(nb.sum(),no.sum(),
          [(names[c],round(100*Fo[c],1),int(po[c])) for c in np.where(nb&~no)[0]],[(names[c],round(100*Fb[c],1),int(pb[c])) for c in np.where(~nb&no)[0]]))
    both=~nb&~no;d=(Fo-Fb)*100
    print('over the %d classes in both means: mean dF0.5_c %+.2f; F0.5_c up by >5: %d, down by >5: %d, within 5: %d'%(both.sum(),np.mean(d[both]),(d[both]>5).sum(),(d[both]<-5).sum(),(abs(d[both])<=5).sum()))
    print('   up >5   (class, dF0.5_c, Base pred, ours pred, gt): %s'%[(names[c],round(d[c],1),int(pb[c]),int(po[c]),int(gb[c])) for c in np.where(both&(d>5))[0]])
    print('   down >5 (class, dF0.5_c, Base pred, ours pred, gt): %s'%[(names[c],round(d[c],1),int(pb[c]),int(po[c]),int(gb[c])) for c in np.where(both&(d<-5))[0]])
    w=gb/gb.sum();print('gt-weighted mean of F0.5_c over the classes in both: Base %.1f, ours %.1f'%(100*np.sum(Fb[both]*w[both])/w[both].sum(),100*np.sum(Fo[both]*w[both])/w[both].sum()))
    fpb,fpo=pb-cb,po-co;dfp=fpo-fpb;rem=-dfp[dfp<0].sum()
    print('\n[2] FALSE POSITIVES (image-class pairs predicted, not disqualified, not in gt): Base %d, ours %d, net %+d; removed (classes with fewer FPs) %d, added %d'%(fpb.sum(),fpo.sum(),dfp.sum(),rem,dfp[dfp>0].sum()))
    o=np.argsort(dfp,kind='stable')
    print('   top 10 FP removed (Base->ours): '+', '.join('%s %d->%d'%(names[c],fpb[c],fpo[c]) for c in o[:10] if dfp[c]<0))
    print('   top 10 FP added   (Base->ours): '+', '.join('%s %d->%d'%(names[c],fpb[c],fpo[c]) for c in o[::-1][:10] if dfp[c]>0))
    top5=np.argsort(-gb,kind='stable')[:5];r5=-dfp[top5][dfp[top5]<0].sum()
    print('   5 most frequent classes (Base gt count): %s; FPs removed there %d = %.0f%% of all removed; net FP change there %+d'%([names[c] for c in top5],r5,100*r5/max(rem,1),dfp[top5].sum()))
    fnb,fno=gb-cb,go-co;print('   (false negatives, gt not predicted: Base %d, ours %d; correct: Base %d, ours %d)'%(fnb.sum(),fno.sum(),cb.sum(),co.sum()))
    p_b,r_b,npb=per_img(Pb,Cb,Gb);p_o,r_o,npo=per_img(Po,Co,Go);dp,dr=p_o-p_b,r_o-r_b
    print('\n[3] PER IMAGE (official micro P_i, R_i; both 0 if the pred or gt set is empty)')
    print('   P_i: higher %d, lower %d, equal %d;   R_i: higher %d, lower %d, equal %d'%((dp>1e-9).sum(),(dp<-1e-9).sum(),(abs(dp)<=1e-9).sum(),(dr>1e-9).sum(),(dr<-1e-9).sum(),(abs(dr)<=1e-9).sum()))
    print('   predicted objects per image: Base mean %.2f, ours %.2f'%(npb.mean(),npo.mean()))
    for lab,m in [('0',npb==0),('1',npb==1),('2',npb==2),('3-4',(npb>=3)&(npb<=4)),('5+',npb>=5)]:
        if m.sum():print('   Base caption has %-3s predicted objects: n %4d  mean dP_i %+5.1f  mean dR_i %+5.1f  mean (ours-Base) #pred %+.2f'%(lab,m.sum(),100*dp[m].mean(),100*dr[m].mean(),(npo-npb)[m].mean()))
    fbn,fon=f05(p_b,r_b),f05(p_o,r_o);fb0,fo0=np.nan_to_num(fbn,nan=0.),np.nan_to_num(fon,nan=0.)
    d0=np.array([fo0[k].mean()-fb0[k].mean() for k in B]);dn=np.array([np.nanmean(fon[k])-np.nanmean(fbn[k]) for k in B])
    print('\n[4] F0.5_ALL with P=R=0 images counted as 0: Base %.1f, ours %.1f, dF0.5 [%+.1f,%+.1f]'%(100*fb0.mean(),100*fo0.mean(),100*np.quantile(d0,.025),100*np.quantile(d0,.975)))
    print('    official (P=R=0 images skipped): Base %.1f, ours %.1f, dF0.5 [%+.1f,%+.1f]; P=R=0 images Base %d, ours %d'%(100*np.nanmean(fbn),100*np.nanmean(fon),100*np.quantile(dn,.025),100*np.quantile(dn,.975),np.isnan(fbn).sum(),np.isnan(fon).sum()))
PY
