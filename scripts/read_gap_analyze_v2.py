"""Read-gap E0 analysis (CPU). Inputs come from read_gap_collect.py.

A. Yes-logit offset baseline: threshold on margin = logit(Yes)-logit(No),
   tuned on half of the POPE images and tested on the other half (2-fold by
   image, 50 random splits). Oracle threshold reported as an upper bound only.
B. Vision-side probe: 80-class logistic probe on pooled image-token states of
   COCO train2014 (layer/pool chosen on a train-only validation split). Applied
   to POPE: probe-as-answer accuracy and the FN-vs-TP score comparison.
C. Answer-side read: per-layer linear probe on the answer-position state,
   2-fold by POPE image (analysis only), compared with the model's own margin.
"""
import argparse,json,math,random,re
from pathlib import Path
import torch

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def metrics(pred,gold):
    tp=sum(p and g for p,g in zip(pred,gold));fp=sum(p and not g for p,g in zip(pred,gold))
    fn=sum((not p) and g for p,g in zip(pred,gold));tn=len(gold)-tp-fp-fn
    prec=tp/max(tp+fp,1);rec=tp/max(tp+fn,1)
    return {'acc':(tp+tn)/len(gold),'f1':2*prec*rec/max(prec+rec,1e-9),'precision':prec,'recall':rec,
            'yes_ratio':(tp+fp)/len(gold),'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def auroc(score,gold):
    pos=[s for s,g in zip(score,gold) if g];neg=[s for s,g in zip(score,gold) if not g]
    if not pos or not neg:return float('nan')
    allv=sorted([(s,1) for s in pos]+[(s,0) for s in neg]);ranks={};i=0
    while i<len(allv):
        j=i
        while j<len(allv) and allv[j][0]==allv[i][0]:j+=1
        for k in range(i,j):ranks.setdefault(allv[k][0],(i+j+1)/2)
        i=j
    rp=sum(ranks[s] for s in pos);return (rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def best_threshold(margin,gold,objective='acc'):
    cands=sorted(set(margin));cands=[cands[0]-1]+[(a+b)/2 for a,b in zip(cands,cands[1:])]+[cands[-1]+1]
    best=None
    # coarse grid keeps this fast on 1,500 items
    step=max(1,len(cands)//400)
    for t in cands[::step]:
        m=metrics([x>t for x in margin],gold)[objective]
        if best is None or m>best[0]:best=(m,t)
    return best[1]

def paired_bootstrap(groups,pred_a,pred_b,gold,n=10000,seed=0):
    by={}
    for i,g in enumerate(groups):by.setdefault(g,[]).append(i)
    keys=sorted(by);rng=random.Random(seed);d={'acc':[],'f1':[]}
    for _ in range(n):
        idx=[i for k in (rng.choice(keys) for _ in keys) for i in by[k]]
        ma=metrics([pred_a[i] for i in idx],[gold[i] for i in idx]);mb=metrics([pred_b[i] for i in idx],[gold[i] for i in idx])
        for k in d:d[k].append(mb[k]-ma[k])
    out={}
    for k,v in d.items():v.sort();out[k]=[v[int(.025*n)],v[int(.975*n)]]
    return out

def yes_offset(rows):
    gold=[r['label']=='yes' for r in rows];margin=[r['margin'] for r in rows];groups=[r['group_id'] for r in rows]
    base=[m>0 for m in margin];keys=sorted(set(groups))
    res={'threshold0':metrics(base,gold),'base_generation_agreement':sum(r['logit_agrees_with_base'] for r in rows)/len(rows)}
    accs=[];f1s=[];ths=[];first=None
    for s in range(50):
        k=keys[:];random.Random(s).shuffle(k);half=set(k[:len(k)//2]);pred=[None]*len(rows)
        for fold in (half,set(keys)-half):
            tr=[i for i,g in enumerate(groups) if g not in fold];te=[i for i,g in enumerate(groups) if g in fold]
            t=best_threshold([margin[i] for i in tr],[gold[i] for i in tr]);ths.append(t)
            for i in te:pred[i]=margin[i]>t
        m=metrics(pred,gold);accs.append(m['acc']);f1s.append(m['f1'])
        if first is None:first=pred
    res['cv2_mean']={'acc':sum(accs)/50,'f1':sum(f1s)/50,'acc_range':[min(accs),max(accs)],'f1_range':[min(f1s),max(f1s)],
                     'threshold_median':sorted(ths)[len(ths)//2]}
    res['cv2_split0']=metrics(first,gold);res['cv2_split0_vs_base_CI95']=paired_bootstrap(groups,base,first,gold)
    t=best_threshold(margin,gold);res['oracle']={'threshold':t,**metrics([m>t for m in margin],gold)}
    res['margin_auroc']=auroc(margin,gold)
    return res

def fit_logreg(X,Y,wd=1e-3,steps=300,lr=0.05):
    torch.manual_seed(0);W=torch.zeros(X.shape[1],Y.shape[1],requires_grad=True);b=torch.zeros(Y.shape[1],requires_grad=True)
    opt=torch.optim.Adam([W,b],lr=lr)
    for _ in range(steps):
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(X@W+b,Y)+wd*(W**2).sum()
        loss.backward();opt.step()
    return W.detach(),b.detach()

def standardize(X,mu=None,sd=None):
    if mu is None:mu=X.mean(0);sd=X.std(0)+1e-4
    return (X-mu)/sd,mu,sd

def feats(pooled,layer,pool):
    x=pooled[:,layer].float()
    return {'mean':x[:,0],'max':x[:,1],'both':x.reshape(len(x),-1)}[pool]

def vision_probe(train_states,train_list,names,pope_states,rows,base_margin_gold):
    items={r['id']:r for r in read(train_list)};ids=train_states['ids']
    Y=torch.zeros(len(ids),80)
    for i,k in enumerate(ids):
        for c in items[k]['objects']:Y[i,c]=1
    P=train_states['pooled'];n=len(ids);perm=torch.randperm(n,generator=torch.Generator().manual_seed(42));va=perm[:n//5];tr=perm[n//5:]
    L=P.shape[1];layers=sorted(set([0,L//4,L//2,(3*L)//4,L-1]+list(range(2,L,4))))
    sel=[]
    for layer in layers:
        for pool in ('mean','max','both'):
            X=feats(P,layer,pool);Xt,mu,sd=standardize(X[tr]);W,b=fit_logreg(Xt,Y[tr])
            S=torch.sigmoid(((X[va]-mu)/sd)@W+b)
            au=[auroc(S[:,c].tolist(),Y[va,c].bool().tolist()) for c in range(80) if 0<Y[va,c].sum()<len(va)]
            sel.append({'layer':layer,'pool':pool,'val_mean_auroc':sum(au)/len(au)})
    best=max(sel,key=lambda s:s['val_mean_auroc'])
    X=feats(P,best['layer'],best['pool']);Xt,mu,sd=standardize(X[tr]);W,b=fit_logreg(Xt,Y[tr])
    Sva=torch.sigmoid(((X[va]-mu)/sd)@W+b)
    # one global probability threshold chosen on the train-side validation split (balanced accuracy)
    flat_s=Sva.flatten().tolist();flat_y=Y[va].flatten().bool().tolist()
    def bacc(t):
        tp=sum(s>t and y for s,y in zip(flat_s,flat_y));tn=sum(s<=t and not y for s,y in zip(flat_s,flat_y))
        return .5*tp/sum(flat_y)+.5*tn/(len(flat_y)-sum(flat_y))
    tau=max([i/100 for i in range(1,100)],key=bacc)
    Xall,mu,sd=standardize(X);W,b=fit_logreg(Xall,Y)  # final probe on all 5000 train images
    gi={g:i for i,g in enumerate(pope_states['groups'])};Xp=(feats(pope_states['pooled'],best['layer'],best['pool'])-mu)/sd
    S=torch.sigmoid(Xp@W+b);idx={nm:i for i,nm in enumerate(names)}
    score=[]
    for r in rows:
        obj=re.match(r'Is there an? (.+) in the image\?',r['prompt']).group(1).strip().lower()
        score.append(float(S[gi[r['group_id']],idx[obj]]))
    gold=[r['label']=='yes' for r in rows];base=[r['margin']>0 for r in rows]
    out={'selection':sel,'chosen':best,'tau_from_train_val':tau,'pope_auroc':auroc(score,gold),
         'probe_as_answer':metrics([s>tau for s in score],gold),'model_margin_auroc':auroc([r['margin'] for r in rows],gold)}
    pos=[i for i,g in enumerate(gold) if g];neg=[i for i,g in enumerate(gold) if not g]
    tp=[score[i] for i in pos if base[i]];fn=[score[i] for i in pos if not base[i]]
    fp=[score[i] for i in neg if base[i]];tn=[score[i] for i in neg if not base[i]]
    q=lambda v:{'n':len(v),'mean':sum(v)/max(len(v),1),'median':sorted(v)[len(v)//2] if v else None,'frac_above_tau':sum(s>tau for s in v)/max(len(v),1)}
    out['positives']={'TP':q(tp),'FN':q(fn),'auroc_TP_vs_FN':auroc(tp+fn,[1]*len(tp)+[0]*len(fn))}
    out['negatives']={'TN':q(tn),'FP':q(fp),'auroc_FP_vs_TN':auroc(fp+tn,[1]*len(fp)+[0]*len(tn))}
    return out,score

def answer_probe(states,rows):
    last=states['last'];assert states['ids']==[r['id'] for r in rows]
    gold=torch.tensor([r['label']=='yes' for r in rows],dtype=torch.float32)[:,None]
    groups=[r['group_id'] for r in rows];keys=sorted(set(groups));random.Random(0).shuffle(keys);half=set(keys[:len(keys)//2])
    fa=torch.tensor([g in half for g in groups]);out=[]
    for layer in range(last.shape[1]):
        X=last[:,layer].float();score=torch.zeros(len(rows))
        for tr in (fa,~fa):
            Xt,mu,sd=standardize(X[tr]);W,b=fit_logreg(Xt,gold[tr],wd=1e-2,steps=200)
            score[~tr]=(((X[~tr]-mu)/sd)@W+b)[:,0]
        out.append({'layer':layer,'auroc':auroc(score.tolist(),gold[:,0].bool().tolist())})
    return out

def main(a):
    report={}
    for model in a.models:
        tag=f'{model}-pope-v2' if (a.e0/f'{model}-pope-v2-rows.jsonl').exists() else f'{model}-pope'
        rows=read(a.e0/f'{tag}-rows.jsonl');r={'pope_tag':tag,'yes_offset':yes_offset(rows)}
        train=a.e0/f'{model}-train-vision-states.pt'
        if train.exists():
            names=json.loads((a.probe_set/'categories.json').read_text())
            r['vision_probe'],_=vision_probe(torch.load(train),a.probe_set/'list.jsonl',names,
                torch.load(a.e0/f'{tag}-vision-states.pt'),rows,None)
        if a.answer_probe:r['answer_probe_by_layer']=answer_probe(torch.load(a.e0/f'{tag}-answer-states.pt'),rows)
        report[model]=r
    out=a.e0/a.out;out.write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1)[:20000])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--e0',type=Path,required=True);p.add_argument('--probe-set',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['qwen','gemma']);p.add_argument('--out',default='analysis.json')
    p.add_argument('--answer-probe',action='store_true')
    main(p.parse_args())
