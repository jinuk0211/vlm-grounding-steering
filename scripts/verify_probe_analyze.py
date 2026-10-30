"""E0' analysis: per-layer linear probes on verify_probe_collect.py output.

Targets   T1  margin < 0          asked "Is there a {node}?", the model would deny it
          T2  kind == hallucinated (CHAIR ground truth)
Features  F1  h_real at the query that predicts the mention (where decoding could intervene)
          F2  h_gray, same position, same caption, gray image: what the text alone gives away
          F4  h_real - h_gray: the image's part only
          F1b / F2b  the same at the mention token itself
          F3  [log p of the mention token, object order, caption position]  (confidence floor)
Unit: first mention of each (image, object). Image-level 5-fold CV. Inside each training fold:
standardize, PCA (<= 256 components, whitened), logistic regression (class-balanced) with L2 picked
from a grid by an inner image-level 3-fold CV. AUROC on pooled out-of-fold scores; 95% CIs and the
F1 - F2 difference by cluster bootstrap over images.
Decision rules, fixed before the run (T1, F1 at its best layer):
  pass            F1 >= 0.85 and F1 - F2 >= 0.10              -> go to steering
  language prior  F1 - F2 < 0.05                              -> read-steering off, keep B only
  no gain         F1 - F3 < 0.05                              -> off
  otherwise ambiguous (e.g. 0.75-0.85)                        -> run the remaining captions
Reference: T2 AUROC of the question-mode margin itself on the same mentions (recovery = F1 / that).
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

DEV='cuda' if torch.cuda.is_available() else 'cpu'
GRID=torch.tensor([1e-3,1e-2,1e-1,1.0])

def auroc(s,y):
    s=np.asarray(s,float);y=np.asarray(y).astype(bool);p=int(y.sum());n=len(y)-p
    if p==0 or n==0:return float('nan')
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True);rk=(np.cumsum(cnt)-(cnt-1)/2)[inv]
    return float((rk[y].sum()-p*(p+1)/2)/(p*n))

def fold_ids(groups,k,seed):
    u=np.unique(groups);perm=np.random.default_rng(seed).permutation(len(u));f={g:i%k for i,g in zip(range(len(u)),u[perm])}
    return np.array([f[g] for g in groups])

def prep(Xtr,Xte,dim=256):
    """per layer: standardize on the training rows, PCA (exact SVD, batched over layers), whiten. X [layers, n, D]"""
    mu=Xtr.mean(1,keepdim=True);sd=Xtr.std(1,keepdim=True)+1e-4;Xtr=(Xtr-mu)/sd;Xte=(Xte-mu)/sd
    if Xtr.shape[2]>dim:
        _,_,Vh=torch.linalg.svd(Xtr,full_matrices=False);V=Vh[:,:min(dim,Xtr.shape[1]-1)].transpose(1,2)
        Xtr=Xtr@V;Xte=Xte@V;s=Xtr.std(1,keepdim=True)+1e-4;Xtr=Xtr/s;Xte=Xte/s
    return Xtr,Xte

def fit(Z,Y):
    """Z [k,n,d], Y [n,t] -> W [k,d,t,g], b [k,t,g]: one class-balanced L2 logistic per layer, target and grid value.
    The objective is a sum of independent convex problems, so they are fitted jointly."""
    k,t,g=Z.shape[0],Y.shape[1],len(GRID);l2=GRID.to(Z.device)
    W=torch.zeros(k,Z.shape[2],t,g,device=Z.device,requires_grad=True);b=torch.zeros(k,t,g,device=Z.device,requires_grad=True)
    pw=((len(Y)-Y.sum(0))/Y.sum(0).clamp_min(1))[:,None].expand(t,g)
    opt=torch.optim.LBFGS([W,b],max_iter=300,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();lg=torch.einsum('knd,kdtg->kntg',Z,W)+b[:,None]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(lg,Y[None,:,:,None].expand_as(lg),pos_weight=pw,reduction='none').mean(1).sum()+(l2*(W**2).sum(1)).sum()
        loss.backward();return loss
    opt.step(closure);return W.detach(),b.detach()

def score(Z,W,b):return (torch.einsum('knd,kdtg->kntg',Z,W)+b[:,None]).cpu().numpy()

def cv(X,Y,groups,outer=5,inner=3,seed=0):
    """X [n, layers, D] -> out-of-fold scores [layers, n, targets] and the chosen L2 per fold/layer/target"""
    X=X.float().to(DEV).transpose(0,1);Yt=torch.from_numpy(Y).float().to(DEV);K=X.shape[0]
    S=np.zeros((K,)+Y.shape);fo=fold_ids(groups,outer,seed);chosen=[]
    for f in range(outer):
        tr=np.where(fo!=f)[0];te=np.where(fo==f)[0]
        Ztr,Zte=prep(X[:,tr],X[:,te]);fi=fold_ids(groups[tr],inner,seed+1);Si=np.zeros((K,len(tr),Y.shape[1],len(GRID)))
        for k in range(inner):
            a=np.where(fi!=k)[0];v=np.where(fi==k)[0];W,b=fit(Ztr[:,a],Yt[tr][a]);Si[:,v]=score(Ztr[:,v],W,b)
        best=np.array([[int(np.nanargmax([auroc(Si[l,:,j,g],Y[tr,j]) for g in range(len(GRID))])) for j in range(Y.shape[1])] for l in range(K)])
        W,b=fit(Ztr,Yt[tr]);s=score(Zte,W,b);chosen.append(GRID.numpy()[best].tolist())
        for l in range(K):
            for j in range(Y.shape[1]):S[l,te,j]=s[l,:,j,best[l,j]]
    return S,chosen

def boot(groups,fn,n=1000,seed=0):
    u=np.unique(groups);idx={g:np.where(groups==g)[0] for g in u};rng=np.random.default_rng(seed);v=[]
    for _ in range(n):
        ii=np.concatenate([idx[g] for g in rng.choice(u,len(u))]);x=fn(ii)
        if not np.isnan(x):v.append(x)
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]

def main(a):
    d=torch.load(a.input,weights_only=False);meta=d['meta'];NL=d['n_layers']
    keep=[i for i,m in enumerate(meta) if m['first_mention']] if not a.all_mentions else list(range(len(meta)))
    M=[meta[i] for i in keep];HR=d['h_real'][keep];HG=d['h_gray'][keep]
    groups=np.array([m['image_id'] for m in M]);Y=np.array([[m['margin']<0,m['kind']=='hallucinated'] for m in M],float)
    tnames=['T1_denied_when_asked','T2_hallucinated']
    out={'input':str(a.input),'unit':'all mentions' if a.all_mentions else 'first mention of each (image, object)','n':len(M),'images':int(len(np.unique(groups))),
         'positives':{t:int(Y[:,j].sum()) for j,t in enumerate(tnames)},'n_layers':NL,'device':DEV,
         'label_agreement_T1_T2':float((Y[:,0]==Y[:,1]).mean()),'unmatched_labels':len(d.get('unmatched',[]))}
    lp=np.array([m['logp_real'] for m in M]);margin=np.array([m['margin'] for m in M])
    out['reference']={t:{'-logp_real':auroc(-lp,Y[:,j]),'-logp_gray':auroc(-np.array([m['logp_gray'] for m in M]),Y[:,j])} for j,t in enumerate(tnames)}
    out['reference']['T2_hallucinated']['question_margin_upper_bound']=auroc(-margin,Y[:,1])
    feats={'F1':lambda:HR[:,:,0],'F2':lambda:HG[:,:,0],'F4':lambda:HR[:,:,0].float()-HG[:,:,0].float(),'F1b':lambda:HR[:,:,1],'F2b':lambda:HG[:,:,1]}
    S={}
    F3=torch.tensor([[m['logp_real'],m['object_order'],m['t']] for m in M]).float()[:,None];s,_=cv(F3,Y,groups);S['F3']=s[0]
    curves={k:{t:[] for t in tnames} for k in feats};lam={}
    for k,f in feats.items():
        s,lam[k]=cv(f(),Y,groups)
        for l in range(NL+1):
            S[(k,l)]=s[l]
            for j,t in enumerate(tnames):curves[k][t].append(auroc(s[l,:,j],Y[:,j]))
        print(f'{k}: T1 '+' '.join(f'{x:.2f}' for x in curves[k][tnames[0]])+' | T2 '+' '.join(f'{x:.2f}' for x in curves[k][tnames[1]]),flush=True)
    out['layer_curves']=curves;out['F3']={t:auroc(S['F3'][:,j],Y[:,j]) for j,t in enumerate(tnames)}
    res={}
    for j,t in enumerate(tnames):
        bl=int(np.nanargmax(curves['F1'][t]));s1=S[('F1',bl)][:,j];s2=S[('F2',bl)][:,j];s3=S['F3'][:,j];y=Y[:,j]
        r={'F1_best_layer':bl,'F1':curves['F1'][t][bl],'F1_ci95':boot(groups,lambda ii:auroc(s1[ii],y[ii])),
           'F2_same_layer':curves['F2'][t][bl],'F2_ci95':boot(groups,lambda ii:auroc(s2[ii],y[ii])),
           'F2_best':float(np.nanmax(curves['F2'][t])),'F2_best_layer':int(np.nanargmax(curves['F2'][t])),
           'F4_same_layer':curves['F4'][t][bl],'F4_best':float(np.nanmax(curves['F4'][t])),'F4_best_layer':int(np.nanargmax(curves['F4'][t])),
           'F1b_best':float(np.nanmax(curves['F1b'][t])),'F1b_best_layer':int(np.nanargmax(curves['F1b'][t])),
           'F3':out['F3'][t],'F1_minus_F2':curves['F1'][t][bl]-curves['F2'][t][bl],
           'F1_minus_F2_ci95':boot(groups,lambda ii:auroc(s1[ii],y[ii])-auroc(s2[ii],y[ii])),
           'F1_minus_F3':curves['F1'][t][bl]-out['F3'][t],'F1_minus_F3_ci95':boot(groups,lambda ii:auroc(s1[ii],y[ii])-auroc(s3[ii],y[ii]))}
        if t=='T2_hallucinated':
            ub=out['reference'][t]['question_margin_upper_bound'];r['question_margin_upper_bound']=ub;r['recovery_of_upper_bound']=r['F1']/ub
        res[t]=r
    t1=res['T1_denied_when_asked']
    if t1['F1']>=0.85 and t1['F1_minus_F2']>=0.10:v='pass: caption-time state carries the verification signal beyond the language prior -> steering'
    elif t1['F1_minus_F2']<0.05:v='language prior: real and gray probes read about the same -> read-steering off, keep B (verification-mode vector) only'
    elif t1['F1_minus_F3']<0.05:v='no gain over confidence: hidden state adds nothing beyond log p / order / position -> off'
    else:v='ambiguous: between the rules (e.g. F1 0.75-0.85) -> run the remaining captions'
    res['verdict']=v;res['note']='best layer is chosen on the same out-of-fold scores, so F1 at it is slightly optimistic; compare against the whole curve'
    out['results']=res;out['chosen_l2_per_fold']=lam
    p=Path(a.input).with_name(Path(a.input).stem+'-probe-summary.json');p.write_text(json.dumps(out,indent=1))
    print(json.dumps({'n':out['n'],'positives':out['positives'],'reference':out['reference'],'results':res},indent=1));print('wrote',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--all-mentions',action='store_true')
    main(p.parse_args())
