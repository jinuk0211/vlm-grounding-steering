"""RBD pilot step 1 (offline): does a reliability-weighted belief over V_obj separate hallucinated from correct
first mentions better than the token's own log p?

belief   A_t(o) = sum_{u<=t} r(u) * log p_u(o),  log b_t = A_t - logsumexp_o A_t      (uniform prior)
scores at the first mention of an object (step t_m, token o*):
  logp            log p_{t_m}(o*)                       (baseline)
  belief_prev     log b_{t_m-1}(o*)                     (evidence before the model writes it)
  belief          log b_{t_m}(o*)
  logp+lam*belief the decoding rule's adjusted logit, lam in LAMS
r kinds: measured (gray-KL curve), const (r=1), step0 (only the first step counts).
AUROC = P(score of a correct mention > score of a hallucinated one); position-matched = pair-weighted mean of
within-bin AUROCs; 95% CIs by bootstrap over images; differences vs logp are paired on the same resamples.
"""
import json,sys
from pathlib import Path
import numpy as np
import torch
from rbd_common import r_array

LAMS=(0.1,0.3,1.0)
PBINS=[0,8,16,32,64,128,10**6]

def auroc(pos,neg):
    pos=np.asarray(pos);neg=np.asarray(neg)
    if len(pos)==0 or len(neg)==0:return np.nan
    allv=np.concatenate([pos,neg]);_,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True)
    rk=(np.cumsum(cnt)-(cnt-1)/2)[inv];return float((rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def pm_auroc(s,y,t):
    num=den=0.0
    for a,b in zip(PBINS,PBINS[1:]):
        m=(t>=a)&(t<b);p=s[m&y];n=s[m&~y]
        if len(p) and len(n):w=len(p)*len(n);num+=w*auroc(p,n);den+=w
    return num/den if den else np.nan

def main(path,model):
    recs=torch.load(path/f'{model}-0-500.pt',weights_only=False);V=recs[0]['logp_vobj'].shape[1]
    vobj=json.loads((path/f'{model}-vobj2.json').read_text());col={v:i for i,v in enumerate(vobj)}
    Tmax=max(r['T'] for r in recs);R={k:r_array(model,k,Tmax) for k in ('measured','const','step0')}
    rows=[];cover=[0,0]
    for ci,r in enumerate(recs):
        L=r['logp_vobj'].float()
        for mn in r['mentions']:
            cover[1]+=1
            if not mn['in_vobj']:continue
            cover[0]+=1;tm=mn['t'];j=col[mn['token']]
            row={'img':ci,'t':tm,'y':mn['kind']=='correct','logp':float(r['logp_actual'][tm])}
            for k,rr in R.items():
                A=torch.cumsum(torch.from_numpy(rr[:r['T'],None])*L,0);lb=A-torch.logsumexp(A,1,keepdim=True)
                row[f'belief_prev_{k}']=float(lb[tm-1,j]) if tm>0 else -np.log(V)
                row[f'belief_{k}']=float(lb[tm,j])
                for lam in LAMS:row[f'logp+{lam}belief_{k}']=row['logp']+lam*row[f'belief_{k}']
            rows.append(row)
    keys=[k for k in rows[0] if k not in ('img','t','y')]
    y=np.array([x['y'] for x in rows]);t=np.array([x['t'] for x in rows]);img=np.array([x['img'] for x in rows])
    S={k:np.array([x[k] for x in rows]) for k in keys}
    rng=np.random.default_rng(0);imgs=np.unique(img);by={i:np.where(img==i)[0] for i in imgs};boots=[]
    for _ in range(1000):
        idx=np.concatenate([by[i] for i in rng.choice(imgs,len(imgs))]);boots.append(idx)
    out={'model':model,'vobj':V,'first_mentions':cover[1],'in_vobj':cover[0],'correct':int(y.sum()),'hallucinated':int((~y).sum()),'scores':{}}
    base_b=[auroc(S['logp'][b][y[b]],S['logp'][b][~y[b]]) for b in boots]
    for k in keys:
        a=auroc(S[k][y],S[k][~y]);bb=[auroc(S[k][b][y[b]],S[k][b][~y[b]]) for b in boots];d=np.array(bb)-np.array(base_b)
        out['scores'][k]={'auroc':a,'ci95':[float(np.nanquantile(bb,.025)),float(np.nanquantile(bb,.975))],
                          'minus_logp_ci95':[float(np.nanquantile(d,.025)),float(np.nanquantile(d,.975))],'position_matched':pm_auroc(S[k],y,t)}
    (path/f'{model}-offline.json').write_text(json.dumps(out,indent=1))
    print(f"{model}: {out['first_mentions']} first mentions, {out['in_vobj']} with the first token in V_obj ({out['correct']} correct / {out['hallucinated']} hallucinated)")
    for k,v in sorted(out['scores'].items(),key=lambda x:-x[1]['auroc']):
        print(f"  {k:28s} AUROC {v['auroc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]  vs logp [{v['minus_logp_ci95'][0]:+.3f},{v['minus_logp_ci95'][1]:+.3f}]  pos-matched {v['position_matched']:.3f}")

if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2])
