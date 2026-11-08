"""CPU re-check of paper claims from saved results. Writes one JSON per model under the output folder."""
import glob,json,sys,warnings
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings('ignore')
sys.path.insert(0,'scripts')
from score_vti_alpha_sweep import chair_scorer
R=Path('/home/project-b/LJW/vlm-grounding-steering/outputs');LL=R/'logit-lens-20261104';OUT=R/'audit-recheck-20261105';OUT.mkdir(exist_ok=True)
BASE={'qwen':R/'qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl','gemma':R/'gemma-grains-fp16-v2-20261029-1793246507/chair/base.jsonl'}
PB=[0,32,64,96,128,192,256,384,513];BAND={'qwen':range(22,28),'gemma':range(30,42)};LATE={'qwen':27,'gemma':34}
rng=np.random.default_rng(0)

def auroc(s,y):
    s=np.asarray(s,float);y=np.asarray(y).astype(bool);ok=~np.isnan(s);s,y=s[ok],y[ok];p=int(y.sum());n=len(y)-p
    if p==0 or n==0:return float('nan')
    _,i,c=np.unique(s,return_inverse=True,return_counts=True);r=(np.cumsum(c)-(c-1)/2)[i];return float((r[y].sum()-p*(p+1)/2)/(p*n))
def strat(s,y,b,min_each=3):
    """AUROC within strata, weighted by the number of positive-negative pairs"""
    a=[];w=[]
    for k in np.unique(b):
        m=b==k;v=s[m];yy=y[m];ok=~np.isnan(v);p=int(yy[ok].sum());n=int((~yy[ok]).sum())
        if p>=min_each and n>=min_each:a.append(auroc(v,yy));w.append(p*n)
    return float(np.average(a,weights=w)) if a else float('nan')
def ci(fn,g,n=500):
    u=np.unique(g);idx={k:np.where(g==k)[0] for k in u};v=[]
    for _ in range(n):
        x=fn(np.concatenate([idx[k] for k in rng.choice(u,len(u))]))
        if x==x:v.append(x)
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]
def logistic(X,y,it=50,l2=1e-4):
    X=np.c_[np.ones(len(X)),X];b=np.zeros(X.shape[1])
    for _ in range(it):
        p=np.clip(1/(1+np.exp(-X@b)),1e-6,1-1e-6);W=p*(1-p);H=(X*W[:,None]).T@X+l2*np.eye(X.shape[1]);b=b+np.linalg.solve(H,X.T@(y-p)-l2*b)
    return b

def load(m):
    bad={json.loads(l)['id'] for l in open(BASE[m]) if 'addCriterion' in json.loads(l)['prediction']}
    vd={r['id']:r for f in sorted(glob.glob(str(R/f'visual-decay-20261030/{m}-[0-9]*-[0-9]*.pt'))) if 'partial' not in f for r in torch.load(f,weights_only=False)}
    tx=torch.load(LL/f'{m}-text-0-500.pt',weights_only=False)['records'];im=torch.load(LL/f'{m}-image-0-500.pt',weights_only=False)
    tk=torch.load(LL/f'{m}-topk-0-500.pt',weights_only=False)
    return bad,vd,tx,im,tk

def main(m):
    bad,vd,tx,im,tk=load(m);out={'model':m,'bad_captions':len(bad)};imr={r['id']:r for r in im['records']};inodes={n:i for i,n in enumerate(im['nodes'])}
    tkr={r['id']:r for r in tk['records']};tnodes={n:i for i,n in enumerate(tk['nodes'])};band=set(BAND[m]);NL=tk['n_layers']
    marg={}
    for l in open(R/f'caption-selfcheck-20261030/{m}-base-rows.jsonl'):
        r=json.loads(l);marg[(r['image_id'],r['node'])]=r
    rows=[]
    for r in tx:
        order={};v=vd[r['id']];real=v['lens_real'][-1].float().numpy();gray=v['lens_gray'][-1].float().numpy();assert len(real)==r['T']
        t_r=tkr[r['id']];ca=np.array(t_r['caption_ids']);prev=np.r_[-1,ca[:-1]];h=t_r['hits'];h=h[(h[:,4]!=ca[h[:,1]])&(h[:,4]!=prev[h[:,1]])];hb=h[np.isin(h[:,0],list(band))];hl=h[h[:,0]==NL]
        for x in r['mentions']:
            if not x['first']:continue
            order[x['node']]=len(order);t=x['t'];n=x['node'];i=tnodes.get(n)
            d={'id':r['id'],'img':r['image_id'],'node':n,'t':t,'T':r['T'],'order':order[n],'hall':x['kind']=='hallucinated','logp':real[t],'gray':gray[t],'e':real[t]-gray[t],
               'rank_late':float(r['rank'][LATE[m],t]),'margin':marg.get((r['image_id'],n),{}).get('margin',np.nan),'bad':r['id'] in bad}
            j=inodes.get(n);d['lens']=float(imr[r['id']]['max'][-1,j]) if j is not None else np.nan
            for k,name,src,dist in ((100,'fore100',hb,5),(10,'fore10',hb,5),(100,'fore100far',hb,17),(100,'own100',hl,5)):
                if i is None or t<dist:d[name]=np.nan
                else:
                    s=src[(src[:,3]==i)&(src[:,2]<k)&(src[:,1]<=t-dist)];d[name]=len(np.unique(s[:,1]))/(t-dist+1)
            rows.append(d)
    A={k:np.array([d[k] for d in rows]) for k in rows[0]}
    y=~A['hall'];b=np.digitize(A['t'],[0,8,16,32,64,128]);g=A['img']  # the position bins of scripts/rbd_offline.py
    # (a) where hallucinations happen
    pos={}
    Ttot=[r['T'] for r in tx]
    for lo,hi in zip(PB,PB[1:]):
        s=(A['t']>=lo)&(A['t']<hi);tok=sum(max(0,min(T,hi)-lo) for T in Ttot)
        pos[f'{lo}-{hi-1}']={'hall':int(A['hall'][s].sum()),'correct':int((~A['hall'][s]).sum()),'share':float(A['hall'][s].mean()) if s.any() else None,'hall_per100':100*float(A['hall'][s].sum())/max(tok,1),'correct_per100':100*float((~A['hall'][s]).sum())/max(tok,1),'tokens':tok}
    out['position']=pos
    h=A['hall'];out['position_quantiles']={'hall_median_t':float(np.median(A['t'][h])),'correct_median_t':float(np.median(A['t'][~h])),'hall_t>=128':float((A['t'][h]>=128).mean()),'hall_t>=64':float((A['t'][h]>=64).mean()),'tokens_t>=128':float(sum(max(0,T-128) for T in Ttot)/sum(Ttot)),'tokens_t>=64':float(sum(max(0,T-64) for T in Ttot)/sum(Ttot)),'hall_rel_median':float(np.median((A['t']/A['T'])[h])),'correct_rel_median':float(np.median((A['t']/A['T'])[~h]))}
    tab={}
    for lo,hi in ((0,64),(64,128),(128,10**6)):
        for o,nm in ((0,'1'),(1,'2'),(2,'3'),(3,'4+')):
            s=(A['t']>=lo)&(A['t']<hi)&((A['order']==o) if o<3 else (A['order']>=3));tab[f't{lo}-{hi}|order{nm}']=[int(s.sum()),float(h[s].mean()) if s.sum() else None]
    out['order_x_position']=tab
    X=np.c_[(np.log1p(A['t'])-np.log1p(A['t']).mean())/np.log1p(A['t']).std(),(A['order']-A['order'].mean())/A['order'].std()];co=logistic(X,h.astype(float));u=np.unique(g);idx={k:np.where(g==k)[0] for k in u};bs=[]
    for _ in range(300):
        ii=np.concatenate([idx[k] for k in rng.choice(u,len(u))]);bs.append(logistic(X[ii],h[ii].astype(float),it=25))
    bs=np.array(bs);out['logit_hall_on_position_and_order']={'log_t_per_sd':[float(co[1]),float(np.quantile(bs[:,1],.025)),float(np.quantile(bs[:,1],.975))],'order_per_sd':[float(co[2]),float(np.quantile(bs[:,2],.025)),float(np.quantile(bs[:,2],.975))]}
    # (b) gap without broken captions
    keep=~A['bad'];gp={}
    for nm,s in (('all',np.ones(len(h),bool)),('without_loop_captions',keep)):
        ok=s&~np.isnan(A['margin']);gp[nm]={'hall':int(h[ok].sum()),'correct':int((~h[ok]).sum()),'hall_denied':float((A['margin'][ok&h]<0).mean()),'correct_denied':float((A['margin'][ok&~h]<0).mean()),'auroc_margin':auroc(A['margin'][ok],y[ok])}
    out['gap']=gp
    # (c) ranking cells: position-matched AUROC, correct > hallucinated
    q=np.quantile(A['logp'],[.2,.4,.6,.8]);cb=np.digitize(A['logp'],q);bc=b*10+cb;rk={}
    for nm,s_ in (('logp',A['logp']),('gray_logp',A['gray']),('writing_step_evidence',A['e']),('late_layer_rank(-)',-A['rank_late']),('image_token_lens',A['lens']),('question_margin',A['margin']),('foreshadow_top100_d>=5',A['fore100']),('foreshadow_top10_d>=5',A['fore10']),('foreshadow_top100_d>=17',A['fore100far']),('last_layer_pooled_top100_d>=5',A['own100'])):
        rk[nm]={'raw':auroc(s_,y),'position':strat(s_,y,b),'position_ci':ci(lambda ii:strat(s_[ii],y[ii],b[ii]),g),'position_confidence':strat(s_,y,bc),'position_confidence_ci':ci(lambda ii:strat(s_[ii],y[ii],bc[ii]),g),'n':int((~np.isnan(s_)).sum()),
                'position_no_loops':strat(s_[keep],y[keep],b[keep])}
    sub=~np.isnan(A['fore100']);rk['logp_on_foreshadow_subset']={'position':strat(A['logp'][sub],y[sub],b[sub])}
    # within object, position-binned: demean by object
    nd=A['node'];dm=lambda v:np.array([v[i]-np.nanmean(v[nd==nd[i]]) for i in range(len(v))])
    for nm,s_ in (('logp',A['logp']),('writing_step_evidence',A['e']),('foreshadow_top100_d>=5',A['fore100']),('image_token_lens',A['lens'])):
        rk[nm]['position_within_object']=strat(dm(s_),y,b)
    out['ranking']=rk
    # e at matched confidence, numbers for the text: mean evidence by confidence bin at t>=64
    late=A['t']>=64;out['evidence_by_confidence_t>=64']={f'q{c}':{'n_hall':int((h&late&(cb==c)).sum()),'n_corr':int((~h&late&(cb==c)).sum()),'e_hall':float(A['e'][h&late&(cb==c)].mean()) if (h&late&(cb==c)).any() else None,'e_corr':float(A['e'][~h&late&(cb==c)].mean()) if (~h&late&(cb==c)).any() else None} for c in range(5)}
    out['evidence_mean_t>=64']={'hall':float(A['e'][h&late].mean()),'correct':float(A['e'][~h&late].mean()),'n':[int((h&late).sum()),int((~h&late).sum())]}
    # (d) co-occurrence
    sc=chair_scorer();test={r['image_id'] for r in tx};cnt=Counter();pair=Counter();N=0
    for iid,objs in sc.imid_to_objects.items():
        if iid in test:continue
        o=sorted(set(objs));N+=1;cnt.update(o);pair.update((a,c) for i,a in enumerate(o) for c in o[i+1:])
    nodes=sorted(set(sc.inverse_synonym_dict.values())-{''});pm=lambda a,c:np.log((pair[tuple(sorted((a,c)))]+1)*N/((cnt[a]+1)*(cnt[c]+1)))
    PMI={a:{c:pm(a,c) for c in nodes if c!=a} for a in nodes};lab={r['image_id']:r['labels'] for r in im['records']};S=[];C=[];ND=[];G=[];LN=[]
    for r in im['records']:
        L=r['labels'];GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
        if not GT:continue
        for n in nodes:
            c='hall' if n in H else 'correct' if n in M and n in GT else 'missed' if n in GT else 'absent' if n not in M else None
            if c is None:continue
            others=[PMI[n][x] for x in GT if x!=n]
            if not others:continue
            j=inodes.get(n);S.append(max(others));C.append(c);ND.append(n);G.append(r['image_id']);LN.append(float(r['max'][-1,j]) if j is not None else np.nan)
    S=np.array(S);C=np.array(C);ND=np.array(ND);G=np.array(G);LN=np.array(LN);mk=(C=='hall')|(C=='absent');w=[auroc(S[mk&(ND==n)],C[mk&(ND==n)]=='hall') for n in np.unique(ND[C=='hall']) if ((C=='hall')&(ND==n)).sum()>=3]
    sp=[]
    for n in nodes:
        a=(C=='absent')&(ND==n)&~np.isnan(LN)
        if a.sum()>50:sp.append(np.corrcoef(np.argsort(np.argsort(S[a])),np.argsort(np.argsort(LN[a])))[0,1])
    out['cooccurrence']={'train_images':N,'hall_vs_absent_pooled':auroc(S[mk],C[mk]=='hall'),'hall_vs_absent_ci':ci(lambda ii:auroc(S[mk][ii],(C[mk]=='hall')[ii]),G[mk],n=200),'hall_vs_absent_within_object':float(np.nanmean(w)),'objects_within':len(w),
        'hall_vs_correct_pooled':auroc(S[(C=='hall')|(C=='correct')],C[(C=='hall')|(C=='correct')]=='hall'),'median_max_pmi':{c:float(np.median(S[C==c])) for c in ('hall','correct','missed','absent')},'share_pmi>1':{c:float((S[C==c]>1).mean()) for c in ('hall','correct','missed','absent')},
        'image_lens_vs_pmi_spearman_within_object_absent':float(np.nanmean(sp)),'pmi_objects':len(sp)}
    (OUT/f'{m}.json').write_text(json.dumps(out,indent=1,default=float));print(json.dumps(out,default=float)[:200]);print('done',m,flush=True)

if __name__=='__main__':main(sys.argv[1])
