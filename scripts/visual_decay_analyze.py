"""Summarize visual_decay_collect.py outputs (CPU only).

E1 attention   by caption-position bin and layer band: mass on pre-image tokens (template and
               system tokens before the image, where sinks sit) / image / prompt after the
               image / caption so far / last 16 caption tokens; shares among content tokens
               (pre-image removed): image vs text, the "text takes over 90%" claim; share of
               image keys the mask still lets the step see (Gemma sliding window); rollout with
               absorbing image tokens (share of attention paths that end on the image); the
               prompt positions that receive the most attention, with their hidden-state norm.
E2 lens/repr   by bin, for every layer: logit-lens image contribution log p(real)-log p(gray),
               ||h_real-h_gray||/||h_real||, cos(h_real,h_gray), cos to the mean image state.
               Linear probes for COCO object presence from the step-t hidden state, real vs gray
               image (the gray run keeps the caption, so it measures what the text alone gives
               away), 5-fold CV over captions, macro AUROC; also scored only on objects the
               caption has not mentioned before t. Fixed cohort (captions long enough for every
               step) so a falling curve cannot come from short captions dropping out.
E3 causal      by bin: mean dlog p and KL to the unmodified run and greedy agreement, per
               condition. Direct vs total effect on tokens after k: image keys blocked from k
               (direct reading only) vs gray image (everything).
Means are over captions (each caption's bin mean counts once); 95% CIs are cluster bootstraps
over captions.
"""
import argparse,glob,json,re,warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings('ignore',category=RuntimeWarning)
BINS=[0,16,32,64,96,128,192,256,384,513]
LAB=[f'{a}-{b-1}' for a,b in zip(BINS,BINS[1:])]
G={'pre_image':0,'image':1,'prompt':2,'caption':3,'recent':4}

def bin_means(v):
    v=np.asarray(v,float);return np.array([np.nanmean(v[a:b]) if a<len(v) else np.nan for a,b in zip(BINS,BINS[1:])])

def ci(x,n=1000,seed=0):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return [None,None]
    m=np.random.default_rng(seed).choice(x,(n,len(x))).mean(1);return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def table(M):
    """M [captions, bins] -> per-bin mean, CI, number of captions"""
    M=np.asarray(M,float);out=[]
    for j,l in enumerate(LAB):
        c=M[:,j];k=int((~np.isnan(c)).sum())
        if k:out.append({'bin':l,'mean':float(np.nanmean(c)),'ci95':ci(c),'n':k})
    return out

def matrix(per_rec):
    """per_rec: list of [rows, T] arrays -> [rows, bins] mean over captions of bin means"""
    return np.nanmean(np.stack([np.stack([bin_means(x) for x in a]) for a in per_rec]),0)

def late_minus_early(vals,T,early=32,late=128,min_T=160):
    d=[np.nanmean(v[late:])-np.nanmean(v[:early]) for v,t in zip(vals,T) if t>=min_T]
    return {'mean':float(np.mean(d)) if d else None,'ci95':ci(d),'n':len(d),'early_steps':f'0-{early-1}','late_steps':f'{late}-end'}

def ratio_ci(num,den,n=1000,seed=0):
    num=np.asarray(num);den=np.asarray(den);rng=np.random.default_rng(seed);i=rng.integers(0,len(num),(n,len(num)))
    r=num[i].mean(1)/den[i].mean(1);return float(num.mean()/den.mean()),[float(np.quantile(r,.025)),float(np.quantile(r,.975))]

# ---------------------------------------------------------------- E1
def e1(recs):
    NL=recs[0]['attn'].shape[0];T=[r['T'] for r in recs]
    full=[l for l in range(NL) if all(float(r['visible'][l].float().min())>0.999 for r in recs)]
    bands={'all':list(range(NL)),'early':list(range(0,NL//3)),'middle':list(range(NL//3,2*NL//3)),'late':list(range(2*NL//3,NL))}
    if len(full)<NL:bands|={'full_attention_layers':full,'window_limited_layers':[l for l in range(NL) if l not in full]}
    out={'n_layers':NL,'layers_always_seeing_the_whole_image':full,
         'shares':'*_share = share among content tokens, i.e. with the pre-image tokens removed','bands':{}}
    for name,ls in bands.items():
        m={k:[] for k in (*G,'image_share','text_share','prompt_share','caption_share','recent_share','image_keys_visible')}
        for r in recs:
            a=r['attn'][ls].float().mean(0).numpy();rest=1-a[0]
            for k,i in G.items():m[k].append(a[i])
            for k,i in (('image',1),('prompt',2),('caption',3),('recent',4)):m[k+'_share'].append(a[i]/rest)
            m['text_share'].append((a[2]+a[3])/rest);m['image_keys_visible'].append(r['visible'][ls].float().mean(0).numpy())
        out['bands'][name]={'layers':ls,**{k:table([bin_means(x) for x in v]) for k,v in m.items()},
                            'image_share_late_minus_early':late_minus_early(m['image_share'],T),
                            'text_share_late_minus_early':late_minus_early(m['text_share'],T)}
    per=[(r['attn'][:,1]/(1-r['attn'][:,0])).float().numpy() for r in recs]
    out['per_layer_image_share']={'bins':LAB,'matrix':matrix(per).tolist()}
    out['per_layer_image_keys_visible']={'bins':LAB,'matrix':matrix([r['visible'].float().numpy() for r in recs]).tolist()}
    ro=[r['rollout'].float().numpy() for r in recs]
    out['rollout_absorbing_image']={'image':table([bin_means(x[1]) for x in ro]),'pre_image':table([bin_means(x[0]) for x in ro]),
                                    'image_late_minus_early':late_minus_early([x[1] for x in ro],T)}
    # where do the sinks sit? prompt positions are aligned across captions when the prompt and image size are fixed
    if len({r['capstart'] for r in recs})==1:
        recv=np.mean([r['key_recv'].float().mean(0).numpy() for r in recs],0)
        nrm=np.mean([r['prompt_norm'][NL//2].float().numpy() for r in recs],0);med=float(np.median(nrm))
        i0,i1=recs[0]['image_span']
        grp=lambda p:'pre_image' if p<i0 else 'image' if p<=i1 else 'prompt'
        out['most_attended_prompt_positions']=[{'position':int(p),'group':grp(p),'attention_received':float(recv[p]),
                                                'hidden_norm_over_median_at_mid_layer':float(nrm[p]/med)} for p in np.argsort(-recv)[:10]]
    # how concentrated is the attention that does reach the image (visual sink tokens)?
    conc=[]
    for r in recs:
        i0,i1=r['image_span'];v=r['key_recv'][:,i0:i1+1].float();v=v/v.sum(-1,keepdim=True).clamp_min(1e-12);k=max(1,int(0.01*v.shape[1]))
        conc.append(float(v.topk(k,-1).values.sum(-1).mean()))
    out['image_attention_on_top1pct_image_tokens']={'mean':float(np.mean(conc)),'ci95':ci(conc)}
    return out

# ---------------------------------------------------------------- E2
def e2(recs):
    T=[r['T'] for r in recs];depth=recs[0]['probe_layers']
    vcd=[(r['lens_real'].float()-r['lens_gray'].float()).numpy() for r in recs]
    rep=[r['repr'].float().numpy() for r in recs]
    metrics={'lens_image_contribution':vcd,'delta_norm_ratio':[x[:,1]/x[:,0] for x in rep],'cos_real_gray':[x[:,2] for x in rep],
             'cos_to_image_mean':[x[:,3] for x in rep],'cos_to_image_mean_centred':[x[:,4] for x in rep]}
    out={'layer_index':'0 = embedding output, l = output of decoder layer l','probe_layers':depth,'per_layer':{},'at_probe_layers':{}}
    for k,v in metrics.items():
        out['per_layer'][k]={'bins':LAB,'matrix':matrix(v).tolist()}
        out['at_probe_layers'][k]={str(l):table([bin_means(x[l]) for x in v]) for l in depth}
        out['at_probe_layers'][k+'_late_minus_early']={str(l):late_minus_early([x[l] for x in v],T) for l in depth}
    # object-mention tokens: logit-lens image contribution at the mention's first token
    obj={}
    for r,x in zip(recs,vcd):
        for nd,w,t,kind in r['mentions']:
            if t<r['T']:
                for l in sorted(set(depth)|{x.shape[0]-1}):obj.setdefault(kind,{}).setdefault(str(l),[]).append((t,float(x[l,t])))
    out['mention_tokens_lens_image_contribution']={kind:{l:{'early_t<64':_mci([v for t,v in xs if t<64]),'late_t>=64':_mci([v for t,v in xs if t>=64])}
                                                         for l,xs in d.items()} for kind,d in obj.items()}
    out['probes']=probes(recs)
    return out

def _mci(x):return {'mean':float(np.mean(x)) if x else None,'ci95':ci(x),'n':len(x)}

def auroc(s,y):
    s=np.asarray(s);y=np.asarray(y).astype(bool);p=int(y.sum());n=len(y)-p
    if p==0 or n==0:return np.nan
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True);rk=(np.cumsum(cnt)-(cnt-1)/2)[inv]
    return float((rk[y].sum()-p*(p+1)/2)/(p*n))

def fit_predict(Xtr,Ytr,Xte,dim=128,l2=1e-2):
    mu=Xtr.mean(0);sd=Xtr.std(0)+1e-4;Xtr=(Xtr-mu)/sd;Xte=(Xte-mu)/sd
    _,_,V=torch.pca_lowrank(Xtr,q=min(dim,*Xtr.shape),center=False);Ztr=Xtr@V;Zte=Xte@V
    s=Ztr.std(0)+1e-4;Ztr=Ztr/s;Zte=Zte/s
    W=torch.zeros(Ztr.shape[1],Ytr.shape[1],device=Ztr.device,requires_grad=True);b=torch.zeros(Ytr.shape[1],device=Ztr.device,requires_grad=True)
    opt=torch.optim.LBFGS([W,b],max_iter=200,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(Ztr@W+b,Ytr)+l2*(W**2).sum();loss.backward();return loss
    opt.step(closure)
    with torch.no_grad():return (Zte@W+b).cpu().numpy()

DEV='cuda' if torch.cuda.is_available() else 'cpu'  # probes are the slow part; a GPU makes them minutes faster

def cv_scores(X,Y,folds=5,seed=0):
    N=len(X);fold=np.random.default_rng(seed).permutation(N)%folds;S=np.zeros(Y.shape);X=X.to(DEV);Y=Y.to(DEV)
    for f in range(folds):
        tr=torch.from_numpy(fold!=f).to(DEV);te=torch.from_numpy(fold==f)
        S[te.numpy()]=fit_predict(X[tr],Y[tr],X[te.to(DEV)])
    return S,fold

def macro_auroc(S,Y,valid,fold,min_each=10):
    """AUROC inside each test fold, averaged over folds, then over classes. Pooling the folds'
    scores is biased below 0.5 for rare classes: a fold with more positives trains on fewer, so
    its intercept is lower exactly where the positives are (constant inputs scored ~0.38)."""
    a=[]
    for c in range(Y.shape[1]):
        v=valid[:,c];y=Y[v,c]
        if y.sum()<min_each or (1-y).sum()<min_each:continue
        per=[auroc(S[v&(fold==f),c],Y[v&(fold==f),c]) for f in np.unique(fold)]
        per=[x for x in per if not np.isnan(x)]
        if per:a.append(float(np.mean(per)))
    return (float(np.mean(a)) if a else None),len(a)

ONLY_LAYER=None  # set by --probe-layer: probe a single depth (index into probe_layers), so depths can run as parallel processes

def probes(recs,min_cohort=200):
    steps=sorted({t for r in recs for t in r['probe_steps']});layers=recs[0]['probe_layers']
    classes=sorted({o for r in recs for o in r['gt_objects']})
    # cohort: captions that reach the largest step that still keeps >= min_cohort captions
    reach=[s for s in steps if sum(r['T']>s for r in recs)>=min_cohort];s_c=max(reach) if reach else steps[0]
    cohort=[r for r in recs if r['T']>s_c]
    out={'classes_seen':len(classes),'cohort':{'max_step':s_c,'n':len(cohort)},'cohort_results':{},'all_captions_results':{}}
    for name,group,stps in (('cohort_results',cohort,[s for s in steps if s<=s_c]),('all_captions_results',recs,steps)):
        for li,l in enumerate(layers):
            if ONLY_LAYER is not None and li!=ONLY_LAYER:continue
            res={}
            for s in stps:
                rs=[r for r in group if s in r['probe_steps']]
                if len(rs)<50:continue
                k=[r['probe_steps'].index(s) for r in rs]
                Y=np.array([[c in r['gt_objects'] for c in classes] for r in rs],float)
                said=[{nd for nd,_,t,_ in r['mentions'] if t<s} for r in rs]
                unsaid=np.array([[c not in sd for c in classes] for sd in said])
                row={'n':len(rs)}
                for src in ('real','gray'):
                    X=torch.stack([r[f'probe_{src}'][li,j].float() for r,j in zip(rs,k)])
                    S,fold=cv_scores(X,torch.from_numpy(Y).float())
                    row[f'{src}_auroc'],row['n_classes']=macro_auroc(S,Y,np.ones_like(unsaid),fold)
                    row[f'{src}_auroc_unmentioned'],row['n_classes_unmentioned']=macro_auroc(S,Y,unsaid,fold)
                for v in ('','_unmentioned'):
                    a,b=row.get('real_auroc'+v),row.get('gray_auroc'+v);row['image_gain'+v]=None if a is None or b is None else a-b
                res[str(s)]=row
            out[name][str(l)]=res
        if name=='cohort_results' and len(group)>=50:
            Y=np.array([[c in r['gt_objects'] for c in classes] for r in group],float)
            out['image_tokens_ceiling']={}
            for li,l in enumerate(layers):
                if ONLY_LAYER is not None and li!=ONLY_LAYER:continue
                S,fold=cv_scores(torch.stack([r['vis_mean'][li].float() for r in group]),torch.from_numpy(Y).float())
                out['image_tokens_ceiling'][str(l)]=macro_auroc(S,Y,np.ones_like(Y,bool),fold)[0]
    return out

# ---------------------------------------------------------------- E3
def e3(recs):
    conds=list(recs[0]['causal']);T=[r['T'] for r in recs];out={'conditions':conds,'by_bin':{}}
    out['teacher_forced_greedy_reproduction']={'raw_argmax':float(np.mean([r['base_greedy'].float().mean() for r in recs])),
        'with_generation_config_repetition_penalty':float(np.mean([r.get('base_greedy_cfg',r['base_greedy']).float().mean() for r in recs]))}
    for c in conds:
        d=[(r['causal'][c]['logp'].float()-r['base_logp'].float()).numpy() for r in recs]
        kl=[r['causal'][c]['kl'].float().clamp_min(0).numpy() for r in recs];ag=[r['causal'][c]['agree'].float().numpy() for r in recs]
        out['by_bin'][c]={'dlogp':table([bin_means(x) for x in d]),'kl':table([bin_means(x) for x in kl]),'greedy_agreement':table([bin_means(x) for x in ag])}
    # direct (image keys blocked from k on) vs total (gray image) effect on tokens t >= k
    dt={}
    for k in (32,64,128):
        name=f'ko{k}';rs=[r for r in recs if r['T']>=k+16]
        if name not in conds or len(rs)<20:continue
        kl={c:np.array([float(r['causal'][c]['kl'][k:].float().clamp_min(0).mean()) for r in rs]) for c in (name,'ko0','gray','swap')}
        dt[f'tokens_t>={k}']={'n':len(rs),**{f'kl_{c}':_mci(list(v)) for c,v in kl.items()}}
        for c in (name,'ko0'):
            r_,ci_=ratio_ci(kl[c],kl['gray']);dt[f'tokens_t>={k}'][f'{c}_over_gray']={'ratio':r_,'ci95':ci_}
    out['direct_vs_total']=dt
    # object mentions
    om={}
    for r in recs:
        for nd,w,t,kind in r['mentions']:
            if t>=r['T']:continue
            for c in ('gray','swap','ko0','ko64'):
                if c in r['causal']:
                    om.setdefault(kind,{}).setdefault(c,[]).append((t,float(r['causal'][c]['logp'][t]-r['base_logp'][t])))
    out['mention_tokens_dlogp']={kind:{c:{'early_t<64':_mci([v for t,v in xs if t<64]),'late_t>=64':_mci([v for t,v in xs if t>=64])}
                                       for c,xs in d.items()} for kind,d in om.items()}
    return out

def headline(s):
    h={}
    b=s['E1_attention']['bands']['all']
    h['attention_image_share_first_vs_last_bin']=[b['image_share'][0],b['image_share'][-1]]
    h['attention_text_share_first_vs_last_bin']=[b['text_share'][0],b['text_share'][-1]]
    h['attention_image_share_late_minus_early']=b['image_share_late_minus_early']
    h['rollout_image_late_minus_early']=s['E1_attention']['rollout_absorbing_image']['image_late_minus_early']
    if 'window_limited_layers' in s['E1_attention']['bands']:
        w=s['E1_attention']['bands']['window_limited_layers']['image_keys_visible'];h['window_limited_layers_image_keys_visible_first_vs_last_bin']=[w[0],w[-1]]
    e=s['E2_lens_repr'];last=str(max(int(x) for x in e['at_probe_layers']['lens_image_contribution']))
    h['lens_image_contribution_final_layer_late_minus_early']=e['at_probe_layers']['lens_image_contribution_late_minus_early'][last]
    mid=str(e['probe_layers'][1]);pr=e['probes']['cohort_results'].get(mid,{})
    if pr:
        ks=sorted(pr,key=int);h[f'probe_image_gain_unmentioned_layer{mid}_cohort']={k:pr[k].get('image_gain_unmentioned') for k in (ks[0],ks[-1])}
    h['causal_direct_vs_total']={k:{c:v[c] for c in v if c.endswith('_over_gray')} for k,v in s['E3_causal']['direct_vs_total'].items()}
    return h

def main(a):
    # tag = qwen / gemma (stored captions) or qwen-regen / gemma-regen (--regenerate); never mix the two
    files=sorted(f for f in glob.glob(str(a.input/'*.pt')) if re.fullmatch(rf'{re.escape(a.model)}-\d+-\d+\.pt',Path(f).name))
    recs=[r for f in files for r in torch.load(f,weights_only=False)]
    assert recs,'no records';assert len({r['id'] for r in recs})==len(recs),'duplicate captions across shards'
    n_all=len(recs);recs=[r for r in recs if float(r.get('base_greedy_cfg',r['base_greedy']).float().mean())>=a.min_repro]
    print(f'{len(recs)} of {n_all} captions reproduce greedy decoding on >= {a.min_repro:.0%} of tokens')
    s={'model':a.model,'files':[Path(f).name for f in files],'captions':len(recs),'captions_before_reproduction_filter':n_all,'min_repro':a.min_repro,'tokens':int(sum(r['T'] for r in recs)),
       'caption_length':{'mean':float(np.mean([r['T'] for r in recs])),'max':int(max(r['T'] for r in recs))},
       'image_tokens':sorted({r['n_image_tokens'] for r in recs}),'prompt_length_before_caption':sorted({r['capstart'] for r in recs})[:5],
       'bins':LAB}
    s['E1_attention']=e1(recs);s['E2_lens_repr']=e2(recs);s['E3_causal']=e3(recs);s['headline']=headline(s)
    p=a.input/f'{a.model}-visual-decay-summary{a.suffix}.json';p.write_text(json.dumps(s,indent=1));print(json.dumps(s['headline'],indent=1));print('wrote',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('model',help='qwen, gemma, qwen-regen or gemma-regen')
    p.add_argument('--min-repro',type=float,default=0.0,help='keep captions whose tokens are the greedy choice at least this often (robustness check, e.g. 0.95)')
    p.add_argument('--suffix',default='',help='appended to the summary file name, e.g. -v2')
    p.add_argument('--probe-layer',type=int,default=None,help='probe only this depth index (0-3); run one process per depth in parallel')
    a=p.parse_args();ONLY_LAYER=a.probe_layer
    if ONLY_LAYER is not None:a.suffix=f'{a.suffix}-probe{ONLY_LAYER}'
    main(a)
