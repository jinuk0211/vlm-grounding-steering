"""E1: at the query that predicts an object mention, does an intervention lower log p of hallucinated
objects while leaving correct ones alone?

Teacher forcing on the same 100 Base captions as E0' ({model}-n100.pt). Every intervention acts at
the query position of every labelled mention at once (what a position-gated decoder would do):
  probe_away_k / probe_toward_k  h_L <- h_L -/+ k*sigma*u at the output of decoder layer L. u = unit direction
        that raises the T1 probe logit ("asked, the model would deny this object"), fitted on the E0'
        captions that are NOT in the 100 (no overlap); L = best T1 layer by CV on those captions;
        sigma = std of h_L.u there.
  random_k     the same shift along a fixed random unit vector (specificity control)
  reread_b_S   +b on the attention logits from that query to every image key, layer set S (all / late half)
Per mention: d log p of the mention's first token vs no intervention, top-1 flip (argmax before, not after),
KL at that position; per caption: mean KL on the other caption positions (collateral).

  --fit       fit u on {model}-n*-all.pt minus the 100 test captions -> {model}-e1-direction.pt
  --conds     run a subset of conditions -> {model}-e1-{tag}.pt
  --summarize merge -> {model}-e1-summary.json
Writes only under --output.
"""
import argparse,glob,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
KS=(2,4,8,16,32);BETAS=(1.0,2.0,3.0)

def all_conditions():
    c=['base']+[f'probe_away_{k}' for k in KS]+[f'probe_toward_{k}' for k in KS]+[f'random_{k}' for k in (4,8,16,32)]
    return c+[f'reread_{b:g}_{s}' for b in BETAS for s in ('all','late')]

# ------------------------------------------------------------------ direction
def fit_direction(a):
    import verify_probe_analyze as V
    test=torch.load(a.output/f'{a.model}-n100.pt',weights_only=False);test_ids={m['image_id'] for m in test['meta']}
    f=sorted(glob.glob(str(a.output/f'{a.model}-n*-all.pt')));assert len(f)==1,f
    d=torch.load(f[0],weights_only=False);NL=d['n_layers']
    keep=[i for i,m in enumerate(d['meta']) if m['first_mention'] and m['image_id'] not in test_ids]
    M=[d['meta'][i] for i in keep];H=d['h_real'][keep][:,:,0]  # [n, layers+1, D]
    y=np.array([[m['margin']<0] for m in M],float);g=np.array([m['image_id'] for m in M])
    S,chosen=V.cv(H,y,g);curve=[V.auroc(S[l,:,0],y[:,0]) for l in range(NL+1)];L=int(np.nanargmax(curve[1:]))+1
    lam=float(np.median([c[L][0] for c in chosen]))
    X=H[:,L].float().to(V.DEV);mu=X.mean(0);sd=X.std(0)+1e-4;Xs=(X-mu)/sd
    _,_,Vh=torch.linalg.svd(Xs,full_matrices=False);P=Vh[:min(256,len(X)-1)].T;Z=Xs@P;s=Z.std(0)+1e-4
    keepg=V.GRID;V.GRID=torch.tensor([lam]);W,b=V.fit((Z/s)[None],torch.from_numpy(y).float().to(V.DEV));V.GRID=keepg
    dvec=(P@(W[0,:,0,0]/s))/sd;u=(dvec/dvec.norm()).float()
    proj=(X@u);sigma=float(proj.std())
    # transfer check on the 100 test captions (first mentions), never used for fitting
    tk=[i for i,m in enumerate(test['meta']) if m['first_mention']];ty=np.array([test['meta'][i]['margin']<0 for i in tk])
    tp=(test['h_real'][tk][:,L,0].float().to(V.DEV)@u).cpu().numpy()
    gen=torch.Generator().manual_seed(0);r=torch.randn(u.shape[0],generator=gen);r=r/r.norm()
    out={'layer':L,'cv_auroc_curve':curve,'cv_auroc_at_layer':curve[L],'l2':lam,'sigma':sigma,'n_train':len(M),'train_images':int(len(np.unique(g))),
         'test_auroc_projection':V.auroc(tp,ty),'h_norm_mean':float(X.norm(dim=-1).mean()),'test_image_overlap':0}
    torch.save({'u':u.cpu(),'random':r,**out},a.output/f'{a.model}-e1-direction.pt');print(json.dumps(out|{'cv_auroc_curve':[round(x,3) for x in curve]},indent=1))

# ------------------------------------------------------------------ interventions
class Intervene:
    def __init__(self,module,layers):
        self.layers=layers;self.idx={id(l.self_attn):i for i,l in enumerate(layers)};self.module=module
        self.orig=module.eager_attention_forward;module.eager_attention_forward=self.attn
        self.add=None;self.boost=None;self.rows=None;self.img=None;self.handle=None
    def attn(self,mod,query,key,value,attention_mask,*args,**kw):
        i=self.idx.get(id(mod))
        if i is not None and self.boost is not None and i in self.boost[1]:
            Lq,Lk=query.shape[2],key.shape[2];assert Lq==Lk
            if attention_mask is None:attention_mask=torch.zeros(1,1,Lq,Lk,dtype=query.dtype,device=query.device)
            attention_mask=attention_mask.clone(memory_format=torch.contiguous_format)
            attention_mask[:,:,self.rows[:,None],self.img[None,:]]+=self.boost[0]
        return self.orig(mod,query,key,value,attention_mask,*args,**kw)
    def set_add(self,L,delta):
        if self.handle:self.handle.remove();self.handle=None
        if delta is None:return
        def hook(mod,args,out):
            h=(out[0] if isinstance(out,tuple) else out).clone();h[0,self.rows]+=delta.to(h.dtype)
            return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h
        self.handle=self.layers[L-1].register_forward_hook(hook)

def run(a):
    sys.path.insert(0,str(ROOT/'scripts'))
    from vlm_grounding.backend import load_adapter
    import read_gap_collect_v2 as C
    from verified_decoding_v3 import mention_token_positions
    from score_vti_alpha_sweep import chair_scorer
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    conds=a.conds.split(',');assert all(c in all_conditions() for c in conds),conds
    D=torch.load(a.output/f'{a.model}-e1-direction.pt',weights_only=False);L=D['layer']
    test=torch.load(a.output/f'{a.model}-n100.pt',weights_only=False);ids100=sorted({m['image_id'] for m in test['meta']})
    labels={(r['image_id'],r['node']):r for r in C.read(ROOT/f'outputs/caption-selfcheck-20261030/{a.model}-base-rows.jsonl') if r['kind']!='missed'}
    CH={'qwen':C.MODELS['qwen']/'original-vti-gpu1-final/chair','gemma':C.MODELS['gemma']/'chair'}
    rows=[r for r in C.read(CH[a.model]/'base.jsonl') if r['metadata']['image_id'] in set(ids100)];assert len(rows)==100
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    tag=a.tag;status=a.output/f'{a.model}-e1-{tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),conds=conds,layer=L,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer;layers=adapter.layers;NL=len(layers)
        iv=Intervene(sys.modules[type(layers[0].self_attn).__module__],layers);scorer=chair_scorer();img_id=C.image_token_id(m)
        u=D['u'].to(adapter.device);rnd=D['random'].to(adapter.device);sig=D['sigma']
        def setup(c):
            iv.boost=None;iv.set_add(L,None)
            if c.startswith('probe_away_'):iv.set_add(L,-int(c.split('_')[-1])*sig*u)
            elif c.startswith('probe_toward_'):iv.set_add(L,int(c.split('_')[-1])*sig*u)
            elif c.startswith('random_'):iv.set_add(L,int(c.split('_')[-1])*sig*rnd)
            elif c.startswith('reread_'):
                _,b,s=c.split('_');iv.boost=(float(b),set(range(NL)) if s=='all' else set(range(NL//2,NL)))
        update(stage='running',gpu_name=torch.cuda.get_device_name(0),sigma=sig)
        res={c:[] for c in conds};t0=time.time()
        for n,r in enumerate(rows):
            iid=r['metadata']['image_id']
            br=adapter.prepare(r['image'],r['prompt'],answer=r['prediction']);lab=br.pop('labels')[0]
            ids=br['input_ids'][0];capstart=int((lab!=-100).nonzero()[0]);T=len(ids)-capstart;capids=ids[capstart:].tolist()
            ments=[(nd,w,t) for nd,w,t in mention_token_positions(tok,capids,scorer) if t<T and (iid,nd) in labels]
            if not ments:continue
            seen=set();info=[]
            for nd,w,t in ments:info.append((nd,w,t,nd not in seen,labels[(iid,nd)]['kind'],labels[(iid,nd)]['margin']));seen.add(nd)
            q=torch.arange(capstart-1,len(ids)-1,device=adapter.device);tgt=ids[capstart:]
            mp=torch.tensor([t for _,_,t,_,_,_ in info],device=adapter.device)
            iv.rows=capstart-1+mp;iv.img=(ids==img_id).nonzero()[:,0]
            other=torch.ones(T,dtype=torch.bool,device=adapter.device);other[mp]=False;other[:int(mp.min())]=False
            with torch.no_grad():
                setup('base');lb=m(**br,use_cache=False,return_dict=True).logits[0,q].float().log_softmax(-1)
                for c in conds:
                    setup(c)
                    try:lp=lb if c=='base' else m(**br,use_cache=False,return_dict=True).logits[0,q].float().log_softmax(-1)
                    finally:setup('base')
                    kl=(lb.exp()*(lb-lp)).sum(-1)
                    res[c].append({'image_id':iid,'mentions':[{'node':nd,'word':w,'t':t,'first':f,'kind':k,'margin':mg,
                        'logp_base':float(lb[t,tgt[t]]),'logp':float(lp[t,tgt[t]]),'top1_base':bool(lb[t].argmax()==tgt[t]),'top1':bool(lp[t].argmax()==tgt[t]),
                        'kl':float(kl[t])} for nd,w,t,f,k,mg in info],'kl_other_mean':float(kl[other].mean()) if other.any() else None})
                    del lp
            del lb;torch.cuda.empty_cache()
            if n%10==0:update(done=n+1,sec_per_caption=(time.time()-t0)/(n+1))
        torch.save({'layer':L,'sigma':sig,'results':res},a.output/f'{a.model}-e1-{tag}.pt');update(stage='complete',done=len(rows))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

# ------------------------------------------------------------------ summary
def boot(groups,fn,n=1000,seed=0):
    u=np.unique(groups);idx={g:np.where(groups==g)[0] for g in u};rng=np.random.default_rng(seed);v=[]
    for _ in range(n):
        ii=np.concatenate([idx[g] for g in rng.choice(u,len(u))]);x=fn(ii)
        if not np.isnan(x):v.append(x)
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))]

def summarize(a):
    D=torch.load(a.output/f'{a.model}-e1-direction.pt',weights_only=False)
    res={}
    for f in sorted(glob.glob(str(a.output/f'{a.model}-e1-*.pt'))):
        if f.endswith('-direction.pt'):continue
        res.update(torch.load(f,weights_only=False)['results'])
    out={'model':a.model,'direction':{k:v for k,v in D.items() if k not in ('u','random','cv_auroc_curve')},'unit':'first mention of each (image, object)','conditions':{}}
    for c in [x for x in all_conditions() if x in res]:
        rows=[(r['image_id'],mm,r['kl_other_mean']) for r in res[c] for mm in r['mentions'] if mm['first']]
        g=np.array([x[0] for x in rows]);hall=np.array([x[1]['kind']=='hallucinated' for x in rows])
        d=np.array([x[1]['logp']-x[1]['logp_base'] for x in rows]);t1b=np.array([x[1]['top1_base'] for x in rows]);t1=np.array([x[1]['top1'] for x in rows])
        kl=np.array([x[1]['kl'] for x in rows]);oth=np.array([r['kl_other_mean'] for r in res[c] if r['kl_other_mean'] is not None])
        mh=lambda ii,msk:float(np.mean(d[ii][msk[ii]])) if msk[ii].any() else np.nan
        flip=lambda ii,msk:float(np.mean(~t1[ii][msk[ii]&t1b[ii]])) if (msk[ii]&t1b[ii]).any() else np.nan
        all_=np.arange(len(d))
        out['conditions'][c]={'n_hall':int(hall.sum()),'n_correct':int((~hall).sum()),
            'dlogp_hall':mh(all_,hall),'dlogp_correct':mh(all_,~hall),
            'selectivity_hall_minus_correct':mh(all_,hall)-mh(all_,~hall),'selectivity_ci95':boot(g,lambda ii:mh(ii,hall)-mh(ii,~hall)),
            'top1_lost_hall':flip(all_,hall),'top1_lost_correct':flip(all_,~hall),
            'top1_lost_hall_minus_correct_ci95':boot(g,lambda ii:flip(ii,hall)-flip(ii,~hall)),
            'kl_at_mentions':float(kl.mean()),'kl_other_tokens':float(oth.mean()) if len(oth) else None}
    p=a.output/f'{a.model}-e1-summary.json';p.write_text(json.dumps(out,indent=1))
    print(f"{a.model}: layer {D['layer']} sigma {D['sigma']:.2f} |h| {D['h_norm_mean']:.1f} train-CV AUROC {D['cv_auroc_at_layer']:.3f} test projection AUROC {D['test_auroc_projection']:.3f}")
    print(f"{'condition':22s} {'dlogp hall':>10s} {'correct':>8s} {'select':>7s} {'95% CI':>16s} {'top1 lost hall':>14s} {'correct':>8s} {'KL other':>8s}")
    for c,v in out['conditions'].items():
        print(f"{c:22s} {v['dlogp_hall']:10.3f} {v['dlogp_correct']:8.3f} {v['selectivity_hall_minus_correct']:7.3f} [{v['selectivity_ci95'][0]:6.3f},{v['selectivity_ci95'][1]:6.3f}] "
              f"{v['top1_lost_hall']:14.3f} {v['top1_lost_correct']:8.3f} {v['kl_other_tokens'] or 0:8.4f}")
    print('wrote',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','gemma'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit',action='store_true');p.add_argument('--summarize',action='store_true');p.add_argument('--conds');p.add_argument('--tag')
    a=p.parse_args()
    if a.fit:fit_direction(a)
    elif a.summarize:summarize(a)
    else:run(a)
