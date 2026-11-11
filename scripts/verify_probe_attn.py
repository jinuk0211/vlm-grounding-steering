"""Step 0 for query-anchored image attention: is the image attention right before a hallucinated
object shaped differently from the one right before a correct object at the same caption position?

Teacher forcing on Base CHAIR captions (all captions with a labelled mention). At every text layer
and head, from the query that predicts each mention's first token (and, head-averaged, from every
caption query), the attention restricted to the image keys:
  mass     share of the row's attention that lands on image keys
  entropy  entropy of the attention within the image keys / log(#image keys)
  top1pct  share of the image attention on the row's own top 1% image keys
  sink     share of the image attention on this caption's "sink" image keys: the top 1% by attention
           received, averaged over heads and all caption queries of that layer
Rows whose image keys are all masked (Gemma sliding-window layers past 512 positions) are NaN.
Labels: caption-selfcheck-20261030 (kind, margin). Unit in the analysis: first mention of each (image, object).

Decision rule, fixed before the run. Main metrics: head mean, then mean over the main layers
(Qwen: all; Gemma: its full-attention layers), of entropy and of sink share.
  difference  position-stratified hallucinated - correct difference has a 95% CI (cluster bootstrap over
              images) that excludes 0, AND the position-adjusted AUROC of that metric is <= 0.45 or >= 0.55,
              for entropy or sink share -> go to step 1
  otherwise   no difference -> drop the design

  collect:    --model M --output DIR --start S --end E
  summarize:  --model M --output DIR --summarize
"""
import argparse,glob,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
METRICS=('mass','entropy','top1pct','sink')
BINS=[0,16,32,64,96,128,192,256,384,513]

class Tap:
    def __init__(self,module,attns):
        self.idx={id(a):i for i,a in enumerate(attns)};self.orig=module.eager_attention_forward;module.eager_attention_forward=self.forward
        self.on=False;self.calls=0
    def forward(self,mod,query,key,value,attention_mask,*args,**kw):
        out,w=self.orig(mod,query,key,value,attention_mask,*args,**kw)
        i=self.idx.get(id(mod))
        if i is not None and self.on:self.calls+=1;self.reduce(i,w)
        return out,w
    def reduce(self,i,w):
        A=w[0][:,self.rows][:,:,self.img].float()  # [H, T, N]
        N=A.shape[-1];k=max(1,math.ceil(0.01*N))
        mass=A.sum(-1);ok=mass>1e-6;P=A/mass.clamp_min(1e-12)[...,None]
        ent=-(P*P.clamp_min(1e-12).log()).sum(-1)/math.log(N);top=P.topk(k,-1).values.sum(-1)
        recv=(A*ok[...,None]).sum((0,1))/ok.sum().clamp_min(1);sink=recv.topk(k).indices;sk=P[...,sink].sum(-1)
        M=torch.stack([mass,ent,top,sk])  # [4, H, T]
        M[1:][:,~ok]=float('nan')
        self.ment[i]=M[:,:,self.mrows].permute(1,0,2).half().cpu()  # [H, 4, M]
        self.pos[i]=M.nanmean(1).half().cpu() if hasattr(torch,'nanmean') else M.mean(1).half().cpu()  # [4, T]

def collect(a):
    sys.path.insert(0,str(ROOT/'scripts'))
    from vlm_grounding.backend import load_adapter
    import read_gap_collect_v2 as C
    from verified_decoding_v3 import mention_token_positions
    from score_vti_alpha_sweep import chair_scorer
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42);out=a.output;out.mkdir(parents=True,exist_ok=True)
    labels={(r['image_id'],r['node']):r for r in C.read(ROOT/f'outputs/caption-selfcheck-20261030/{a.model}-base-rows.jsonl') if r['kind']!='missed'}
    CH={'qwen':C.MODELS['qwen']/'original-vti-gpu1-final/chair','gemma':C.MODELS['gemma']/'chair'}
    ids_l={i for i,_ in labels};rows=[r for r in C.read(CH[a.model]/'base.jsonl') if r['metadata']['image_id'] in ids_l][a.start:a.end]
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    tag=f'{a.model}-attn-{a.start}-{a.end}';status=out/f'{tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),captions=len(rows),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer;layers=adapter.layers;NL=len(layers)
        tap=Tap(sys.modules[type(layers[0].self_attn).__module__],[l.self_attn for l in layers]);scorer=chair_scorer();img_id=C.image_token_id(m)
        lt=getattr(m.config.get_text_config(),'layer_types',None);full=[i for i,t in enumerate(lt) if t=='full_attention'] if lt else list(range(NL))
        update(stage='running',gpu_name=torch.cuda.get_device_name(0),n_layers=NL,full_attention_layers=full)
        recs=[];t0=time.time()
        for n,r in enumerate(rows):
            iid=r['metadata']['image_id'];br=adapter.prepare(r['image'],r['prompt'],answer=r['prediction']);lab=br.pop('labels')[0]
            ids=br['input_ids'][0];capstart=int((lab!=-100).nonzero()[0]);T=len(ids)-capstart;capids=ids[capstart:].tolist()
            ments=[(nd,w,t) for nd,w,t in mention_token_positions(tok,capids,scorer) if t<T and (iid,nd) in labels]
            if not ments:continue
            seen=set();info=[]
            for nd,w,t in ments:lb=labels[(iid,nd)];info.append({'node':nd,'word':w,'t':t,'first':nd not in seen,'kind':lb['kind'],'margin':lb['margin']});seen.add(nd)
            tap.rows=torch.arange(capstart-1,len(ids)-1,device=adapter.device);tap.img=(ids==img_id).nonzero()[:,0]
            tap.mrows=torch.tensor([x['t'] for x in info],device=adapter.device);tap.ment=[None]*NL;tap.pos=[None]*NL;tap.calls=0
            with torch.no_grad():
                tap.on=True
                try:m(**br,use_cache=False,return_dict=True)
                finally:tap.on=False
            assert tap.calls==NL,(tap.calls,NL)
            recs.append({'image_id':iid,'T':T,'n_image':int(len(tap.img)),'mentions':info,'ment':torch.stack(tap.ment),'pos':torch.stack(tap.pos)})  # [L,H,4,M], [L,4,T]
            torch.cuda.empty_cache()
            if n%20==0:update(done=n+1,sec_per_caption=(time.time()-t0)/(n+1))
        torch.save({'records':recs,'n_layers':NL,'full_attention_layers':full,'metrics':METRICS},out/f'{tag}.pt');update(stage='complete',done=len(rows),records=len(recs))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

# ------------------------------------------------------------------ analysis
def auroc(s,y):
    s=np.asarray(s,float);y=np.asarray(y).astype(bool);ok=~np.isnan(s);s,y=s[ok],y[ok];p=int(y.sum());n=len(y)-p
    if p==0 or n==0:return float('nan')
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True);rk=(np.cumsum(cnt)-(cnt-1)/2)[inv]
    return float((rk[y].sum()-p*(p+1)/2)/(p*n))

def binof(t):return next(i for i in range(len(BINS)-1) if BINS[i]<=t<BINS[i+1]) if t<BINS[-1] else len(BINS)-2

def strat(v,h,b,ii):
    d=[];w=[]
    for k in np.unique(b[ii]):
        s=ii[b[ii]==k];hh=s[h[s]];cc=s[~h[s]];hv=v[hh][~np.isnan(v[hh])];cv=v[cc][~np.isnan(v[cc])]
        if len(hv)>=3 and len(cv)>=3:d.append(hv.mean()-cv.mean());w.append(len(hv))
    return float(np.average(d,weights=w)) if d else float('nan')

def adj_auroc(v,h,b):
    r=v.copy()
    for k in np.unique(b):s=b==k;r[s]=v[s]-np.nanmean(v[s])
    return auroc(r,h)

def test(v,h,b,g,n=1000,seed=0):
    u=np.unique(g);idx={x:np.where(g==x)[0] for x in u};rng=np.random.default_rng(seed);a=np.arange(len(v));bs=[]
    for _ in range(n):
        ii=np.concatenate([idx[x] for x in rng.choice(u,len(u))]);x=strat(v,h,b,ii)
        if not np.isnan(x):bs.append(x)
    return {'hall_minus_correct':strat(v,h,b,a),'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
            'auroc_hall_high_pos_adjusted':adj_auroc(v,h,b),'hall_mean':float(np.nanmean(v[h])),'correct_mean':float(np.nanmean(v[~h]))}

def summarize(a):
    files=sorted(glob.glob(str(a.output/f'{a.model}-attn-*-*.pt')));D=[torch.load(f,weights_only=False) for f in files]
    recs=[r for d in D for r in d['records']];NL=D[0]['n_layers'];full=D[0]['full_attention_layers']
    main=full if a.model=='gemma' else list(range(NL))
    V=[];h=[];t=[];g=[];head=[]
    for r in recs:
        for j,x in enumerate(r['mentions']):
            if not x['first']:continue
            mj=r['ment'][:,:,:,j].float()  # [L,H,4]
            V.append(torch.nanmean(mj,1).numpy() if hasattr(torch,'nanmean') else mj.mean(1).numpy());head.append(mj.numpy())
            h.append(x['kind']=='hallucinated');t.append(x['t']);g.append(r['image_id'])
    V=np.stack(V);head=np.stack(head);h=np.array(h);b=np.array([binof(x) for x in t]);g=np.array(g)  # V [n,L,4]
    out={'model':a.model,'captions':len(recs),'first_mentions':int(len(h)),'hallucinated':int(h.sum()),'main_layers':main,
         'note':'AUROC > 0.5: the metric is higher before hallucinated objects; adjusted = metric minus its position-bin mean'}
    out['main']={mt:test(np.nanmean(V[:,main,k],1),h,b,g) for k,mt in enumerate(METRICS)}
    out['per_layer']={mt:[test(V[:,l,k],h,b,g,n=300) for l in range(NL)] for k,mt in enumerate(METRICS)}
    # heads: position-adjusted AUROC for every (layer, head); selection over many heads, so descriptive only
    hl={}
    for k,mt in enumerate(METRICS[1:],1):
        A=np.array([[adj_auroc(head[:,l,hh,k],h,b) for hh in range(head.shape[2])] for l in range(NL)]);top=np.argsort(-np.abs(np.nan_to_num(A-0.5)).ravel())[:10]
        hl[mt]={'top10_by_abs_auroc_minus_half':[{'layer':int(i//A.shape[1]),'head':int(i%A.shape[1]),'auroc':float(A.ravel()[i])} for i in top],
                'share_heads_auroc_outside_0.45_0.55':float(np.mean(np.abs(np.nan_to_num(A-0.5))>=0.05))}
    out['heads']=hl
    # position trend over all caption queries (head mean, main layers)
    pos={mt:[[] for _ in BINS[:-1]] for mt in METRICS}
    for r in recs:
        P=r['pos'].float().numpy()  # [L,4,T]
        for k,mt in enumerate(METRICS):
            x=np.nanmean(P[main,k],0)
            for bi in range(len(BINS)-1):
                s=x[BINS[bi]:BINS[bi+1]]
                if len(s) and not np.all(np.isnan(s)):pos[mt][bi].append(np.nanmean(s))
    out['position_trend']={mt:[{'bin':f'{BINS[i]}-{BINS[i+1]-1}','mean':float(np.mean(v)),'captions':len(v)} for i,v in enumerate(pos[mt]) if v] for mt in METRICS}
    hit=[mt for mt in ('entropy','sink') if (out['main'][mt]['ci95'][0]>0 or out['main'][mt]['ci95'][1]<0) and abs(out['main'][mt]['auroc_hall_high_pos_adjusted']-0.5)>=0.05]
    out['verdict']=('difference in '+', '.join(hit)+' -> step 1') if hit else 'no difference in entropy or sink share -> drop the design'
    p=a.output/f'{a.model}-attn-summary.json';p.write_text(json.dumps(out,indent=1))
    print(f"== {a.model}: {out['first_mentions']} first mentions ({out['hallucinated']} hallucinated), {out['captions']} captions, main layers {len(main)}")
    for mt,v in out['main'].items():print(f"  {mt:8s} hall {v['hall_mean']:.4f} correct {v['correct_mean']:.4f}  pos-matched diff {v['hall_minus_correct']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  adj AUROC {v['auroc_hall_high_pos_adjusted']:.3f}")
    for mt in METRICS:print(f"  trend {mt:8s} "+' '.join(f"{x['bin']}:{x['mean']:.3f}" for x in out['position_trend'][mt]))
    for mt,v in hl.items():print(f"  heads {mt}: {100*v['share_heads_auroc_outside_0.45_0.55']:.0f}% outside 0.45-0.55; top "+', '.join(f"L{x['layer']}H{x['head']} {x['auroc']:.2f}" for x in v['top10_by_abs_auroc_minus_half'][:5]))
    print('  verdict:',out['verdict']);print('wrote',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','gemma'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=10**6);p.add_argument('--summarize',action='store_true')
    a=p.parse_args();summarize(a) if a.summarize else collect(a)
