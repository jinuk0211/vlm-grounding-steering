"""Logit lens for the CHAIR captioning setup (Qwen2.5-VL-7B, Gemma 4 E4B).

Every hidden state is read the standard way: final norm -> unembedding (-> Gemma's final logit softcap)
-> softmax over the vocabulary. Layer 0 = embedding output, layer l = output of decoder layer l; at the
last layer this reproduces the model's own output, which is checked on every item.

image    Image-token lens (as in Jiang et al. 2024, Neo et al. 2024). One prefill per CHAIR image with the
         Base prompt. Image tokens come before any text, so their states do not depend on the caption.
         Per layer and per COCO object (80): the highest probability any image token gives to the object's
         word tokens (first token of each single-word CHAIR synonym, with/without a leading space,
         lower/capitalised), the mean of its 5 highest patches, and which patch gives the maximum.
         Plus the top-5 tokens of every patch at 25/50/75/100% depth, for looking at the patch grid.
text     Caption-position lens (nostalgebraist 2020). Teacher forcing on the Base caption. At every caption
         position and layer: log p and rank of the token that was written, the top-1 token, and KL from
         the final-layer distribution. Object mentions (CHAIR nodes, first token) are marked with their kind.
analyze  CPU only. Image lens, per layer, AUROC of the image-token confidence for
           presence      GT objects vs the other COCO objects (all 80 per image)
           hallucinated  correctly mentioned (positive) vs hallucinated objects
           missed        GT objects the caption did not mention (positive) vs absent, unmentioned objects
         pooled and within object (mean over objects with >= 5 of each class); 95% CI by image-cluster
         bootstrap at the best layer (chosen on the same data, so slightly optimistic); Spearman with the
         question-mode margin from caption-selfcheck. Text lens: top-1 rate / log p of the written token and
         KL to the final layer by caption position; at object mentions the layer from which the written
         object stays top-1, hallucinated vs correct, position-matched.
Writes only under --output.

  python scripts/logit_lens.py image   --model qwen --output DIR [--start S --end E]
  python scripts/logit_lens.py text    --model qwen --output DIR [--start S --end E]
  python scripts/logit_lens.py analyze --model qwen --output DIR      (writes {model}-lens-summary.json, .html)
"""
import argparse,glob,json,math,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

BINS=[0,16,32,64,96,128,192,256,384,513]
DEPTHS=(0.25,0.5,0.75,1.0)
KS=(1,5,10)
TOPK=100
TOPK_KS=(1,5,10,50,100)
BAND={'qwen':range(22,28),'gemma':range(30,42)}  # layers where written words start to appear in the lens
PB=[0,32,64,96,128,192,256,384,513]

# ------------------------------------------------------------------ collection
def load(a):
    import read_gap_collect_v2 as C
    from visual_decay_collect import CHAIR,Resid
    from vlm_grounding.backend import load_adapter,_first_path
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    if a.cpu_smoke:  # code-path check only; numbers are not comparable to the bf16 runs
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';config=dict(config,device='cpu',dtype='float32')
    else:assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    adapter=load_adapter(config);m=adapter.model
    for p in m.parameters():p.requires_grad_(False)
    norm=_first_path(m,('model.language_model.norm','language_model.model.norm','model.norm'));head=m.get_output_embeddings()
    cap=getattr(m.config.get_text_config(),'final_logit_softcapping',None)
    def lens(h):  # [n, D] -> [n, V] float logits
        lg=head(norm(h))  # softcap in the model's dtype, as the model does, so the last layer reproduces its output
        return (torch.tanh(lg/cap)*cap if cap else lg).float()
    return C,CHAIR,Resid,adapter,lens

def object_tokens(tok,scorer):
    """CHAIR object -> token ids that name it. Words: every single-word synonym, and the last word (head noun) of
    multi-word ones; spellings with/without a leading space, lower/capitalised. Only words that are a single token
    (>= 3 letters) count: word pieces such as "bicy" (bicyclist -> person) or "street" (streetlight) would fire on
    other things. A token claimed by several objects goes to the object for which it is a whole single-word synonym
    ("dog" -> dog, not hot dog); if that does not decide, it is dropped. Objects left without a token get an empty
    list (the lens skips them and lists them)."""
    syn={}
    for s,n in scorer.inverse_synonym_dict.items():
        if n.strip():syn.setdefault(n,set()).add(s)
    cand={}  # node -> {token: strength}; 2 = the token is a whole single-word synonym, 1 = head noun / word start
    for n,ss in syn.items():
        for s in ss:
            w=s.split()[-1]
            for v in (' '+w,' '+w.capitalize(),w,w.capitalize()):
                e=tok.encode(v,add_special_tokens=False);piece=tok.decode(e[:1]).strip() if e else ''
                if len(e)==1 and len(piece)>=3:
                    k=2 if len(e)==1 and ' ' not in s else 1;c=cand.setdefault(n,{});c[e[0]]=max(c.get(e[0],0),k)
    owners={}
    for n,ts in cand.items():
        for t,k in ts.items():owners.setdefault(t,[]).append((k,n))
    win={}
    for t,o in owners.items():
        top=max(k for k,_ in o);best=[n for k,n in o if k==top]
        if len(best)==1:win[t]=best[0]
    return sorted(syn),{n:sorted(t for t in cand.get(n,{}) if win.get(t)==n) for n in syn}

def image_labels(C,CHAIR,model):
    sents={s['image_id']:s for s in json.loads((CHAIR[model]/'base.chair.official.json').read_text())['sentences']}
    marg={}
    for r in C.read(C.ROOT/f'outputs/caption-selfcheck-20261030/{model}-base-rows.jsonl'):marg.setdefault(r['image_id'],{})[r['node']]=r['margin']
    def lab(iid):
        s=sents[iid]
        return {'gt':sorted(set(s['mscoco_gt_words'])),'mentioned':sorted(set(s['mscoco_generated_words'])),
                'hallucinated':sorted({n for _,n in s['mscoco_hallucinated_words']}),'margins':marg.get(iid,{})}
    return lab

def run(a,body):
    out=a.output;out.mkdir(parents=True,exist_ok=True);torch.manual_seed(42)
    tag=f'{a.model}-{a.mode}-{a.start}-{a.end}';status=out/f'{tag}-status.json';status.unlink(missing_ok=True)  # no stale fields from an earlier run
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),model=a.model,mode=a.mode,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),argv=sys.argv)
    try:
        res=body(a,update);torch.save(res,out/f'{tag}.pt');update(stage='complete',records=len(res['records']))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

def image_mode(a,update):
    C,CHAIR,Resid,adapter,lens=load(a)
    from score_vti_alpha_sweep import chair_scorer
    m=adapter.model;tok=adapter.processor.tokenizer;layers=adapter.layers;NL=len(layers);res=Resid(layers);dev=adapter.device
    allnodes,ids=object_tokens(tok,chair_scorer());nodes=[n for n in allnodes if ids[n]];untokenized=[n for n in allnodes if not ids[n]]
    cols=torch.tensor([t for n in nodes for t in ids[n]],device=dev);owner=torch.tensor([i for i,n in enumerate(nodes) for _ in ids[n]],device=dev)
    depths=sorted({round(f*NL) for f in DEPTHS});img_id=C.image_token_id(m);lab=image_labels(C,CHAIR,a.model)
    rows=C.read(CHAIR[a.model]/'base.jsonl')[a.start:a.end];recs=[];err=0.0;t0=time.time()
    update(stage='running',n_layers=NL,nodes=len(nodes),objects_without_token=untokenized,tokens_per_node=float(np.mean([len(ids[n]) for n in nodes])),depths=depths)
    for n,r in enumerate(rows):
        iid=r['metadata']['image_id'];b=adapter.prepare(r['image'],r['prompt']);img=b['input_ids'][0]==img_id;P=int(img.sum())
        g=b.get('image_grid_thw');grid=[int(g[0,1])//2,int(g[0,2])//2] if g is not None else ([int(math.isqrt(P))]*2 if math.isqrt(P)**2==P else None)
        with torch.no_grad():
            res.on=True
            try:o=m(**b,use_cache=False,return_dict=True)
            finally:res.on=False
            mx=[];t5=[];am=[];topk={};top1={}
            for l,h in enumerate(res.h):
                pr=lens(h[0][img] if h.dim()==3 else h[img]).softmax(-1)
                if l==NL:err=max(err,float((pr-o.logits[0,img].float().softmax(-1)).abs().max()))
                per=torch.zeros(P,len(nodes),device=dev).scatter_reduce(1,owner.expand(P,-1),pr[:,cols],'amax',include_self=False)
                v,i=per.max(0);mx.append(v);am.append(i);t5.append(per.topk(min(5,P),0).values.mean(0))
                if l in depths:k=pr.topk(5,-1).indices;topk[l]=k.int().cpu();top1[l]=[tok.decode([int(x)]) for x in k[:,0]]
                del pr,per
            del o
        res.h=[None]*(NL+1)
        recs.append({'id':r['id'],'image_id':iid,'labels':lab(iid),'n_image_tokens':P,'grid':grid,'max':torch.stack(mx).half().cpu(),
                     'top5mean':torch.stack(t5).half().cpu(),'argmax':torch.stack(am).short().cpu(),'topk':topk,'top1_str':top1})
        if dev.type=='cuda':torch.cuda.empty_cache()
        assert err<0.02,f'lens at the last layer differs from the model output by {err}'
        if n%20==0:update(done=n+1,sec_per_image=(time.time()-t0)/(n+1),last_layer_max_prob_err=err)
    return {'records':recs,'nodes':nodes,'objects_without_token':untokenized,'node_tokens':{n:[tok.decode([t]) for t in ids[n]] for n in nodes},
            'n_layers':NL,'depths':depths,'last_layer_max_prob_err':err}

def text_mode(a,update):
    C,CHAIR,Resid,adapter,lens=load(a)
    from score_vti_alpha_sweep import chair_scorer
    from verified_decoding_v3 import mention_token_positions
    m=adapter.model;tok=adapter.processor.tokenizer;layers=adapter.layers;NL=len(layers);res=Resid(layers);dev=adapter.device
    scorer=chair_scorer();lab=image_labels(C,CHAIR,a.model);rows=C.read(CHAIR[a.model]/'base.jsonl')[a.start:a.end];recs=[];err=0.0;t0=time.time()
    update(stage='running',n_layers=NL)
    for n,r in enumerate(rows):
        if not r['prediction'].strip():continue
        iid=r['metadata']['image_id'];b=adapter.prepare(r['image'],r['prompt'],answer=r['prediction']);lb=b.pop('labels')[0];ids=b['input_ids'][0]
        cs=int((lb!=-100).nonzero()[0]);q=torch.arange(cs-1,len(ids)-1,device=dev);tgt=ids[cs:];T=len(tgt)
        with torch.no_grad():
            res.on=True
            try:o=m(**b,use_cache=False,return_dict=True)
            finally:res.on=False
            out_lp=o.logits[0,q].float().log_softmax(-1).gather(-1,tgt[:,None])[:,0];del o
            H=[(h[0] if h.dim()==3 else h) for h in res.h];res.h=[None]*(NL+1)
            lp=torch.zeros(NL+1,T);rk=torch.zeros(NL+1,T,dtype=torch.int32);t1=torch.zeros(NL+1,T,dtype=torch.int32);kl=torch.zeros(NL+1,T)
            for c in range(0,T,64):
                s=slice(c,min(c+64,T));qq=q[s];tt=tgt[s];fin=lens(H[NL][qq]).log_softmax(-1);pf=fin.exp()
                for l in range(NL+1):
                    x=fin if l==NL else lens(H[l][qq]).log_softmax(-1);v=x.gather(-1,tt[:,None])
                    lp[l,s]=v[:,0].cpu();rk[l,s]=(x>v).sum(-1).int().cpu();t1[l,s]=x.argmax(-1).int().cpu();kl[l,s]=(pf*(fin-x)).sum(-1).cpu()
                del fin,pf
            err=max(err,float((lp[NL]-out_lp.cpu()).abs().max()));del H
        assert err<0.5,f'lens at the last layer differs from the model output by {err}'
        L=lab(iid);hall=set(L['hallucinated']);seen=set();ments=[]
        for nd,w,t in mention_token_positions(tok,tgt.tolist(),scorer):
            if t<T:ments.append({'node':nd,'word':w,'t':t,'first':nd not in seen,'kind':'hallucinated' if nd in hall else 'correct','margin':L['margins'].get(nd)});seen.add(nd)
        recs.append({'id':r['id'],'image_id':iid,'T':T,'logp':lp.half(),'rank':rk,'top1':t1,'kl':kl.half(),'mentions':ments})
        if dev.type=='cuda':torch.cuda.empty_cache()
        if n%20==0:update(done=n+1,sec_per_caption=(time.time()-t0)/(n+1),last_layer_max_logp_err=err)
    return {'records':recs,'n_layers':NL,'last_layer_max_logp_err':err}

def detect_mode(a,update):
    """How often does an object's word actually appear when the image tokens are read as words? Per layer and object:
    the number of patches whose top-1 / top-5 / top-10 lens tokens contain one of the object's tokens."""
    C,CHAIR,Resid,adapter,lens=load(a)
    from score_vti_alpha_sweep import chair_scorer
    m=adapter.model;tok=adapter.processor.tokenizer;layers=adapter.layers;NL=len(layers);res=Resid(layers);dev=adapter.device
    allnodes,ids=object_tokens(tok,chair_scorer());nodes=[n for n in allnodes if ids[n]];N=len(nodes)
    node_of=torch.full((m.get_output_embeddings().out_features,),N,device=dev)
    for i,n in enumerate(nodes):node_of[ids[n]]=i
    img_id=C.image_token_id(m);lab=image_labels(C,CHAIR,a.model);rows=C.read(CHAIR[a.model]/'base.jsonl')[a.start:a.end];recs=[];err=0.0;t0=time.time()
    update(stage='running',n_layers=NL,nodes=N)
    for n,r in enumerate(rows):
        iid=r['metadata']['image_id'];b=adapter.prepare(r['image'],r['prompt']);img=b['input_ids'][0]==img_id;P=int(img.sum())
        with torch.no_grad():
            res.on=True
            try:o=m(**b,use_cache=False,return_dict=True)
            finally:res.on=False
            cnt=torch.zeros(NL+1,len(KS),N,dtype=torch.int16)
            for l,h in enumerate(res.h):
                lg=lens(h[img])
                if l==NL:err=max(err,float((lg.softmax(-1)-o.logits[0,img].float().softmax(-1)).abs().max()))
                own=node_of[lg.topk(max(KS),-1).indices]
                for ki,k in enumerate(KS):
                    hit=torch.zeros(P,N+1,dtype=torch.bool,device=dev);hit.scatter_(1,own[:,:k],True);cnt[l,ki]=hit[:,:N].sum(0).short().cpu()
                del lg,own
            del o
        res.h=[None]*(NL+1);recs.append({'id':r['id'],'image_id':iid,'labels':lab(iid),'n_image_tokens':P,'count':cnt})
        if dev.type=='cuda':torch.cuda.empty_cache()
        assert err<0.02,f'lens at the last layer differs from the model output by {err}'
        if n%20==0:update(done=n+1,sec_per_image=(time.time()-t0)/(n+1),last_layer_max_prob_err=err)
    return {'records':recs,'nodes':nodes,'token_ids':{n:ids[n] for n in nodes},'n_layers':NL,'ks':KS,'last_layer_max_prob_err':err}

def topk_mode(a,update):
    """Caption-position lens, top-100: every object-word token among the top 100 lens tokens, per layer and position
    (columns: layer, caption position, rank, object index, token). Only these hits are saved, not the 100 words."""
    C,CHAIR,Resid,adapter,lens=load(a)
    from score_vti_alpha_sweep import chair_scorer
    from verified_decoding_v3 import mention_token_positions
    m=adapter.model;tok=adapter.processor.tokenizer;layers=adapter.layers;NL=len(layers);res=Resid(layers);dev=adapter.device
    scorer=chair_scorer();allnodes,ids=object_tokens(tok,scorer);nodes=[n for n in allnodes if ids[n]];N=len(nodes)
    node_of=torch.full((m.get_output_embeddings().out_features,),N,device=dev)
    for i,n in enumerate(nodes):node_of[ids[n]]=i
    lab=image_labels(C,CHAIR,a.model);rows=C.read(CHAIR[a.model]/'base.jsonl')[a.start:a.end];recs=[];err=0.0;t0=time.time()
    update(stage='running',n_layers=NL,nodes=N)
    for n,r in enumerate(rows):
        if not r['prediction'].strip():continue
        iid=r['metadata']['image_id'];b=adapter.prepare(r['image'],r['prompt'],answer=r['prediction']);lb=b.pop('labels')[0];tk=b['input_ids'][0]
        cs=int((lb!=-100).nonzero()[0]);q=torch.arange(cs-1,len(tk)-1,device=dev);tgt=tk[cs:];T=len(tgt);hits=[]
        with torch.no_grad():
            res.on=True
            try:o=m(**b,use_cache=False,return_dict=True)
            finally:res.on=False
            out_lp=o.logits[0,q].float().log_softmax(-1).gather(-1,tgt[:,None])[:,0];del o
            H=list(res.h);res.h=[None]*(NL+1)
            for c in range(0,T,64):
                qq=q[c:c+64]
                for l in range(NL+1):
                    lg=lens(H[l][qq]);top=lg.topk(TOPK,-1).indices;own=node_of[top];i,j=(own<N).nonzero(as_tuple=True)
                    hits.append(torch.stack([torch.full_like(i,l),i+c,j,own[i,j],top[i,j]],1).int().cpu())
                    if l==NL:err=max(err,float((lg.log_softmax(-1).gather(-1,tgt[c:c+64,None])[:,0]-out_lp[c:c+64]).abs().max()))
                    del lg,top,own
            del H
        assert err<0.5,f'lens at the last layer differs from the model output by {err}'
        L=lab(iid);hall=set(L['hallucinated']);seen=set();ments=[]
        for nd,w,t in mention_token_positions(tok,tgt.tolist(),scorer):
            if t<T:ments.append({'node':nd,'word':w,'t':t,'first':nd not in seen,'kind':'hallucinated' if nd in hall else 'correct'});seen.add(nd)
        recs.append({'id':r['id'],'image_id':iid,'T':T,'caption_ids':tgt.tolist(),'labels':L,'mentions':ments,'hits':torch.cat(hits).numpy()})
        if dev.type=='cuda':torch.cuda.empty_cache()
        if n%10==0:update(done=n+1,sec_per_caption=(time.time()-t0)/(n+1),last_layer_max_logp_err=err,max_gpu_gb=torch.cuda.max_memory_allocated()/2**30 if dev.type=='cuda' else None)
    return {'records':recs,'nodes':nodes,'n_layers':NL,'topk':TOPK,'last_layer_max_logp_err':err}

# ------------------------------------------------------------------ analysis
def auroc(s,y):
    s=np.asarray(s,float);y=np.asarray(y).astype(bool);ok=~np.isnan(s);s,y=s[ok],y[ok];p=int(y.sum());n=len(y)-p
    if p==0 or n==0:return float('nan')
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True);rk=(np.cumsum(cnt)-(cnt-1)/2)[inv]
    return float((rk[y].sum()-p*(p+1)/2)/(p*n))

def spearman(x,y):
    ok=~(np.isnan(x)|np.isnan(y));x,y=x[ok],y[ok]
    if len(x)<3:return float('nan')
    rx=np.argsort(np.argsort(x));ry=np.argsort(np.argsort(y));return float(np.corrcoef(rx,ry)[0,1])

def boot_ci(groups,fn,n=1000,seed=0):
    u=np.unique(groups);idx={g:np.where(groups==g)[0] for g in u};rng=np.random.default_rng(seed);v=[]
    for _ in range(n):
        x=fn(np.concatenate([idx[g] for g in rng.choice(u,len(u))]))
        if not np.isnan(x):v.append(x)
    return [float(np.quantile(v,.025)),float(np.quantile(v,.975))] if v else [None,None]

def within(score,y,node,min_each=5):
    a=[auroc(score[node==k],y[node==k]) for k in np.unique(node) if y[node==k].sum()>=min_each and (~y[node==k]).sum()>=min_each]
    return (float(np.nanmean(a)) if a else float('nan')),len(a)

def binof(t):return next((i for i in range(len(BINS)-1) if BINS[i]<=t<BINS[i+1]),len(BINS)-2)

def analyze_image(D):
    nodes=D[0]['nodes'];NL=D[0]['n_layers'];recs=[r for d in D for r in d['records']]
    S={'max':[],'top5mean':[]};g=[];nd=[];gt=[];men=[];hal=[];mg=[]
    for r in recs:
        L=r['labels'];GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
        for j,n in enumerate(nodes):
            for k in S:S[k].append(r[k][:,j].float().numpy())
            g.append(r['image_id']);nd.append(n);gt.append(n in GT);men.append(n in M);hal.append(n in H);mg.append(L['margins'].get(n,np.nan))
    S={k:np.stack(v) for k,v in S.items()};g=np.array(g);nd=np.array(nd);gt=np.array(gt);men=np.array(men);hal=np.array(hal);mg=np.array(mg,float)
    tasks={'presence':(np.ones(len(g),bool),gt),'hallucinated':(men,~hal),'missed':(~men,gt)}
    out={'images':len(recs),'pairs':int(len(g)),'n_layers':NL,'last_layer_max_prob_err':max(d['last_layer_max_prob_err'] for d in D),
         'objects':len(nodes),'objects_without_token':D[0].get('objects_without_token',[]),'node_tokens':D[0].get('node_tokens',{}),
         'class_counts':{t:{'positive':int(y[mk].sum()),'negative':int((~y[mk]).sum())} for t,(mk,y) in tasks.items()},
         'note':'AUROC > 0.5: higher image-token confidence for the positive class (present / correct / missed GT)','tasks':{}}
    for t,(mk,y) in tasks.items():
        res={}
        for k,sc in S.items():
            pooled=[auroc(sc[mk,l],y[mk]) for l in range(NL+1)];wn=[within(sc[mk,l],y[mk],nd[mk]) for l in range(NL+1)]
            bl=int(np.nanargmax(pooled));gi=g[mk];s_,y_=sc[mk,bl],y[mk]
            res[k]={'pooled':pooled,'within_object':[x[0] for x in wn],'objects_used_within':wn[bl][1],'best_layer':bl,'best_pooled':pooled[bl],
                    'best_ci95':boot_ci(gi,lambda ii:auroc(s_[ii],y_[ii])),'best_within_object':wn[bl][0]}
        out['tasks'][t]=res
    rows=~np.isnan(mg)
    out['spearman_with_question_margin']={k:[spearman(sc[rows,l],mg[rows]) for l in range(NL+1)] for k,sc in S.items()}
    out['question_margin_auroc_hallucinated']=auroc(mg[men&rows],(~hal)[men&rows])
    grids=[]
    for r in recs[:3]:
        l=sorted(r['top1_str'])[-2] if len(r['top1_str'])>1 else max(r['top1_str']);  # 75% depth: the last layer decodes next-token guesses
        s=[x.strip()[:8] or '_' for x in r['top1_str'][l]];w=r['grid'][1] if r['grid'] else 16
        grids.append({'image_id':r['image_id'],'layer':l,'gt':r['labels']['gt'],'hallucinated':r['labels']['hallucinated'],'rows':[' '.join(s[i:i+w]) for i in range(0,len(s),w)]})
    out['patch_top1_examples']=grids
    return out

def analyze_text(D):
    NL=D[0]['n_layers'];recs=[r for d in D for r in d['records']];nb=len(BINS)-1
    acc={k:[[[] for _ in range(nb)] for _ in range(NL+1)] for k in ('top1','logp','kl')}
    for r in recs:
        rk=r['rank'].numpy();lp=r['logp'].float().numpy();kl=r['kl'].float().numpy()
        for bi in range(nb):
            s=slice(BINS[bi],min(BINS[bi+1],r['T']))
            if s.start>=r['T']:continue
            for l in range(NL+1):
                acc['top1'][l][bi].append(float((rk[l,s]==0).mean()));acc['logp'][l][bi].append(float(lp[l,s].mean()));acc['kl'][l][bi].append(float(kl[l,s].mean()))
    lab=[f'{BINS[i]}-{BINS[i+1]-1}' for i in range(nb)]
    out={'captions':len(recs),'n_layers':NL,'last_layer_max_logp_err':max(d['last_layer_max_logp_err'] for d in D),'bins':lab,
         'by_position':{k:[[float(np.mean(c)) if c else None for c in row] for row in v] for k,v in acc.items()}}
    # object mentions: layer from which the written object stays top-1 (NL+1 = not top-1 at the output)
    E=[];h=[];b=[];g=[];LP=[];T1=[]
    for r in recs:
        rk=r['rank'].numpy();lp=r['logp'].float().numpy()
        for x in r['mentions']:
            if not x['first']:continue
            col=rk[:,x['t']]==0;e=NL+1
            for l in range(NL,-1,-1):
                if col[l]:e=l
                else:break
            E.append(e);h.append(x['kind']=='hallucinated');b.append(binof(x['t']));g.append(r['image_id']);LP.append(lp[:,x['t']]);T1.append(col)
    E=np.array(E,float);h=np.array(h);b=np.array(b);g=np.array(g);LP=np.stack(LP);T1=np.stack(T1)
    def strat(v,ii):
        d=[];w=[]
        for k in np.unique(b[ii]):
            s=ii[b[ii]==k];hh=v[s[h[s]]];cc=v[s[~h[s]]]
            if len(hh)>=3 and len(cc)>=3:d.append(hh.mean()-cc.mean());w.append(len(hh))
        return float(np.average(d,weights=w)) if d else float('nan')
    a=np.arange(len(E));top=E<=NL
    out['mentions']={'n':int(len(E)),'hallucinated':int(h.sum()),'top1_at_output':{'hallucinated':float(top[h].mean()),'correct':float(top[~h].mean())},
        'emergence_layer_mean (top-1 at output only)':{'hallucinated':float(E[h&top].mean()),'correct':float(E[~h&top].mean())},
        'emergence_hall_minus_correct_position_matched':strat(E,a[top]),
        'emergence_ci95':boot_ci(g[top],lambda ii:strat(E,a[top][ii])),
        'top1_rate_by_layer':{'hallucinated':T1[h].mean(0).tolist(),'correct':T1[~h].mean(0).tolist()},
        'logp_by_layer':{'hallucinated':LP[h].mean(0).tolist(),'correct':LP[~h].mean(0).tolist()}}
    return out

CLASSES=('correct','missed','hallucinated','absent')

def object_class(n,L):
    GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated'])
    return 'hallucinated' if n in H else 'correct' if n in M and n in GT else 'missed' if n in GT else 'absent' if n not in M else None

def within_diff(x,c,nd,a,b,min_each=5):
    """mean over objects seen at least min_each times in both classes: detection rate in a minus in b"""
    d=[x[(c==a)&(nd==k)].mean()-x[(c==b)&(nd==k)].mean() for k in np.unique(nd) if ((c==a)&(nd==k)).sum()>=min_each and ((c==b)&(nd==k)).sum()>=min_each]
    return (float(np.mean(d)) if d else None),len(d)

def detect_analyze(a):
    """Detection rates: image tokens (any patch) and caption positions (lens top-1 at any position of that caption)."""
    D=torch.load(a.output/f'{a.model}-detect-0-500.pt',weights_only=False);nodes=D['nodes'];NL=D['n_layers'];recs=D['records'];ids=D['token_ids']
    C=np.stack([r['count'].numpy() for r in recs]).transpose(0,3,1,2).reshape(-1,NL+1,len(KS))  # [pairs, layers, ks]
    nd=np.array([n for r in recs for n in nodes]);c=np.array([object_class(n,r['labels']) or 'other' for r in recs for n in nodes])
    pairs=(('correct','hallucinated'),('missed','absent'),('hallucinated','absent'))
    out={'model':a.model,'images':len(recs),'objects':len(nodes),'ks':list(KS),'class_counts':{k:int((c==k).sum()) for k in CLASSES},'image_tokens':{},'caption_positions':{}}
    for ki,k in enumerate(KS):
        x=C[:,:,ki]>0
        out['image_tokens'][f'top{k}']={'rate':{cl:x[c==cl].mean(0).tolist() for cl in CLASSES},
            'patches_when_detected':{cl:[float(C[:,l,ki][(c==cl)&x[:,l]].mean()) if ((c==cl)&x[:,l]).any() else None for l in range(NL+1)] for cl in CLASSES},
            'within_object':{f'{p}-{q}':[within_diff(x[:,l].astype(float),c,nd,p,q)[0] for l in range(NL+1)] for p,q in pairs},
            'objects_used':{f'{p}-{q}':within_diff(x[:,NL].astype(float),c,nd,p,q)[1] for p,q in pairs}}
    # caption positions, from the stored lens top-1 tokens. Written objects count only before their first mention (else trivial).
    T=torch.load(a.output/f'{a.model}-text-0-500.pt',weights_only=False)['records'];lab={r['image_id']:r['labels'] for r in recs};rows=[]
    for r in T:
        t1=r['top1'].numpy();first={}
        for x in r['mentions']:
            if x['first']:first[x['node']]=x['t']
        for n in nodes:
            cl=object_class(n,lab[r['image_id']])
            if cl is None:continue
            end=first.get(n,r['T']);hit=np.isin(t1[:,:end],ids[n]).any(1) if end>0 else np.zeros(NL+1,bool);rows.append((cl,n,hit))
    cc=np.array([x[0] for x in rows]);nn=np.array([x[1] for x in rows]);H=np.stack([x[2] for x in rows]).astype(float)
    out['caption_positions']={'note':'lens top-1 at any caption position; correct/hallucinated objects only before their first mention, missed/absent over the whole caption',
        'rate':{cl:H[cc==cl].mean(0).tolist() for cl in CLASSES},'within_object':{f'{p}-{q}':[within_diff(H[:,l],cc,nn,p,q)[0] for l in range(NL+1)] for p,q in (('missed','absent'),('correct','hallucinated'))}}
    p=a.output/f'{a.model}-detect-summary.json';p.write_text(json.dumps(out,indent=1))
    show=sorted({round(f*NL) for f in (0,.25,.5,.6,.75,.85,.9,.95,1.0)})
    line=lambda v,l:f"  L{l:2d} "+' '.join(f"{cl[:4]} {100*v['rate'][cl][l]:5.1f}%" for cl in CLASSES)+'  within: '+' '.join(f"{q} {'n/a' if w[l] is None else f'{100*w[l]:+.1f}'}" for q,w in v['within_object'].items())
    print(f"== {a.model}: {len(recs)} images, {len(nodes)} objects; rows: {out['class_counts']}; last-layer check {D['last_layer_max_prob_err']:.4f}")
    for k in KS:
        v=out['image_tokens'][f'top{k}'];print(f' image tokens, word in top-{k} of any patch')
        for l in show:print(line(v,l))
        print('   objects used',v['objects_used'])
    v=out['caption_positions'];print(' caption positions, word is lens top-1 somewhere')
    for l in show:print(line(v,l))
    print('wrote',p)

def topk_analyze(a):
    """Later in the caption, do hallucinated objects appear more among a middle-layer band's top-K lens words? Unit: one object
    at one caption position, hit if any band layer has its word in the top K. Hits equal to the input token or to the token
    written at that position are dropped. Rates are per object, so classes with few objects (hallucinated) and many
    (absent) compare directly; "excess" subtracts the same object's rate at the same position bin when it is absent and
    unwritten (chance hits, generic words). Written objects are split into before / after their first mention."""
    D=[torch.load(f,weights_only=False) for f in sorted(glob.glob(str(a.output/f'{a.model}-topk-*-*.pt')))];nodes=D[0]['nodes'];N=len(nodes);ni={n:i for i,n in enumerate(nodes)}
    recs=[r for d in D for r in d['records']];band=set(BAND[a.model]);nb=len(PB)-1;nk=len(TOPK_KS);assert len({r['id'] for r in recs})==len(recs)
    CL=('hall_before','hall_after','correct_before','correct_after','missed','absent')
    cnt=np.zeros((len(recs),nk,nb,N,2));pos=np.zeros((len(recs),nb,N,2));cls=np.full((len(recs),N),-1)
    for x,r in enumerate(recs):
        T=r['T'];c=r['caption_ids'];L=r['labels'];GT,M,H=set(L['gt']),set(L['mentioned']),set(L['hallucinated']);first=np.full(N,T+1)
        for m_ in r['mentions']:
            if m_['first'] and m_['node'] in ni:first[ni[m_['node']]]=m_['t']
        for n,i in ni.items():
            cls[x,i]=0 if n in H else 2 if n in M and n in GT else 4 if n in GT else 5 if n not in M else -1
        for b in range(nb):
            ts=np.arange(PB[b],min(PB[b+1],T))
            if len(ts):pos[x,b,:,0]=(ts[None]<first[:,None]).sum(1);pos[x,b,:,1]=(ts[None]>first[:,None]).sum(1)
        h=r['hits'];h=h[np.isin(h[:,0],list(band))];ca=np.array(c);keep=h[:,4]!=ca[h[:,1]];prev=np.r_[-1,ca[:-1]];h=h[keep&(h[:,4]!=prev[h[:,1]])]
        for ki,k in enumerate(TOPK_KS):
            u=np.unique(h[h[:,2]<k][:,[1,3]],axis=0) if len(h) else np.zeros((0,2),int)
            for t,i in u:
                if t==first[i]:continue
                cnt[x,ki,np.digitize(t,PB)-1,i,int(t>first[i])]+=1
    def rate(ii,cl,ki,bins):
        k=CL.index(cl);base=cls[ii]==(k//2*2 if k<4 else k);af=int(cl.endswith('after'));c=cnt[ii][:,ki][:,bins][...,af].sum(1);p=pos[ii][:,bins][...,af].sum(1)
        P=p[base].sum();return float(c[base].sum()/P) if P>0 else float('nan')
    base=np.array([[cnt[cls[:,i]==5,ki,b,i,0].sum()/max(pos[cls[:,i]==5,b,i,0].sum(),1) for i in range(N)] for ki in range(nk) for b in range(nb)]).reshape(nk,nb,N)
    def excess(ii,cl,ki,bins):
        k=CL.index(cl);m_=cls[ii]==(k//2*2 if k<4 else k);af=int(cl.endswith('after'));c=p=e=0.0
        for b in bins:
            pb=pos[ii][:,b,:,af]*m_;c+=(cnt[ii][:,ki,b,:,af]*m_).sum();p+=pb.sum();e+=(pb*base[ki,b]).sum()
        return float((c-e)/p) if p>0 else float('nan')
    allr=np.arange(len(recs));out={'model':a.model,'captions':len(recs),'objects':N,'band':[min(band),max(band)],'ks':list(TOPK_KS),'bins':[f'{x}-{y-1}' for x,y in zip(PB,PB[1:])],'rate':{},'excess':{}}
    for ki,k in enumerate(TOPK_KS):
        out['rate'][str(k)]={cl:[rate(allr,cl,ki,[b]) for b in range(nb)] for cl in CL};out['excess'][str(k)]={cl:[excess(allr,cl,ki,[b]) for b in range(nb)] for cl in CL}
    coh=np.array([i for i,r in enumerate(recs) if r['T']>=192],dtype=int);E,Lt=[0,1],[4,5,6,7];out['late_minus_early']={'cohort_captions':len(coh),'early_tokens':'0-63','late_tokens':'128+'}
    rng=np.random.default_rng(0)
    for ki,k in enumerate(TOPK_KS):
        for cl in ('hall_before','correct_before','missed','absent'):
            for name,f in (('rate',rate),('excess',excess)):
                if name=='excess' and cl=='absent':continue
                v=f(coh,cl,ki,Lt)-f(coh,cl,ki,E);bs=[]
                for _ in range(300):
                    ii=rng.choice(coh,len(coh));x=f(ii,cl,ki,Lt)-f(ii,cl,ki,E)
                    if x==x:bs.append(x)
                out['late_minus_early'][f'top{k}_{cl}_{name}']={'diff':v,'ci95':[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))]}
    p=a.output/f'{a.model}-topk-summary.json';p.write_text(json.dumps(out,indent=1))
    print(f"== {a.model}: {len(recs)} captions, {N} objects, band L{min(band)}-L{max(band)}; last-layer check {max(d['last_layer_max_logp_err'] for d in D):.4f}; hits saved {sum(len(r['hits']) for r in recs):,}")
    pc=lambda x:'  n/a' if x!=x else f'{100*x:5.2f}'
    for k in TOPK_KS:
        print(f' top-{k}: % of positions where one object of the class appears (raw) / excess over the same object when absent')
        for b in range(nb):
            print(f"  {out['bins'][b]:8s} "+'  '.join(f"{cl.replace('_before','<').replace('_after','>')} {pc(out['rate'][str(k)][cl][b])}/{pc(out['excess'][str(k)][cl][b])}" for cl in CL))
        print('  late-early (same captions): '+'; '.join(f"{cl} {100*out['late_minus_early'][f'top{k}_{cl}_rate']['diff']:+.2f} {[round(100*x,2) for x in out['late_minus_early'][f'top{k}_{cl}_rate']['ci95']]}" for cl in ('hall_before','correct_before','missed','absent')))
        print('  excess late-early: '+'; '.join(f"{cl} {100*out['late_minus_early'][f'top{k}_{cl}_excess']['diff']:+.2f} {[round(100*x,2) for x in out['late_minus_early'][f'top{k}_{cl}_excess']['ci95']]}" for cl in ('hall_before','correct_before','missed')))
    print('wrote',p)

def report(path,model,im,tx):
    import html
    from visual_decay_plot import CSS,line_chart
    nn=lambda v:[None if x is None or (isinstance(x,float) and math.isnan(x)) else x for x in v]
    cards=[]
    if im:
        x=[str(l) for l in range(im['n_layers']+1)]
        for t,v in im['tasks'].items():
            s=[(f'{k} pooled',nn(v[k]['pooled']),None) for k in ('max','top5mean')]+[('max within object',nn(v['max']['within_object']),None)]
            cards.append(line_chart(f'Image-token lens: {t}',f"AUROC per layer; best {v['max']['best_pooled']:.3f} at layer {v['max']['best_layer']}",x,s,('layer','AUROC')))
        cards.append(line_chart('Image-token confidence vs question margin','Spearman per layer (hallucinated / correct / missed objects)',x,
                                [(k,nn(v),None) for k,v in im['spearman_with_question_margin'].items()],('layer','Spearman'),zero=True))
    if tx:
        x=[str(l) for l in range(tx['n_layers']+1)];bp=tx['by_position']
        for key,name in (('top1','Written token is top-1'),('kl','KL from the final layer')):
            cards.append(line_chart(name,'caption-position lens, first vs last position bins',x,
                                    [(tx['bins'][i],nn([row[i] for row in bp[key]]),None) for i in (0,3,6,len(tx['bins'])-1)],('layer',key)))
        mt=tx['mentions']
        cards.append(line_chart('Object mentions: written object is top-1','first mention of each object',x,
                                [(k,v,None) for k,v in mt['top1_rate_by_layer'].items()],('layer','top-1 rate')))
    path.write_text(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Logit lens: {html.escape(model)}</title><style>{CSS}</style></head>'
                    f'<body class="viz-root"><h1>Logit lens: {html.escape(model)}</h1><div class="row">{"".join(cards)}</div></body></html>',encoding='utf-8')

def analyze(a):
    out={'model':a.model};im=tx=None
    fi=sorted(f for f in glob.glob(str(a.output/f'{a.model}-image-*-*.pt')));ft=sorted(glob.glob(str(a.output/f'{a.model}-text-*-*.pt')))
    if fi:im=analyze_image([torch.load(f,weights_only=False) for f in fi]);out['image_lens']=im
    if ft:tx=analyze_text([torch.load(f,weights_only=False) for f in ft]);out['text_lens']=tx
    assert fi or ft,'nothing to analyze'
    p=a.output/f'{a.model}-lens-summary.json';p.write_text(json.dumps(out,indent=1));report(a.output/f'{a.model}-lens-report.html',a.model,im,tx)
    if im:
        print(f"== image-token lens, {a.model}: {im['images']} images, last-layer check {im['last_layer_max_prob_err']:.4f}")
        for t,v in im['tasks'].items():
            m_=v['max'];print(f"  {t:12s} best layer {m_['best_layer']:2d}  pooled AUROC {m_['best_pooled']:.3f} {m_['best_ci95']}  within object {m_['best_within_object']:.3f} ({m_['objects_used_within']} objects)")
        sp=im['spearman_with_question_margin']['max'];print(f"  Spearman with question margin: max {np.nanmax(sp):.3f} at layer {int(np.nanargmax(sp))}; question margin itself, hallucinated task AUROC {im['question_margin_auroc_hallucinated']:.3f}")
        for e in im['patch_top1_examples'][:1]:print(f"  patch top-1 at layer {e['layer']}, image {e['image_id']} (GT {e['gt']}, hallucinated {e['hallucinated']}):\n    "+'\n    '.join(e['rows']))
    if tx:
        mt=tx['mentions'];print(f"== caption-position lens, {a.model}: {tx['captions']} captions, last-layer check {tx['last_layer_max_logp_err']:.4f}")
        print(f"  mentions {mt['n']} ({mt['hallucinated']} hallucinated); top-1 at output: hall {mt['top1_at_output']['hallucinated']:.2f} correct {mt['top1_at_output']['correct']:.2f}")
        print(f"  layer from which the object stays top-1: hall {mt['emergence_layer_mean (top-1 at output only)']['hallucinated']:.1f} correct {mt['emergence_layer_mean (top-1 at output only)']['correct']:.1f}; "
              f"position-matched difference {mt['emergence_hall_minus_correct_position_matched']:+.2f} {mt['emergence_ci95']}")
    print('wrote',p)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['image','text','analyze','detect','detect-analyze','topk','topk-analyze']);p.add_argument('--model',choices=['qwen','gemma'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=500)
    p.add_argument('--cpu-smoke',action='store_true',help='float32 on CPU, for checking the code path only')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='analyze':analyze(a)
    elif a.mode=='detect-analyze':detect_analyze(a)
    elif a.mode=='topk-analyze':topk_analyze(a)
    else:run(a,{'image':image_mode,'text':text_mode,'detect':detect_mode,'topk':topk_mode}[a.mode])
