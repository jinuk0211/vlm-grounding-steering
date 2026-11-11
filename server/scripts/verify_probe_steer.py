"""Verified-object logit steering on CHAIR (500 images).

ask    For every image, ask the model about all 80 COCO objects ("Is there a {node} in the image?\nPlease
       answer yes or no.", the caption-selfcheck prompt), batched; margin = logsumexp(Yes variants) -
       logsumexp(No variants) at the answer position.  -> {model}-ask.jsonl
gen    Greedy captioning (Base prompt, 512 tokens, the model's own generation config) with one logits
       processor; no rollback, no re-asking. Lists are fixed before the run (no tuning):
         whitelist  margin > +2   boost: +delta on the first token of each single-word CHAIR synonym of a
                                  whitelisted object that has not been written yet, from generated token 64 on
         blacklist  margin < -2   suppress: -delta on the same tokens from the first generated token
       Tokens shared by a whitelisted and a blacklisted object, or shorter than 3 characters, are left alone.
       Conditions: base (delta 0, same batched pipeline), boost_D, both_D.
summarize  official CHAIR scorer; CHAIR_s, CHAIR_i, recall, object F1, tokens; paired bootstrap vs the
       pipeline base; stored Base / REVIS / RUDDER / GrAInS / self-check rows for reference; list quality.
Writes only under --output.
"""
import argparse,glob,json,math,os,re,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
TAU_YES,TAU_NO,START=2.0,-2.0,64

def setup():
    sys.path.insert(0,str(ROOT/'scripts'))
    import read_gap_collect_v2 as C
    from score_vti_alpha_sweep import chair_scorer
    return C,chair_scorer

def load_image(path,size):
    with Image.open(path) as s:im=ImageOps.exif_transpose(s).convert('RGB')
    return im.resize((size,size),Image.Resampling.BICUBIC) if size else im

def batch_inputs(adapter,images,prompts):
    proc=adapter.processor;proc.tokenizer.padding_side='left'
    texts=[proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':p}]}],tokenize=False,add_generation_prompt=True) for p in prompts]
    ims=[load_image(i,adapter.image_size) for i in images]
    ims=[[im] for im in ims] if adapter.model_type=='gemma4' else ims
    b=proc(text=texts,images=ims,padding=True,return_tensors='pt')
    return {k:v.to(adapter.device,dtype=adapter.dtype if v.is_floating_point() else v.dtype) for k,v in b.items() if torch.is_tensor(v)}

def nodes_and_tokens(tok,scorer):
    """80 COCO objects (CHAIR's empty synonym line excluded) and the tokens that name them; same rule as the
    logit lens (head noun of multi-word synonyms, whole word or >= 4-letter word start, no shared tokens).
    Objects without such a token are still asked about but cannot be steered."""
    from logit_lens import object_tokens
    return object_tokens(tok,scorer)

Q=lambda x:f"Is there a {x} in the image?\nPlease answer yes or no."

def ask_slow(adapter,image,nodes,v,bs):
    mg={}
    for c in range(0,len(nodes),bs):
        ns=nodes[c:c+bs];b=batch_inputs(adapter,[image]*len(ns),[Q(x) for x in ns])
        lg=adapter.model(**b,use_cache=False,return_dict=True).logits[:,-1].float()
        mg.update({x:float(y) for x,y in zip(ns,torch.logsumexp(lg[:,v['yes']],-1)-torch.logsumexp(lg[:,v['no']],-1))})
    return mg

def ask_fast(adapter,image,nodes,v):
    """image + chat prefix once (KV cache), then all questions as one batch continuing from that cache."""
    proc=adapter.processor;tok=proc.tokenizer;m=adapter.model
    raw=lambda x:proc.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':Q(x)}]}],tokenize=False,add_generation_prompt=True)
    full=batch_inputs(adapter,[image],[Q(nodes[0])]);F=full['input_ids'][0].tolist()
    tail=lambda x:tok.encode(raw(x)[raw(x).index('Is there a'):],add_special_tokens=False)
    t0=tail(nodes[0]);assert F[-len(t0):]==t0,'suffix tokenization differs from the joint one';P=len(F)-len(t0)
    pre={k:(v_[:,:P] if k in('input_ids','attention_mask','token_type_ids','mm_token_type_ids') else v_) for k,v_ in full.items()}
    o=m(**pre,use_cache=True,return_dict=True);cache=o.past_key_values
    tails=[tail(x) for x in nodes];S=max(map(len,tails));n=len(nodes);pad=tok.pad_token_id
    ids=torch.full((n,S),pad,dtype=torch.long,device=adapter.device);am=torch.zeros((n,P+S),dtype=torch.long,device=adapter.device);am[:,:P]=1
    for i,t in enumerate(tails):ids[i,:len(t)]=torch.tensor(t);am[i,P:P+len(t)]=1
    cache.batch_repeat_interleave(n);kw={}
    if adapter.model_type.startswith('qwen'):  # its cached-decoding path builds positions over the whole mask; give the 3-D positions directly
        kw['position_ids']=(torch.arange(P,P+S,device=adapter.device)[None,None,:]+m.model.rope_deltas.view(1,-1,1)).expand(3,n,S)
    lg=m(input_ids=ids,attention_mask=am,past_key_values=cache,cache_position=torch.arange(P,P+S,device=adapter.device),use_cache=True,return_dict=True,**kw).logits.float()
    last=torch.tensor([len(t)-1 for t in tails],device=adapter.device);lg=lg[torch.arange(n,device=adapter.device),last]
    return {x:float(y) for x,y in zip(nodes,torch.logsumexp(lg[:,v['yes']],-1)-torch.logsumexp(lg[:,v['no']],-1))}

def ask(a):
    C,chair_scorer=setup();from vlm_grounding.backend import load_adapter;import caption_selfcheck as S
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model'];refs=C.read(C.MODELS[a.model]/'data/chair.jsonl')[a.start:a.end]
    adapter=load_adapter(config);v=S.variants(adapter);nodes,_=nodes_and_tokens(adapter.processor.tokenizer,chair_scorer())
    out=a.output/f'{a.model}-ask-{a.start}-{a.end}.jsonl';t0=time.time()
    with torch.no_grad():
        # the fast path must reproduce the plain batched forward on the first image before it is used
        s=ask_slow(adapter,refs[0]['image'],nodes,v,a.ask_batch);f_=ask_fast(adapter,refs[0]['image'],nodes,v)
        d=np.array([f_[x]-s[x] for x in nodes]);agree=float(np.mean([(f_[x]>0)==(s[x]>0) for x in nodes]))
        print(f'fast vs slow on image 0: max|diff| {np.abs(d).max():.3f} mean|diff| {np.abs(d).mean():.3f} sign agreement {agree:.3f}',flush=True)
        assert np.abs(d).max()<1.0 and agree>=0.95,'prefix-cache path does not reproduce the full forward'
    with out.open('w') as f, torch.no_grad():
        for n,r in enumerate(refs):
            mg=ask_fast(adapter,r['image'],nodes,v)
            f.write(json.dumps({'id':r['id'],'image_id':r['metadata']['image_id'],'margins':mg})+'\n');f.flush()
            if n%20==0:print(f'{n+1}/{len(refs)} {(time.time()-t0)/(n+1):.2f}s/image',flush=True)

class Steer:
    def __init__(self,rows_tokens,delta,boost,suppress,prompt_len,device):
        self.d=delta;self.boost=boost;self.sup=suppress;self.P=prompt_len;self.rows=rows_tokens;self.dev=device
        self.mentioned=[set() for _ in rows_tokens]
    def __call__(self,input_ids,scores):
        step=input_ids.shape[1]-self.P
        for b,(wl,bl,tok2node) in enumerate(self.rows):
            if step>0:
                n=tok2node.get(int(input_ids[b,-1]))
                if n:self.mentioned[b].add(n)
            if self.sup and bl:scores[b,bl]-=self.d
            if self.boost and step>=START:
                ids=[t for node,ts in wl.items() if node not in self.mentioned[b] for t in ts]
                if ids:scores[b,ids]+=self.d
        return scores

def lists(margins,ids):
    wl={n for n,m in margins.items() if m>TAU_YES and n in ids};bl={n for n,m in margins.items() if m<TAU_NO and n in ids}  # ask files from before the fix hold an empty-name object
    wt={t for n in wl for t in ids[n]};bt={t for n in bl for t in ids[n]};both=wt&bt
    W={n:[t for t in ids[n] if t not in both] for n in wl};B=sorted(bt-both)
    t2n={t:n for n in wl for t in ids[n] if t not in both}
    return W,B,t2n,sorted(wl),sorted(bl)

def gen(a):
    C,chair_scorer=setup();from vlm_grounding.backend import load_adapter;from transformers import LogitsProcessorList
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model'];refs=C.read(C.MODELS[a.model]/'data/chair.jsonl')
    asks={}
    for f in glob.glob(str(a.output/f'{a.model}-ask-*.jsonl')):
        for r in C.read(f):asks[r['id']]=r['margins']
    refs=refs[a.start:a.end];assert all(r['id'] in asks for r in refs),'ask incomplete';adapter=load_adapter(config);tok=adapter.processor.tokenizer;_,ids=nodes_and_tokens(tok,chair_scorer())
    kind,delta=(a.cond.split('_')[0],float(a.cond.split('_')[1])) if a.cond!='base' else ('base',0.0)
    out=a.output/f'{a.model}-{a.cond}-{a.start}-{a.end}.jsonl';status=a.output/f'{a.model}-{a.cond}-{a.start}-{a.end}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='running',pid=os.getpid(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'));t0=time.time()
    try:
        with out.open('w') as f:
            for c in range(0,len(refs),a.batch):
                rb=refs[c:c+a.batch];b=batch_inputs(adapter,[r['image'] for r in rb],[r['prompt'] for r in rb]);P=b['input_ids'].shape[1]
                L=[lists(asks[r['id']],ids) for r in rb];st=Steer([(x[0],x[1],x[2]) for x in L],delta,kind in('boost','both'),kind=='both',P,adapter.device)
                tb=time.perf_counter()
                with torch.no_grad():
                    o=adapter.model.generate(**b,do_sample=False,num_beams=1,max_new_tokens=512,use_cache=True,
                                             logits_processor=LogitsProcessorList([st]) if kind!='base' else None)
                new=o[:,P:];pad=tok.pad_token_id;el=(time.perf_counter()-tb)/len(rb)
                texts=tok.batch_decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                for r,x,txt,nw in zip(rb,L,texts,new):
                    f.write(json.dumps({'id':r['id'],'metadata':r['metadata'],'image':r['image'],'prompt':r['prompt'],'method':a.cond,'prediction':txt,
                        'generated_tokens':int((nw!=pad).sum()),'elapsed_seconds':el,'whitelist':x[3],'blacklist':x[4]})+'\n')
                f.flush();update(done=c+len(rb),sec_per_caption=(time.time()-t0)/(c+len(rb)))
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

def summarize(a):
    C,chair_scorer=setup();from score_verified_v2 import stats,agg,boot,FINAL
    refs=C.read(C.MODELS[a.model]/'data/chair.jsonl')[a.start:a.end];order=[r['metadata']['image_id'] for r in refs];scorer=chair_scorer()  # subset: same images for every row
    def f1(x):p=1-x['CHAIR_i'];return 2*p*x['recall']/(p+x['recall'])
    conds={}
    for f in glob.glob(str(a.output/f'{a.model}-*-*-*.jsonl')):
        if '-ask-' in f:continue
        for r in C.read(f):conds.setdefault(r['method'],{})[r['id']]=r
    res={}
    def score(name,rows):
        p=a.output/f'{a.model}-{name}-{a.start}-{a.end}-captions.json';p.write_text(json.dumps([{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in rows]))
        d=scorer.compute_chair(str(p),'image_id','caption');(a.output/f'{a.model}-{name}-{a.start}-{a.end}-chair.official.json').write_text(json.dumps(d));return stats(d,order)
    S={}
    for name,rows in conds.items():
        if any(r['id'] not in rows for r in refs):print(name,'incomplete',len(rows));continue
        rows=[rows[r['id']] for r in refs];S[name]=score(name,rows)
        res[name]={**agg(S[name]),'mean_tokens':float(np.mean([r['generated_tokens'] for r in rows])),'at_cap':int(sum(r['generated_tokens']>=512 for r in rows))}
    for m in ('base','revis','rudder_clone','grains_clone'):
        d=json.loads((FINAL[a.model]/f'{m}.chair.official.json').read_text());S['stored_'+m]=stats(d,order);res['stored_'+m]=agg(S['stored_'+m])
    sc=ROOT/f'outputs/verified-decoding-20261030/{a.model}-tau0-score/chair.official.json'
    if sc.exists():S['self_check_tau0']=stats(json.loads(sc.read_text()),order);res['self_check_tau0']=agg(S['self_check_tau0'])
    for k,v in res.items():
        v['object_F1']=f1(v)
        if 'base' in S and k!='base':v['vs_pipeline_base_CI95']=boot(S['base'],S[k])
    # list quality against the CHAIR ground truth
    asks={r['id']:r['margins'] for f in glob.glob(str(a.output/f'{a.model}-ask-*.jsonl')) for r in C.read(f)}
    wl_tp=wl_n=bl_gt=bl_n=gt_n=gt_in_wl=0
    for r in refs:
        gt=set(scorer.imid_to_objects[r['metadata']['image_id']]);m=asks[r['id']]
        wl={n for n,x in m.items() if x>TAU_YES};bl={n for n,x in m.items() if x<TAU_NO}
        wl_tp+=len(wl&gt);wl_n+=len(wl);bl_gt+=len(bl&gt);bl_n+=len(bl);gt_n+=len(gt);gt_in_wl+=len(gt&wl)
    lq={'whitelist_per_image':wl_n/len(refs),'whitelist_precision':wl_tp/max(wl_n,1),'gt_recall_of_whitelist':gt_in_wl/max(gt_n,1),
        'blacklist_per_image':bl_n/len(refs),'gt_objects_blacklisted_share':bl_gt/max(gt_n,1)}
    out={'model':a.model,'tau_yes':TAU_YES,'tau_no':TAU_NO,'boost_from_token':START,'list_quality':lq,'results':res}
    (a.output/f'{a.model}-steer-summary-{a.start}-{a.end}.json').write_text(json.dumps(out,indent=1))
    print(f"== {a.model}: whitelist {lq['whitelist_per_image']:.1f}/image precision {lq['whitelist_precision']:.3f} covers {100*lq['gt_recall_of_whitelist']:.1f}% of GT; "
          f"blacklist {lq['blacklist_per_image']:.1f}/image holds {100*lq['gt_objects_blacklisted_share']:.1f}% of GT")
    for k,v in sorted(res.items(),key=lambda x:(not x[0].startswith('stored'),x[0])):
        ci=v.get('vs_pipeline_base_CI95',{})
        print(f"  {k:18s} CHs {100*v['CHAIR_s']:5.1f} CHi {100*v['CHAIR_i']:5.2f} rec {100*v['recall']:5.1f} F1 {100*v['object_F1']:5.1f}"
              +(f" tok {v['mean_tokens']:4.0f}" if 'mean_tokens' in v else '          ')
              +(f"  dCHs [{100*ci['CHAIR_s'][0]:+.1f},{100*ci['CHAIR_s'][1]:+.1f}] drec [{100*ci['recall'][0]:+.1f},{100*ci['recall'][1]:+.1f}]" if ci else ''))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['ask','gen','summarize']);p.add_argument('--model',choices=['qwen','gemma'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=500)
    p.add_argument('--cond',default='base');p.add_argument('--batch',type=int,default=16);p.add_argument('--ask-batch',type=int,default=20)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    {'ask':ask,'gen':gen,'summarize':summarize}[a.mode](a)
