"""Self-verified decoding for captions (CHAIR protocol: 500 images, greedy, 512 tokens).

Greedy decoding as in Base. When a word boundary completes a COCO object
mention (detected with the unmodified CHAIR scorer's caption_to_words) whose
object node has not been verified yet, the same model is asked the POPE-style
question "Is there a {node} in the image?\nPlease answer yes or no." If the
Yes-No margin is below tau, decoding rolls back to the first token of that
mention, bans that token at that position, and continues greedily. Verdicts are
cached per (image, node). tau=-inf is the restart-only control (never rejects).
"""
import argparse,json,math,os,time,traceback
from pathlib import Path
import torch
from transformers import LogitsProcessor,LogitsProcessorList,StoppingCriteria,StoppingCriteriaList
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from score_vti_alpha_sweep import chair_scorer

MAX_BANS=20

class Ban(LogitsProcessor):
    def __init__(self,start,ids):self.start=start;self.ids=list(ids)
    def __call__(self,input_ids,scores):
        if input_ids.shape[1]==self.start and self.ids:scores[:,self.ids]=-float('inf')
        return scores

class StopAtPending(StoppingCriteria):
    """Stop right after a word boundary if the text so far mentions an unaccepted node."""
    def __init__(self,base_len,tok,scorer,accepted,key):self.base_len=base_len;self.tok=tok;self.scorer=scorer;self.accepted=accepted;self.key=key
    def __call__(self,input_ids,scores,**kw):
        ids=input_ids[0,self.base_len:].tolist();flag=False
        if len(ids)>=2:
            last=self.tok.decode(ids[-1:],skip_special_tokens=True)
            if last and not last[0].isalnum():
                w,nodes,_,_=self.scorer.caption_to_words(self.tok.decode(ids[:-1],skip_special_tokens=True))
                flag=any(n not in self.accepted for n in (w if self.key=='word' else nodes))
        return torch.tensor([flag],device=input_ids.device)

def mention_token_positions(tok,ids,scorer):
    """(node, token index where the mention starts) for every mention, in order."""
    text=tok.decode(ids,skip_special_tokens=True);low=text.lower()
    import nltk
    words=nltk.word_tokenize(low)
    # char offset of each nltk token by sequential search
    offs=[];p=0
    for w in words:
        q=low.find(w,p)
        if q<0:q=p
        offs.append(q);p=q+len(w)
    mwords,nodes,idxs,_=scorer.caption_to_words(text)
    # cumulative decoded length per token
    cum=[0]
    for i in range(1,len(ids)+1):cum.append(len(tok.decode(ids[:i],skip_special_tokens=True)))
    out=[]
    for n,mw,k in zip(nodes,mwords,idxs):
        c=offs[k] if k<len(offs) else len(text)
        t=next((i for i in range(len(ids)) if cum[i]<=c<cum[i+1]),len(ids)-1)
        out.append((n,mw,t))
    return out

def extend(batch,ids):
    if not ids:return dict(batch)
    L=batch['input_ids'].shape[1];add=torch.tensor([ids],device=batch['input_ids'].device,dtype=batch['input_ids'].dtype)
    out={}
    for k,v in batch.items():
        if k=='input_ids':out[k]=torch.cat([v,add],1)
        elif k=='attention_mask':out[k]=torch.cat([v,torch.ones_like(add,dtype=v.dtype)],1)
        elif torch.is_tensor(v) and v.dim()==2 and v.shape==(1,L):out[k]=torch.cat([v,torch.zeros_like(add,dtype=v.dtype)],1)
        else:out[k]=v
    return out

@torch.no_grad()
def verify(adapter,variants,image,node,img_id):
    art='an' if node[0] in 'aeiou' else 'a'
    logits,_,_,_=C.run(adapter,image,f"Is there {art} {node} in the image?\nPlease answer yes or no.",img_id)
    return float(torch.logsumexp(logits[variants['yes']],0)-torch.logsumexp(logits[variants['no']],0))

@torch.no_grad()
def caption(adapter,scorer,variants,img_id,ref,tau,eos,key='node'):
    tok=adapter.processor.tokenizer;batch=adapter.prepare(ref['image'],ref['prompt'])
    base_len=batch['input_ids'].shape[1];budget=ref['max_new_tokens']
    ids=[];accepted=set();rejected={};margins={};bans={};log=[];calls=0;forced=0
    while True:
        inputs={k:v for k,v in extend(batch,ids).items() if k!='labels'}
        calls+=1
        out=adapter.model.generate(**inputs,do_sample=False,num_beams=1,use_cache=True,max_new_tokens=budget-len(ids),
            logits_processor=LogitsProcessorList([Ban(inputs['input_ids'].shape[1],bans.get(len(ids),()))]),
            stopping_criteria=StoppingCriteriaList([StopAtPending(base_len,tok,scorer,accepted,key)]))
        ids=out[0,base_len:].tolist()
        ended=(ids and ids[-1] in eos) or len(ids)>=budget
        roll=None
        for nd,mw,t in mention_token_positions(tok,ids,scorer):
            node=mw if key=='word' else nd
            if node in accepted:continue
            if node not in margins:margins[node]=verify(adapter,variants,ref['image'],node,img_id)
            if margins[node]>=tau or sum(len(b) for b in bans.values())>=MAX_BANS:
                if margins[node]<tau:forced+=1
                accepted.add(node);continue
            roll=(node,t);break
        if roll is None:
            if ended:break
            continue
        node,t=roll;rejected[node]=rejected.get(node,0)+1
        bans.setdefault(t,set()).add(ids[t]);log.append({'node':node,'pos':t,'banned':ids[t],'word':tok.decode([ids[t]])})
        ids=ids[:t]
    text=tok.decode(ids,skip_special_tokens=True)
    return text,len(ids),{'verified_margins':margins,'accepted':sorted(accepted),'rejections':log,'generate_calls':calls,'forced_accepts':forced}

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    refs=C.read(a.data or C.MODELS[a.model]/'data/chair.jsonl');assert a.data or len(refs)==500;a.end=min(a.end,len(refs))
    name=f'{a.tag}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    prefix=C.read(rows_file) if rows_file.exists() else []
    for r,i in zip(prefix,range(a.start,a.end)):assert r['id']==refs[i]['id']
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),model=a.model,tau=a.tau,key=a.key,data=str(a.data),range=[a.start,a.end],resumed_from=len(prefix),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);img_id=C.image_token_id(adapter.model);scorer=chair_scorer()
        import caption_selfcheck as S;variants=S.variants(adapter)
        eos=adapter.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        update(stage='running',gpu_name=torch.cuda.get_device_name(0),variants=variants,eos=sorted(eos))
        with rows_file.open('a') as f:
            for i in range(a.start+len(prefix),a.end):
                ref=refs[i];t0=time.perf_counter()
                text,n,info=caption(adapter,scorer,variants,img_id,ref,a.tau,eos,a.key)
                f.write(json.dumps({**ref,'method':f'selfcheck_{a.key}_tau{a.tau}','verify_key':a.key,'prediction':text,'generated_tokens':n,
                    'elapsed_seconds':time.perf_counter()-t0,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'tau':a.tau,
                    'dtype':config['dtype'],'checkpoint':config['checkpoint'],'revision':config['revision'],**info})+'\n');f.flush()
                update(done=i-a.start+1,last=i)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(C.MODELS),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--tau',type=float,default=0.0);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=500)
    p.add_argument('--key',choices=['node','word'],default='node');p.add_argument('--data',type=Path)
    main(p.parse_args())
