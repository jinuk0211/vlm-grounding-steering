"""E0': when the caption writes an object, does the hidden state already say "asked, I would deny it"?

Teacher forcing on Base CHAIR captions (Qwen's include repetition_penalty 1.05, as generated),
once with the real image and once with a gray 448x448 image under the same caption.
For every object mention (CHAIR node, mapped to its first caption token as in
verified_decoding_v3) store, at every layer (0 = embedding output, l = output of decoder layer l):
  pos 0  the query that predicts the mention's first token (where decoding could intervene)
  pos 1  the mention's first token itself
plus log p of that token (real / gray), mention order among distinct objects, caption position.
Labels come from caption-selfcheck-20261030: kind (hallucinated / correct) and margin
(Yes-No when the same model is asked "Is there a {node} in the image?").
Default subset: --n captions drawn (seed 0) from those with >= 1 hallucinated object.
Writes only under --output.
"""
import argparse,json,os,random,sys,time,traceback
from pathlib import Path
import torch
from PIL import Image
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from verified_decoding_v3 import mention_token_positions
from score_vti_alpha_sweep import chair_scorer
from visual_decay_collect import Resid,CHAIR

LABELS=C.ROOT/'outputs/caption-selfcheck-20261030'

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    gray=out/'gray448.png'
    if not gray.exists():Image.new('RGB',(448,448),(128,128,128)).save(gray)
    labels={(r['image_id'],r['node']):r for r in C.read(LABELS/f'{a.model}-base-rows.jsonl') if r['kind']!='missed'}
    pool=sorted({i for (i,_),r in labels.items() if r['kind']=='hallucinated'} if not a.all_images else {i for i,_ in labels})
    pick=set(random.Random(a.seed).sample(pool,a.n)) if 0<a.n<len(pool) else set(pool)
    rows=[r for r in C.read(CHAIR[a.model]/'base.jsonl') if r['metadata']['image_id'] in pick];assert len(rows)==len(pick)
    tag=f'{a.model}-n{len(rows)}'+('-all' if a.all_images else '');status=out/f'{tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),model=a.model,captions=len(rows),pool=len(pool),seed=a.seed,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],argv=sys.argv)
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer
        layers=adapter.layers;NL=len(layers);res=Resid(layers);scorer=chair_scorer()
        update(stage='running',gpu_name=torch.cuda.get_device_name(0),n_layers=NL)
        meta=[];HR=[];HG=[];unmatched=[];t0=time.time()
        for n,r in enumerate(rows):
            iid=r['metadata']['image_id'];pred=r['prediction']
            br=adapter.prepare(r['image'],r['prompt'],answer=pred);bg=adapter.prepare(str(gray),r['prompt'],answer=pred)
            assert torch.equal(br['input_ids'],bg['input_ids']);bg.pop('labels')
            ids=br['input_ids'][0];lab=br.pop('labels')[0];capstart=int((lab!=-100).nonzero()[0]);L=len(ids);T=L-capstart
            capids=ids[capstart:].tolist()
            ments=[(nd,w,t) for nd,w,t in mention_token_positions(tok,capids,scorer) if t<T]
            order={}
            for nd,_,_ in ments:order.setdefault(nd,len(order))
            seen=set();items=[]
            for k,(nd,w,t) in enumerate(ments):
                if (iid,nd) not in labels:unmatched.append([iid,nd]);continue
                items.append((nd,w,t,k,nd not in seen));seen.add(nd)
            missing=[nd for (i,nd) in labels if i==iid and nd not in seen]
            if missing:unmatched+=[[iid,nd,'label-without-mention'] for nd in missing]
            if not items:continue
            pos=torch.tensor([[capstart-1+t,capstart+t] for _,_,t,_,_ in items],device=adapter.device)
            tg=ids[pos[:,1]]
            with torch.no_grad():
                st={}
                for name,b in (('real',br),('gray',bg)):
                    res.on=True;o=m(**b,use_cache=False,return_dict=True);res.on=False
                    st[name]=torch.stack([h[pos] for h in res.h],1).half().cpu()  # [mentions, layers+1, 2, D]
                    st[name+'_logp']=o.logits[0,pos[:,0]].float().log_softmax(-1).gather(-1,tg[:,None])[:,0].cpu()
                    del o;res.h=[None]*(NL+1)
            HR.append(st['real']);HG.append(st['gray'])
            for j,(nd,w,t,k,first) in enumerate(items):
                lb=labels[(iid,nd)]
                meta.append({'image_id':iid,'node':nd,'word':w,'kind':lb['kind'],'margin':lb['margin'],'first_mention':first,
                             'mention_index':k,'object_order':order[nd],'t':t,'T':T,
                             'logp_real':float(st['real_logp'][j]),'logp_gray':float(st['gray_logp'][j])})
            torch.cuda.empty_cache()
            if n%10==0:update(done=n+1,mentions=len(meta),sec_per_caption=(time.time()-t0)/(n+1),max_gpu_gb=torch.cuda.max_memory_allocated()/2**30)
        torch.save({'meta':meta,'h_real':torch.cat(HR),'h_gray':torch.cat(HG),'n_layers':NL,'positions':['query predicting the mention','mention token'],
                    'unmatched':unmatched},out/f'{tag}.pt')
        update(stage='complete',done=len(rows),mentions=len(meta),first_mentions=sum(x['first_mention'] for x in meta),unmatched=len(unmatched))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(CHAIR),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=100);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--all-images',action='store_true',help='draw from every caption with a labelled mention, not only those with a hallucination')
    main(p.parse_args())
