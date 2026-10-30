"""Read-gap E0 collector: one prefill forward per item, no intervention.

pope mode  : POPE adversarial 3000 questions -> Yes/No logits at the answer
             position, last-position hidden state of every layer (answer-side
             read), and mean/max pooled image-token states per layer for each
             image (vision-side read). Image tokens precede the question, so
             their states do not depend on the question.
images mode: an image list (probe training) -> pooled image-token states only.

Model, dtype, image size and prompt template are those of the verified Base runs.
Writes only under --output.
"""
import argparse,json,os,time,traceback
from pathlib import Path

import torch
from vlm_grounding.backend import load_adapter

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
MODELS={'qwen':ROOT/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386',
        'gemma':ROOT/'outputs/gemma-grains-fp16-v2-20261029-1793246507'}
BASE_POPE={'qwen':MODELS['qwen']/'original-vti-gpu1-final/pope/base.jsonl',
           'gemma':MODELS['gemma']/'pope-complete/base.jsonl'}

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def image_token_id(model):
    cfg=model.config
    for k in ('image_token_id','image_token_index'):
        if getattr(cfg,k,None) is not None:return int(getattr(cfg,k))
    raise ValueError('no image token id')

@torch.no_grad()
def run(adapter,image,prompt,img_id):
    batch=adapter.prepare(image,prompt)
    out=adapter.model(**batch,output_hidden_states=True,use_cache=False,return_dict=True)
    hs=out.hidden_states
    pos=(batch['input_ids'][0]==img_id).nonzero().squeeze(1)
    assert len(pos)>0 and pos.max()<batch['input_ids'].shape[1]-5
    vis=torch.stack([torch.stack([h[0,pos].float().mean(0),h[0,pos].float().amax(0)]) for h in hs]).half().cpu()
    last=torch.stack([h[0,-1] for h in hs]).half().cpu()
    return out.logits[0,-1].float(),last,vis,len(pos)

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((MODELS[a.model]/'config.json').read_text())['model']
    assert config['dtype']=='bfloat16' and config['image_size']==448 and config['local_files_only']
    status=out/f'{a.tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),model=a.model,mode=a.mode,config=config,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);img_id=image_token_id(adapter.model)
        variants=None
        if a.answer_variants:
            tok=adapter.processor.tokenizer;variants={}
            for key,words in (('yes',('Yes','yes',' Yes',' yes')),('no',('No','no',' No',' no'))):
                ids={w:tok.encode(w,add_special_tokens=False) for w in words}
                variants[key]=sorted({v[0] for v in ids.values() if len(v)==1})
            assert not set(variants['yes'])&set(variants['no']) and adapter.yes_token_id in variants['yes']
        update(stage='running',gpu_name=torch.cuda.get_device_name(0),yes_id=adapter.yes_token_id,no_id=adapter.no_token_id,answer_variants=variants)
        if a.mode=='pope':
            data=read(MODELS[a.model]/'data/pope.jsonl');assert len(data)==3000
            base={r['id']:r['prediction'] for r in read(BASE_POPE[a.model])};assert len(base)==3000
            rows=[];lasts=[];vis={};agree=0
            for n,r in enumerate(data):
                logits,last,v,ntok=run(adapter,r['image'],r['prompt'],img_id)
                if variants:
                    # case/space variants of the answer word (Gemma answers mostly "no"/"yes")
                    ly=float(torch.logsumexp(logits[variants['yes']],0));ln=float(torch.logsumexp(logits[variants['no']],0))
                else:ly,ln=float(logits[adapter.yes_token_id]),float(logits[adapter.no_token_id])
                top=int(logits.argmax())
                if r['group_id'] not in vis:vis[r['group_id']]=v
                pred_yes=ly>ln;base_yes=base[r['id']].strip().lower().startswith('yes')
                agree+=pred_yes==base_yes
                rows.append({'id':r['id'],'group_id':r['group_id'],'label':r['label'],
                    'object':r['metadata'].get('object') if isinstance(r.get('metadata'),dict) else None,
                    'prompt':r['prompt'],'logit_yes':ly,'logit_no':ln,'margin':ly-ln,'top1':top,
                    'top1_is_yes_or_no':top in ((variants['yes']+variants['no']) if variants else (adapter.yes_token_id,adapter.no_token_id)),
                    'base_prediction':base[r['id']],'logit_agrees_with_base':pred_yes==base_yes,'image_tokens':ntok})
                lasts.append(last)
                if n%100==0:update(done=n+1,agree=agree)
            with (out/f'{a.tag}-rows.jsonl').open('x') as f:
                for r in rows:f.write(json.dumps(r)+'\n')
            torch.save({'ids':[r['id'] for r in rows],'last':torch.stack(lasts)},out/f'{a.tag}-answer-states.pt')
            groups=sorted(vis);torch.save({'groups':groups,'pooled':torch.stack([vis[g] for g in groups]),'pool':['mean','max']},out/f'{a.tag}-vision-states.pt')
            update(stage='complete',done=len(rows),agree=agree,agree_rate=agree/len(rows))
        else:
            items=read(a.image_list);pooled=[];ids=[]
            for n,r in enumerate(items):
                _,_,v,_=run(adapter,r['image'],'Please describe this image in detail.',img_id)
                pooled.append(v);ids.append(r['id'])
                if n%200==0:update(done=n+1)
            torch.save({'ids':ids,'pooled':torch.stack(pooled),'pool':['mean','max']},out/f'{a.tag}-vision-states.pt')
            update(stage='complete',done=len(ids))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--mode',choices=['pope','images'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--image-list',type=Path)
    p.add_argument('--answer-variants',action='store_true',help='margin over case/space variants of Yes/No')
    main(p.parse_args())
