"""E0-caption: does the model, when asked, deny objects it wrote in its own caption?

For every Base CHAIR caption (500 images) and every COCO object node that is
(a) mentioned and hallucinated, (b) mentioned and correct, or (c) in the
ground truth but not mentioned, ask the same model the POPE-style question
"Is there a {node} in the image?\nPlease answer yes or no." and record the
Yes-No margin (case/space variants pooled). No intervention, Base config.
"""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C

CHAIR={'qwen':C.MODELS['qwen']/'original-vti-gpu1-final/chair',
       'gemma':C.MODELS['gemma']/'chair'}

def variants(adapter):
    tok=adapter.processor.tokenizer;v={}
    for key,words in (('yes',('Yes','yes',' Yes',' yes')),('no',('No','no',' No',' no'))):
        v[key]=sorted({ids[0] for ids in (tok.encode(w,add_special_tokens=False) for w in words) if len(ids)==1})
    assert not set(v['yes'])&set(v['no']);return v

def queries(model,method='base'):
    off=json.loads((CHAIR[model]/f'{method}.chair.official.json').read_text())['sentences']
    rows={str(r['metadata']['image_id']):r for r in C.read(CHAIR[model]/f'{method}.jsonl')}
    out=[]
    for s in off:
        r=rows[str(s['image_id'])];hall={n for _,n in s['mscoco_hallucinated_words']}
        gen=s['mscoco_generated_words'];gt=set(s['mscoco_gt_words'])
        first={}
        for n in gen:first.setdefault(n,len(first))
        for n in sorted(set(gen)|gt):
            kind='hallucinated' if n in hall else ('correct' if n in gen else 'missed')
            out.append({'image_id':s['image_id'],'image':r['image'],'node':n,'kind':kind,
                        'mention_count':gen.count(n),'mention_order':first.get(n),'n_nodes':len(first),
                        'caption_tokens':r.get('generated_tokens')})
    return out

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    status=out/f'{a.tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    qs=queries(a.model,a.method)
    update(stage='loading',pid=os.getpid(),model=a.model,method=a.method,n_queries=len(qs),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);img_id=C.image_token_id(adapter.model);v=variants(adapter)
        update(stage='running',gpu_name=torch.cuda.get_device_name(0),variants=v)
        vis={}
        with (out/f'{a.tag}-rows.jsonl').open('x') as f:
            for n,q in enumerate(qs):
                prompt=f"Is there a {q['node']} in the image?\nPlease answer yes or no."
                logits,_,pooled,_=C.run(adapter,q['image'],prompt,img_id)
                ly=float(torch.logsumexp(logits[v['yes']],0));ln=float(torch.logsumexp(logits[v['no']],0))
                if q['image_id'] not in vis:vis[q['image_id']]=pooled
                f.write(json.dumps({**q,'prompt':prompt,'margin':ly-ln,'logit_yes':ly,'logit_no':ln})+'\n');f.flush()
                if n%100==0:update(done=n+1)
        ids=sorted(vis);torch.save({'groups':[f'coco:{i}' for i in ids],'image_ids':ids,'pooled':torch.stack([vis[i] for i in ids]),'pool':['mean','max']},
                                   out/f'{a.tag}-vision-states.pt')
        update(stage='complete',done=len(qs))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(CHAIR),required=True)
    p.add_argument('--method',default='base');p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    main(p.parse_args())
