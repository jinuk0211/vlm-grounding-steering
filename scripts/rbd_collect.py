"""RBD pilot step 1: log p over V_obj at every caption step (teacher forcing on the stored Base captions).

Per caption: T x |V_obj| log-probabilities (full-vocabulary log-softmax, then restricted to V_obj), the id of the
token actually written, and the object mentions with CHAIR's own hallucinated/correct label (first mention per node).
"""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from verified_decoding_v2 import mention_token_positions
from score_vti_alpha_sweep import chair_scorer
from rbd_common import build_vobj
CHAIR={'qwen':C.MODELS['qwen']/'original-vti-gpu1-final/chair','gemma':C.MODELS['gemma']/'chair'}

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    tag=f'{a.model}-{a.start}-{a.end}';status=out/f'{tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid())
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer
        vp=out/f'{a.model}-vobj2.json'
        if vp.exists():vobj=json.loads(vp.read_text())
        else:vobj=build_vobj(tok);vp.write_text(json.dumps(vobj))
        vidx=torch.tensor(vobj,device=adapter.device);vset={v:i for i,v in enumerate(vobj)}
        update(stage='running',vobj_size=len(vobj),example=[tok.convert_ids_to_tokens(i) for i in vobj[:: max(1,len(vobj)//40)]])
        scorer=chair_scorer();rows=C.read(CHAIR[a.model]/'base.jsonl')[a.start:a.end];recs=[]
        for n,r in enumerate(rows):
            pred=r['prediction']
            if not pred.strip():continue
            gt=scorer.imid_to_objects[int(r['metadata']['image_id'])]
            b=adapter.prepare(r['image'],r['prompt'],answer=pred);lab=b.pop('labels')[0];ids=b['input_ids'][0]
            cs=int((lab!=-100).nonzero()[0]);L=len(ids)
            with torch.no_grad():
                lg=m(**b,use_cache=False,return_dict=True).logits[0,cs-1:L-1].float()
                lp=lg.log_softmax(-1)
            tgt=ids[cs:];lp_act=lp.gather(-1,tgt[:,None])[:,0]
            capids=tgt.tolist();seen=set();ments=[]
            for nd,w,t in mention_token_positions(tok,capids,scorer):
                if nd in seen:continue
                seen.add(nd);ments.append({'node':nd,'word':w,'t':t,'token':capids[t],'in_vobj':capids[t] in vset,'kind':'correct' if nd in gt else 'hallucinated'})
            recs.append({'id':r['id'],'image_id':r['metadata']['image_id'],'T':len(capids),'caption_ids':capids,'logp_actual':lp_act.half().cpu(),
                         'logp_vobj':lp[:,vidx].half().cpu(),'mentions':ments})
            if n%25==0:update(done=n+1)
        torch.save(recs,out/f'{tag}.pt');update(stage='complete',done=len(rows),records=len(recs))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(CHAIR),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=500);main(p.parse_args())
