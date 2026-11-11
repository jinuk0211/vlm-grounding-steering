"""Checks for cdm_decode.py before the CHAIR runs (one model per call, 2 images of the set).
M3ID: row 1 of the pair (image span masked out of attention) must equal a text-only forward of the same prompt.
      Decode 8 tokens through the pair (row 0 greedy), record row 1's log-probs at every step, then teacher-force the
      text-only prompt + those tokens in one forward; compare top-1 and max |d log p| over the reference top-50.
VCD:  noised pixel stats (author schedule at the step) and that Qwen's two temporal copies got the same noise.
usage: cdm_check.py MODEL ATTN SET.jsonl
"""
import json,sys
import torch
from transformers import LogitsProcessor,LogitsProcessorList
import cdm_decode as D
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C

class Rec(LogitsProcessor):
    def __init__(self):self.lu=[]
    def __call__(self,input_ids,scores):
        self.lu.append(scores[1].float().log_softmax(-1).cpu());scores[1]=scores[0];return scores

def main(model,attn,data):
    config=dict(json.loads((C.MODELS[model]/'config.json').read_text())['model'],attn_implementation=attn)
    adapter=load_adapter(config);m=adapter.model;refs=C.read(data)[:2];ok=True
    class A:method='m3id';vcd_noise=500
    for ref in refs:
        batch,info=D.pair(adapter,ref,A);batch={k:v.to(adapter.device) for k,v in batch.items()};pl=batch['input_ids'].shape[1]
        rec=Rec()
        with torch.no_grad():
            o=m.generate(**batch,do_sample=False,num_beams=1,use_cache=True,max_new_tokens=8,logits_processor=LogitsProcessorList([rec]))
        new=o[0,pl:].tolist();k=len(rec.lu)
        tb=adapter.prepare(None,ref['prompt']);L=tb['input_ids'].shape[1];ext=torch.tensor([new[:k-1]],device=adapter.device)
        tf={}
        for key,v in tb.items():
            if key=='labels':continue
            if torch.is_tensor(v) and v.dim()==2 and v.shape==(1,L) and not v.is_floating_point():
                fill=ext if key=='input_ids' else (torch.ones_like(ext) if key=='attention_mask' else torch.zeros_like(ext))
                tf[key]=torch.cat([v,fill.to(v.dtype)],1)
            else:tf[key]=v
        with torch.no_grad():ref_lp=m(**tf,use_cache=False).logits[0,L-1:L-1+k].float().log_softmax(-1).cpu()
        top1=[int(rec.lu[j].argmax())==int(ref_lp[j].argmax()) for j in range(k)]
        d=[float((rec.lu[j][ref_lp[j].topk(50).indices]-ref_lp[j][ref_lp[j].topk(50).indices]).abs().max()) for j in range(k)]
        print(f"[m3id] {ref['id']} span {info['masked_span']} of {info['prompt_len']} | text-only prompt {L} tok | "
              f"top1 agree {sum(top1)}/{k} | max|dlogp| top50 per step {[round(x,3) for x in d]}")
        print('        masked span starts', repr(adapter.processor.tokenizer.decode(batch['input_ids'][0,info['masked_span'][0]:info['masked_span'][0]+2])),
              'ends', repr(adapter.processor.tokenizer.decode(batch['input_ids'][0,info['masked_span'][1]-2:info['masked_span'][1]])))
        ok&=sum(top1)>=k-1 and max(d)<0.5
        pv=adapter.prepare(ref['image'],ref['prompt'])['pixel_values'];nv=D.noised(adapter,pv,ref,500)
        x=pv.float();y=nv.float();c=torch.corrcoef(torch.stack([x.flatten(),y.flatten()]))[0,1]
        extra=''
        if adapter.model_type in ('qwen2_5_vl','qwen3_vl'):
            ip=adapter.processor.image_processor;T=int(ip.temporal_patch_size);P=int(ip.patch_size)
            v=y.view(y.shape[0],3,T,P,P);extra=f' | temporal copies equal after noise: {bool(torch.equal(v[:,:,0],v[:,:,1]))}'
        print(f"[vcd]  {ref['id']} pixel {tuple(pv.shape)} {pv.dtype} mean/std {x.mean():.3f}/{x.std():.3f} -> noised {y.mean():.3f}/{y.std():.3f}, corr {c:.3f}{extra}")
    print('CHECK_OK' if ok else 'CHECK_FAIL')

if __name__=='__main__':main(*sys.argv[1:4])
