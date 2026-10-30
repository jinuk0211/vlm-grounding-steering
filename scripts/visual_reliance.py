"""How much does each caption token rely on the image? (teacher forcing on Base captions)

Three measures per caption token t, all on the unmodified Base model:
  attn  : attention mass from the position predicting t onto image tokens (per layer)
  vcd   : log p(t | real image) - log p(t | gray image of the same size)
  jac   : gradient x input attribution of log p(t) at the vision->LLM interface
          (input of decoder layer 0, i.e. projected visual tokens merged with
          text embeddings); image share = attribution on image positions / all
          earlier positions. Computed for object-mention tokens (CHAIR labels:
          hallucinated / correct) and every 16th caption token (position control).
For each mentioned object we also ask "Is there a {node} in the image?" and take
the same image share for the answer token, to contrast the caption position with
the question position for the same object.
"""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
from PIL import Image
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from verified_decoding_v2 import mention_token_positions
from score_vti_alpha_sweep import chair_scorer

CHAIR={'qwen':C.MODELS['qwen']/'original-vti-gpu1-final/chair','gemma':C.MODELS['gemma']/'chair'}

class Capture:
    """Make the decoder-layer-0 input a leaf that requires grad."""
    def __init__(self,layer):self.h=None;self.on=False;self.handle=layer.register_forward_pre_hook(self.hook,with_kwargs=True)
    def hook(self,mod,args,kwargs):
        if not self.on:return None
        if args:
            h=args[0].detach().requires_grad_(True);self.h=h;return (h,)+tuple(args[1:]),kwargs
        h=kwargs['hidden_states'].detach().requires_grad_(True);self.h=h;kwargs=dict(kwargs,hidden_states=h);return args,kwargs

def share(grad,h,img,upto):
    a=(grad*h).float().sum(-1).abs()[0,:upto]
    return float(a[img[:upto]].sum()/a.sum().clamp_min(1e-12)),float(grad.float()[0,:upto][img[:upto]].norm()/grad.float()[0,:upto].norm().clamp_min(1e-12))

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=json.loads((C.MODELS[a.model]/'config.json').read_text())['model']
    gray=out/'gray448.png'
    if not gray.exists():Image.new('RGB',(448,448),(128,128,128)).save(gray)
    sents={str(s['image_id']):s for s in json.loads((CHAIR[a.model]/'base.chair.official.json').read_text())['sentences']}
    rows=C.read(CHAIR[a.model]/'base.jsonl')[a.start:a.end]
    tag=f'{a.model}-{a.start}-{a.end}';status=out/f'{tag}-status.json'
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),model=a.model,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer
        for p in m.parameters():p.requires_grad_(False)
        img_id=C.image_token_id(m);scorer=chair_scorer();cap=Capture(adapter.layers[0])
        import caption_selfcheck as S;variants=S.variants(adapter)
        update(stage='running',gpu_name=torch.cuda.get_device_name(0))
        records=[]
        for n,r in enumerate(rows):
            pred=r['prediction']
            if not pred.strip():continue
            s=sents[str(r['metadata']['image_id'])];hall={x[1] for x in s['mscoco_hallucinated_words']}
            br=adapter.prepare(r['image'],r['prompt'],answer=pred);bb=adapter.prepare(str(gray),r['prompt'],answer=pred)
            assert torch.equal(br['input_ids'],bb['input_ids'])
            ids=br['input_ids'][0];lab=br.pop('labels')[0];bb.pop('labels')
            capstart=int((lab!=-100).nonzero()[0]);L=len(ids);img=(ids==img_id)
            capids=ids[capstart:].tolist()
            with torch.no_grad():
                o=m(**br,output_attentions=True,use_cache=False,return_dict=True)
                lp=torch.log_softmax(o.logits[0].float(),-1)
                attn=torch.stack([x[0,:,:,:].float()[:,:,img].sum(-1).mean(0) for x in o.attentions])  # [layers, L]
                del o
                ob=m(**bb,use_cache=False,return_dict=True);lpb=torch.log_softmax(ob.logits[0].float(),-1);del ob
            pos=list(range(capstart,L))
            lpr=torch.tensor([float(lp[p-1,ids[p]]) for p in pos]);lpbk=torch.tensor([float(lpb[p-1,ids[p]]) for p in pos])
            at=attn[:,[p-1 for p in pos]].half()
            # object mentions (caption-relative token index) and control positions
            ments=[(nd,w,t) for nd,w,t in mention_token_positions(tok,capids,scorer)]
            targets={t:('hallucinated' if nd in hall else 'correct',nd,w) for nd,w,t in ments}
            for t in range(0,len(capids)-1,16):targets.setdefault(t,('control',None,None))
            cap.on=True
            o=m(**br,use_cache=False,return_dict=True);h=cap.h;cap.on=False
            jac={}
            for t,(kind,nd,w) in sorted(targets.items()):
                p=capstart+t
                if p>=L:continue
                lpt=torch.log_softmax(o.logits[0,p-1].float(),-1)[ids[p]]
                g,=torch.autograd.grad(lpt,h,retain_graph=True)
                jac[t]={'kind':kind,'node':nd,'word':w,'img_share':share(g,h,img,p),'logp':float(lpt)}
            del o,h
            # same objects asked as a question
            qs={}
            for nd in sorted({nd for nd,_,_ in ments}):
                art='an' if nd[0] in 'aeiou' else 'a'
                bq=adapter.prepare(r['image'],f"Is there {art} {nd} in the image?\nPlease answer yes or no.")
                qimg=(bq['input_ids'][0]==img_id);cap.on=True
                oq=m(**bq,use_cache=False,return_dict=True);hq=cap.h;cap.on=False
                lg=oq.logits[0,-1].float();ly=torch.logsumexp(lg[variants['yes']],0);ln=torch.logsumexp(lg[variants['no']],0)
                ans=ly if ly>ln else ln
                g,=torch.autograd.grad(ans-torch.logsumexp(lg,0),hq)
                qs[nd]={'margin':float(ly-ln),'kind':'hallucinated' if nd in hall else 'correct','img_share':share(g,hq,qimg,hq.shape[1])}
                del oq,hq
            records.append({'id':r['id'],'image_id':r['metadata']['image_id'],'n_caption_tokens':len(capids),'n_image_tokens':int(img.sum()),
                            'logp_real':lpr.half(),'logp_gray':lpbk.half(),'attn_img':at,'jacobian':jac,'question':qs})
            torch.cuda.empty_cache()
            if n%10==0:update(done=n+1)
        torch.save(records,out/f'{tag}.pt');update(stage='complete',done=len(rows))
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(CHAIR),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=200)
    main(p.parse_args())
