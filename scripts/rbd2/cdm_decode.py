"""VCD and M3ID baselines on the caption sets, through the lockstep pair path of vdd_decode.py: each caption is
decoded as a batch of two, row 0 = real image, row 1 = the method's contrast input, row 1 always fed row 0's token.
Greedy, the set's prompt and max_new_tokens, the model config of the vdd Base rows (448 px, bf16, attn per model).

VCD  Leng et al., CVPR 2024. Official code DAMO-NLP-SG/VCD @ d6568ff (vcd_utils/vcd_sample.py, vcd_add_noise.py).
     row 1 = same prompt, image through the author's add_diffusion_noise(noise_step) (author schedule, applied to the
     normalised pixel tensor the vision tower consumes, as in the author eval scripts: Qwen's CLIP-normalised patches;
     Gemma 4's 2*(x-0.5), mapped back to the processor's [0, 1]). Qwen stores a still image as two identical
     temporal copies; both copies get the same noise, so row 1 sees one noised image.
     cutoff = log(beta) + max logit_real ; cd = (1+alpha) logit_real - alpha logit_noise ; tokens with
     logit_real < cutoff -> -inf  (author 'version 2' constraint). Author defaults: alpha 1, beta 0.1, noise_step 500.
     The author scripts sample (T 1, top-p 1) from cd; here next = argmax cd, because every row of the CHAIR table is greedy.
M3ID Favero et al., CVPR 2024. No official code; paper Algorithm 1 / Eq. 3-4.
     row 1 = image removed: the image span (vision start .. vision end) is masked out of attention; the kept tokens are
     exactly the text-only prompt (asserted per image), cdm_check.py checks that row 1 then matches a text-only forward.
     lc, lu = log-softmax with / without image; gamma_t = exp(-lam t), t = 1 at the first generated token;
     l = lc + 1[max lc < log thr] * (1 - gamma_t) / gamma_t * (lc - lu) ; next = argmax l.
     Paper values lam 0.02, thr 0.3 (best LLaVA-7B setting, paper Table 3). No plausibility cutoff (paper has none).
--vcd-alpha 0 is Base decoding of row 0 through the identical code path (repro check against the vdd Base rows).
Rows: {output}/{tag}-{start}-{end}.jsonl (resumable), row format of the rbd2 runs, plus per-caption diagnostics.
"""
import argparse,hashlib,json,math,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor,LogitsProcessorList
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C

_linear=F.linear
def _rowsplit_linear(x,w,b=None):   # as vdd_decode.py: decode-step linears row by row (row 0 = batch-1 kernels)
    if x.dtype==torch.bfloat16 and x.dim()==3 and x.shape[0]==2 and x.shape[1]==1:
        return torch.cat([_linear(x[i:i+1],w,b) for i in range(2)],0)
    return _linear(x,w,b)
F.linear=_rowsplit_linear

# ---- author code: VCD vcd_utils/vcd_add_noise.py (schedule verbatim; randn from a seeded generator) ----
def add_diffusion_noise(image_tensor,noise_step,generator=None,noise_shape=None):
    num_steps=1000
    betas=torch.linspace(-6,6,num_steps)
    betas=torch.sigmoid(betas)*(0.5e-2-1e-5)+1e-5
    alphas=1-betas
    alphas_prod=torch.cumprod(alphas,dim=0)
    alphas_bar_sqrt=torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt=torch.sqrt(1-alphas_prod)
    def q_x(x_0,t):
        noise=torch.randn(noise_shape or x_0.shape,generator=generator,dtype=torch.float32)
        return alphas_bar_sqrt[t]*x_0+one_minus_alphas_bar_sqrt[t]*noise
    return q_x(image_tensor.clone(),noise_step)

def seed_of(ref):return int(hashlib.sha256(('vcd:'+ref['id']).encode()).hexdigest()[:8],16)

def noised(adapter,pv,ref,step):
    """pixel_values -> noised pixel_values (fp32 math on the bf16 tensor the model sees, back to its dtype)"""
    g=torch.Generator().manual_seed(seed_of(ref));x=pv.detach().float().cpu()
    if adapter.model_type in ('qwen2_5_vl','qwen3_vl'):
        ip=adapter.processor.image_processor;T=int(ip.temporal_patch_size);P=int(ip.patch_size)
        assert x.dim()==2 and x.shape[1]==3*T*P*P,(x.shape,T,P)
        v=x.view(x.shape[0],3,T,P,P)
        assert torch.equal(v[:,:,0],v[:,:,1]),'temporal copies of a still image should be identical'
        y=add_diffusion_noise(v,step,g,noise_shape=(x.shape[0],3,1,P,P)).reshape(x.shape)   # one noise for both copies
    elif adapter.model_type=='gemma4':
        # Gemma 4's processor returns [0, 1]; the model feeds the vision tower 2 * (x - 0.5). The author noises the tensor the
        # vision tower consumes (CLIP-normalised for LLaVA), so noise in that space and map back.
        y=(add_diffusion_noise(2*(x-0.5),step,g)+1)/2
    else:
        y=add_diffusion_noise(x,step,g)
    return y.to(pv.device,pv.dtype)

def image_span(full,text):
    """full = prompt with image, text = same prompt without image -> [a, b) such that full minus [a, b) == text"""
    n,m=len(full),len(text);p=0
    while p<m and full[p]==text[p]:p+=1
    s=0
    while s<m-p and full[n-1-s]==text[m-1-s]:s+=1
    assert p+s==m,('image prompt is not text prompt + one span',p,s,m,n)
    return p,n-s

def image_token_ids(adapter):
    cfg=adapter.model.config;ids=set()
    for k in ('image_token_id','image_token_index'):
        if getattr(cfg,k,None) is not None:ids.add(int(getattr(cfg,k)))
    assert ids,'no image token id in config';return ids

def pair(adapter,ref,a):
    b0=adapter.prepare(ref['image'],ref['prompt']);b0={k:v for k,v in b0.items() if k!='labels'};info={}
    if a.method=='vcd':
        b1=dict(b0);b1['pixel_values']=noised(adapter,b0['pixel_values'],ref,a.vcd_noise)
        info['noise_seed']=seed_of(ref)
    else:
        b1=dict(b0);full=b0['input_ids'][0].tolist()
        text=adapter.prepare(None,ref['prompt'])['input_ids'][0].tolist()
        s,e=image_span(full,text);imt=image_token_ids(adapter)
        assert all(full[i] not in imt for i in range(len(full)) if not s<=i<e),'image tokens outside the masked span'
        m=b0['attention_mask'].clone();m[0,s:e]=0;b1['attention_mask']=m
        info['masked_span']=[s,e];info['prompt_len']=len(full)
    assert b0.keys()==b1.keys() and torch.equal(b0['input_ids'],b1['input_ids'])
    return {k:torch.cat([b0[k],b1[k]],0) for k in b0},info

class VCD(LogitsProcessor):
    def __init__(self,alpha,beta):self.alpha=alpha;self.beta=beta;self.steps=0;self.changed=0
    def __call__(self,input_ids,scores):
        lr=scores[0].float();ln=scores[1].float()
        cutoff=math.log(self.beta)+lr.max()
        diffs=(1+self.alpha)*lr-self.alpha*ln
        cd=diffs.masked_fill(lr<cutoff,-float('inf'))
        self.steps+=1;self.changed+=int(cd.argmax()!=lr.argmax())
        scores[0]=cd.to(scores.dtype);scores[1]=scores[0]
        return scores

class M3ID(LogitsProcessor):
    def __init__(self,lam,thr,pl):self.lam=lam;self.thr=thr;self.pl=pl;self.steps=0;self.fired=0;self.changed=0;self.maxfac=0.0
    def __call__(self,input_ids,scores):
        t=input_ids.shape[1]-self.pl+1
        lc=scores[0].float().log_softmax(-1);lu=scores[1].float().log_softmax(-1);l=lc
        if lc.max()<math.log(self.thr):
            g=math.exp(-self.lam*t);fac=(1-g)/g;l=lc+fac*(lc-lu)
            self.fired+=1;self.maxfac=max(self.maxfac,fac)
        self.steps+=1;self.changed+=int(l.argmax()!=lc.argmax())
        scores[0]=l.to(scores.dtype);scores[1]=scores[0]
        return scores

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    torch.manual_seed(42)
    config=dict(json.loads((C.MODELS[a.model]/'config.json').read_text())['model'],attn_implementation=a.attn)
    refs=C.read(a.data)[a.start:a.end]
    name=f'{a.tag}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    done=C.read(rows_file) if rows_file.exists() else []
    assert not done or rows_file.read_bytes().endswith(b'\n')
    for r,ref in zip(done,refs):assert r['id']==ref['id'] and r['method']==a.tag
    if a.method=='vcd':setting={'method':'vcd','alpha':a.vcd_alpha,'beta':a.vcd_beta,'noise_step':a.vcd_noise}
    else:setting={'method':'m3id','lam':a.m3id_lam,'thr':a.m3id_thr}
    setting.update(attn=a.attn,greedy=True,model=a.model)
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),args={k:str(v) for k,v in vars(a).items()},total=len(refs),done=len(done),resumed_from=len(done))
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer;gpu=torch.cuda.get_device_name(0)
        update(stage='running',gpu=gpu)
        with rows_file.open('a') as f:
            for i,ref in enumerate(refs[len(done):],start=len(done)):
                assert hashlib.sha256(Path(ref['image']).read_bytes()).hexdigest()==ref['image_sha256'],ref['image']
                t0=time.perf_counter();batch,info=pair(adapter,ref,a)
                batch={k:v.to(adapter.device) for k,v in batch.items()};pl=batch['input_ids'].shape[1]
                proc=VCD(a.vcd_alpha,a.vcd_beta) if a.method=='vcd' else M3ID(a.m3id_lam,a.m3id_thr,pl)
                with torch.no_grad():
                    o=m.generate(**batch,do_sample=False,num_beams=1,use_cache=True,max_new_tokens=int(ref['max_new_tokens']),
                                 logits_processor=LogitsProcessorList([proc]))
                assert torch.equal(o[0],o[1])
                ids=o[0,pl:].tolist()
                if tok.pad_token_id in ids:ids=ids[:ids.index(tok.pad_token_id)]
                text=tok.decode(ids,skip_special_tokens=True)
                diag={'steps':proc.steps,'changed':proc.changed,**info}
                if a.method=='m3id':diag.update(fired=proc.fired,max_factor=round(proc.maxfac,2))
                row={**ref,'method':a.tag,'prediction':text,'generated_tokens':len(ids),'elapsed_seconds':time.perf_counter()-t0,
                     'decoder':'cdm_decode.py','setting':setting,'diag':diag,'gpu':gpu}
                f.write(json.dumps(row)+'\n');f.flush();update(done=i+1)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(C.MODELS),required=True)
    p.add_argument('--method',choices=['vcd','m3id'],required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=60);p.add_argument('--attn',default='sdpa')
    p.add_argument('--vcd-alpha',type=float,default=1.0);p.add_argument('--vcd-beta',type=float,default=0.1)
    p.add_argument('--vcd-noise',type=int,default=500)
    p.add_argument('--m3id-lam',type=float,default=0.02);p.add_argument('--m3id-thr',type=float,default=0.3)
    main(p.parse_args())
