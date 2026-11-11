"""RBD, AMBER-oriented variant: vdd_decode.py plus three options that target why AMBER moved little.
With every new option at its default this decodes exactly like vdd_decode.py (same code path, same rows).

Diagnosis behind the options (teacher-forced Base captions, AMBER dev 101-160 and test 1-60, pb_diag_evidence*.py):
  - raw evidence log p_real - log p_gray is negative for almost every object at its mention (truth -1.5, hallucinated
    -2.0); median-normalised evidence separates truth from hallucinated objects much better (AUROC 0.72-0.76 vs
    0.57-0.61) but then hallucinated objects get a positive belief (+0.2 to +0.5), so lam * belief raises them, and the
    flat penalty pen pushes down truth objects as often as it pushes down hallucinated ones near the operating point.
  - scene/stuff nouns (sky, grass, wall, ...) that are present get lower evidence than things (+1.25 vs +1.9 median
    belief), and AMBER counts them in Cover, so a flat penalty costs Cover on AMBER only (CHAIR/THRONE ignore them).
  - a share of AMBER hallucinations sit in hedged clauses ("possibly a phone"); once "possibly a" is written, lowering
    one object just swaps in another.

New options:
  --gpen P --tau T [--gtemp S]   gated penalty: logits0(o) -= P * w(t) * g(o),  g = sigmoid((T - b(o)) / S)  (S=0: step)
                                 b = clip(A/sum r) is the belief (use with --norm median). Objects whose accumulated
                                 visual evidence is above T are left alone; there is no boost unless --lam is also set.
  --stuff exempt                 scene/stuff nouns (STUFF below, with plurals) get neither --pen nor --gpen.
  --hedge H                      logits0(h) -= H * w(t) for hedging words (HEDGE below), so late, weakly grounded
                                 steps do not open speculative clauses. A stylistic term: report it as such.
"""
import argparse,json,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LogitsProcessor,LogitsProcessorList
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
from rbd_common import ROOT,r_curve

GRAY=Path(os.environ.get('VDD_GRAY','/workspace/rbd_pilot/gray448.png'))  # project-b: set VDD_GRAY

# same list as pb_diag_amber_mentions.py
STUFF=('sky ground sea grass forest cloud water road sand snow mountain beach desert field wall floor lake river ocean hill '
       'sun street sidewalk dirt rock wave tree stone soil land shore cliff valley meadow').split()
HEDGE='possibly perhaps maybe likely probably potentially presumably might'.split()

_linear=F.linear
def _rowsplit_linear(x,w,b=None):
    if x.dtype==torch.bfloat16 and x.dim()==3 and x.shape[0]==2 and x.shape[1]==1:
        return torch.cat([_linear(x[i:i+1],w,b) for i in range(2)],0)
    return _linear(x,w,b)
F.linear=_rowsplit_linear

def word_ids(tok,words,capitalize=False):
    """single-token ids of ' word' (and ' Word' / 'Word' when capitalize), for words that are one token"""
    ids=set()
    for w in words:
        forms=[' '+w]+([' '+w.capitalize(),w.capitalize()] if capitalize else [])
        for f in forms:
            t=tok.encode(f,add_special_tokens=False)
            if len(t)==1:ids.add(t[0])
    return sorted(ids)

class State:
    def __init__(self):self.t=0;self.on=False

class Evidence(LogitsProcessor):
    """belief + penalties on row 0, then row 1 gets row 0's scores so both rows take the same token"""
    def __init__(self,st,vidx,rf,a,pl,penmask,hidx):
        self.st=st;self.vidx=vidx;self.rf=rf;self.a=a;self.pl=pl;self.penmask=penmask;self.hidx=hidx
        self.A=torch.zeros(len(vidx),device=vidx.device);self.R=0.0;self.top=[]
    def w(self,t):return 1-self.rf(t) if self.a.bsched=='decay' else 1.0
    def __call__(self,input_ids,scores):
        t=input_ids.shape[1]-self.pl;self.st.t=t+1   # steering for the next forward uses the next position
        a=self.a
        if a.lam or a.gpen or a.log_belief:
            lp=scores.float().log_softmax(-1)[:,self.vidx]
            d=lp[0]-lp[1]
            if a.norm=='median':d=d-d.median()   # remove the shift every object gets at this step (peaked vs flat distribution)
            r=self.rf(t);self.A+=r*d;self.R+=r
        if a.lam or a.pen or a.gpen:
            done=torch.isin(self.vidx,input_ids[0,self.pl:]);adj=torch.zeros_like(self.A)
            b=(self.A/max(self.R,1e-6)).clamp(-a.clip,a.clip)
            if a.lam:adj+=a.lam*self.w(t)*b.clamp(0.0 if a.bpos else -a.clip,a.clip)
            if a.pen:adj-=a.pen*self.w(t)*self.penmask
            if a.gpen:
                g=torch.sigmoid((a.tau-b)/a.gtemp) if a.gtemp>0 else (b<a.tau).float()
                adj-=a.gpen*self.w(t)*g*self.penmask
            adj=torch.where(done,torch.zeros_like(adj),adj);scores[0,self.vidx]+=adj.to(scores.dtype)
        if a.hedge and len(self.hidx):
            scores[0,self.hidx]-=torch.tensor(a.hedge*self.w(t),dtype=scores.dtype,device=scores.device)
        if a.log_belief and t in (8,32,128):
            k=torch.topk(self.A/max(self.R,1e-6),12);self.top.append([t,[(int(self.vidx[i]),round(float(v),2)) for v,i in zip(k.values,k.indices)]])
        scores[1]=scores[0]
        return scores

class Steer:
    def __init__(self,layer,st,a,rf):
        self.st=st;self.a=a;self.rf=rf;layer.register_forward_hook(self.hook)
    def s(self,t):return 1-self.rf(t) if self.a.ssched=='decay' else 1.0
    def hook(self,mod,args,out):
        h=out[0] if isinstance(out,tuple) else out
        if not self.st.on or h.shape[1]!=1 or not self.a.alpha:return None
        h=h.clone();h[0]=h[0]+(self.a.alpha*self.s(self.st.t)*(h[0]-h[1]).float()).to(h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out,tuple) else h

def pair(adapter,ref):
    b0=adapter.prepare(ref['image'],ref['prompt']);b1=adapter.prepare(str(GRAY),ref['prompt'])
    assert b0.keys()==b1.keys() and torch.equal(b0['input_ids'],b1['input_ids'])
    return {k:torch.cat([b0[k],b1[k]],0) for k in b0 if k!='labels'}

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count()==1
    assert not (a.gpen and a.norm!='median'),'--gpen thresholds were set on median-normalised beliefs; use --norm median'
    torch.manual_seed(42)
    if not GRAY.exists():Image.new('RGB',(448,448),(128,128,128)).save(GRAY)
    config=dict(json.loads((C.MODELS[a.model]/'config.json').read_text())['model'],attn_implementation=a.attn)
    refs=C.read(a.data)[a.start:a.end]
    name=f'{a.tag}-{a.start}-{a.end}';rows_file=out/f'{name}.jsonl';status=out/f'{name}-status.json'
    done=C.read(rows_file) if rows_file.exists() else []
    setting={k:getattr(a,k) for k in ('lam','clip','pen','bsched','alpha','layer','ssched','r','attn','bpos','norm',
                                      'gpen','tau','gtemp','stuff','hedge')}
    def update(**k):
        s=json.loads(status.read_text()) if status.exists() else {}
        s.update(k,updated_at=time.time());status.write_text(json.dumps(s,indent=1))
    update(stage='loading',pid=os.getpid(),args={k:str(v) for k,v in vars(a).items()},resumed_from=len(done))
    try:
        adapter=load_adapter(config);m=adapter.model;tok=adapter.processor.tokenizer
        vobj=json.loads((ROOT/f'outputs/rbd-20261030/collect/{a.model}-vobj2.json').read_text())
        vidx=torch.tensor(vobj,device=adapter.device);rf=r_curve(a.model,a.r);st=State()
        penmask=torch.ones(len(vobj),device=adapter.device)
        stuff_ids=[]
        if a.stuff=='exempt':
            col={v:i for i,v in enumerate(vobj)}
            stuff_ids=[i for i in word_ids(tok,STUFF+[w+'s' for w in STUFF]) if i in col]
            for i in stuff_ids:penmask[col[i]]=0.0
        hidx=torch.tensor(word_ids(tok,HEDGE,capitalize=True) if a.hedge else [],dtype=torch.long,device=adapter.device)
        Steer(adapter.layers[a.layer-1],st,a,rf)   # output of decoder layer a.layer (1-based)
        update(stage='running',vobj=len(vobj),stuff_exempt=[tok.decode([i]) for i in stuff_ids],
               hedge_tokens=[tok.decode([i]) for i in hidx.tolist()])
        with rows_file.open('a') as f:
            for i,ref in enumerate(refs[len(done):],start=len(done)):
                t0=time.perf_counter();batch={k:v.to(adapter.device) for k,v in pair(adapter,ref).items()};pl=batch['input_ids'].shape[1]
                proc=Evidence(st,vidx,rf,a,pl,penmask,hidx);st.t=0;st.on=True
                try:
                    with torch.no_grad():
                        o=m.generate(**batch,do_sample=False,num_beams=1,use_cache=True,max_new_tokens=int(ref['max_new_tokens']),
                                     logits_processor=LogitsProcessorList([proc]))
                finally:st.on=False
                assert torch.equal(o[0],o[1])
                ids=o[0,pl:].tolist()
                if tok.pad_token_id in ids:ids=ids[:ids.index(tok.pad_token_id)]
                text=tok.decode(ids,skip_special_tokens=True)
                row={**ref,'method':a.tag,'prediction':text,'generated_tokens':len(ids),'elapsed_seconds':time.perf_counter()-t0,
                     'decoder':'vdd3_decode.py','setting':setting}
                if a.log_belief:row['belief_top']=[[t,[(tok.decode([j]),v) for j,v in tv]] for t,tv in proc.top]
                f.write(json.dumps(row)+'\n');f.flush();update(done=i+1)
        update(stage='complete')
    except BaseException:
        update(stage='failed',error=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(C.MODELS),required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tag',required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=60)
    p.add_argument('--lam',type=float,default=0.0);p.add_argument('--clip',type=float,default=5.0)
    p.add_argument('--pen',type=float,default=0.0);p.add_argument('--bsched',choices=['decay','const'],default='decay')
    p.add_argument('--alpha',type=float,default=0.0);p.add_argument('--layer',type=int,default=16)
    p.add_argument('--ssched',choices=['decay','const'],default='decay')
    p.add_argument('--r',choices=['measured','const','step0'],default='measured')
    p.add_argument('--attn',default='sdpa');p.add_argument('--bpos',action='store_true');p.add_argument('--norm',choices=['raw','median'],default='raw')
    p.add_argument('--gpen',type=float,default=0.0);p.add_argument('--tau',type=float,default=1.0);p.add_argument('--gtemp',type=float,default=0.5)
    p.add_argument('--stuff',choices=['pen','exempt'],default='pen');p.add_argument('--hedge',type=float,default=0.0)
    p.add_argument('--log-belief',action='store_true')
    main(p.parse_args())
