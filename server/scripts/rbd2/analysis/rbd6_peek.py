"""Interim look at rbd6 candidates (CPU, read-only on the runs): every candidate with its finished rows vs vdd Base and the RBD row on the same images. Selection
is still done by rbd6_ctl.py on complete stages; this does not feed it.   usage: rbd6_peek.py"""
import glob,hashlib,json,os,re
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';X=O/'runs';TMP=X/'rbd6_peek_tmp';TMP.mkdir(exist_ok=True)
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_peek');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state']);sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];ids=[r['id'] for r in refs];IM={r['id']:r['metadata']['image_id'] for r in refs}
def score(by,sel):
    f=TMP/'c.json';f.write_text(json.dumps([{'image_id':IM[i],'caption':by[i]['prediction']} for i in sel]))
    d=sc.compute_chair(str(f),'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];Rc=m['Recall'];s={x['image_id']:x for x in d['sentences']}
    hw=[{n for _,n in s[IM[i]]['mscoco_hallucinated_words']} for i in sel];t=[by[i]['generated_tokens'] for i in sel]
    p=dict(hal=np.array([bool(x) for x in hw],float),halx=np.array([bool(x-{'orange'}) for x in hw],float),gt=np.array([len(set(s[IM[i]]['mscoco_gt_words'])) for i in sel],float),hit=np.array([len(set(s[IM[i]]['mscoco_gt_words'])&set(s[IM[i]]['mscoco_generated_words'])) for i in sel],float))
    return dict(s=100*m['CHAIRs'],x=100*p['halx'].mean(),i=100*m['CHAIRi'],r=100*Rc,f=100*2*P*Rc/(P+Rc),t=float(np.mean(t)),a=int(sum(v>=512 for v in t))),p
def ci(a,b):
    n=len(a['hal']);g=np.random.default_rng(0);ds=[];dr=[]
    for _ in range(2000):
        k=g.integers(0,n,n);ds.append(100*(a['hal'][k].mean()-b['hal'][k].mean()));dr.append(100*(a['hit'][k].sum()/a['gt'][k].sum()-b['hit'][k].sum()/b['gt'][k].sum()))
    q=lambda v:'[%+.1f,%+.1f]'%(np.quantile(v,.025),np.quantile(v,.975));return f"s{q(ds)} r{q(dr)}"
rd=lambda p:{r['id']:r for r in map(json.loads,open(p))}
for od,srcd,rb,name in (('rbd6_qwen3','qchair_test500_all','s20a1bl1p05','Qwen3'),('rbd6_gemma','gchair_test500_all','s24a1bl1p05','Gemma'),('rbd6_qwen25','chair_test500_all','s16a2bl1p05','Qwen2.5')):
    S=X/od;tags={}
    for f in glob.glob(str(S/'*-*-*.jsonl')):
        t=re.sub(r'-\d+-\d+\.jsonl$','',os.path.basename(f));by=tags.setdefault(t,{})
        for l in open(f):
            if l.endswith('\n') and l.strip():r=json.loads(l);by[r['id']]=r
    if not tags:continue
    B=rd(X/srcd/'base-0-500.jsonl');D=rd(X/srcd/f'{rb}-0-500.jsonl');print(f'=== {name}: current RBD row {rb}',flush=True)
    for t,by in sorted(tags.items(),key=lambda x:x[0]):
        sel=[i for i in ids if i in by];vb,pb=score(B,sel);vd,pd=score(D,sel);v,p=score(by,sel)
        print(f"{t:14s} n {len(sel):3d} | CHAIR_s {v['s']:5.1f} (Base {vb['s']:5.1f} RBD {vd['s']:5.1f}) no-orange {v['x']:5.1f} ({vb['x']:5.1f} {vd['x']:5.1f}) | CHAIR_i {v['i']:5.2f} ({vb['i']:5.2f} {vd['i']:5.2f}) | recall {v['r']:5.1f} ({vb['r']:5.1f} {vd['r']:5.1f}) | F1 {v['f']:5.1f} ({vb['f']:5.1f} {vd['f']:5.1f}) | tok {v['t']:4.0f} @512 {v['a']:2d} ({vb['a']:2d} {vd['a']:2d}) | vsBase {ci(p,pb)} vsRBD {ci(p,pd)}",flush=True)
print('PEEK_DONE',flush=True)
