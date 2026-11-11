"""Interim CHAIR scoring of the running Qwen3-VL baselines (CPU only; runs/qchair_test500_tbl is only read).
Complete lines of each shard are copied to SNAP (a torn last line is dropped). Every method is compared with the vdd Base and
RBD on the images that method has finished, overall and per shard, with the unmodified CHAIR scorer of score_sets.py and the
same paired bootstrap (2000, seed 0). Also: first-divergence statistics between caption pairs (path comparison).
usage: qchair_interim.py SNAP"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';T=O/'runs/qchair_test500_tbl'
SNAP=Path(sys.argv[1]);TMP=SNAP/'tmp';TMP.mkdir(parents=True,exist_ok=True)
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];ids=[r['id'] for r in refs];imid={r['id']:r['metadata']['image_id'] for r in refs}
METHODS=('rudder_clone','revis','vti');RBD='s20a1bl1p05'
print('snapshot',time.strftime('%m-%d %H:%M:%S'),'->',SNAP,flush=True)
def snap(name):
    b=(T/name).read_bytes();lines=b.split(b'\n');torn=len(lines[-1])>0;rows=[json.loads(x) for x in lines[:-1] if x.strip()]
    (SNAP/name).write_bytes(b''.join(x+b'\n' for x in lines[:-1] if x.strip()));return rows,torn
base,_=snap('base-0-500.jsonl');rbd,_=snap(f'{RBD}-0-500.jsonl');B={r['id']:r for r in base};D={r['id']:r for r in rbd}
assert all(i in B and i in D for i in ids)
runs={}
for m in METHODS:
    runs[m]={}
    for s,(a,e) in {'0-250':(0,250),'250-500':(250,500)}.items():
        rows,torn=snap(f'{m}-{a}-{e}.jsonl');assert [r['id'] for r in rows]==ids[a:a+len(rows)],(m,s)
        runs[m][s]=rows;print(f'{m} {s}: {len(rows)} rows, images {a}-{a+len(rows)-1}, torn last line dropped: {torn}',flush=True)
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['built_with_unmodified_constructor'] and c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_interim');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
def score(by,sel):
    f=TMP/'c.json';f.write_text(json.dumps([{'image_id':imid[i],'caption':by[i]['prediction']} for i in sel]))
    d=sc.compute_chair(str(f),'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];Rc=m['Recall'];s={x['image_id']:x for x in d['sentences']}
    per=dict(hal=np.array([len(s[imid[i]]['mscoco_hallucinated_words'])>0 for i in sel],float),
             n_gt=np.array([len(set(s[imid[i]]['mscoco_gt_words'])) for i in sel],float),
             n_hit=np.array([len(set(s[imid[i]]['mscoco_gt_words'])&set(s[imid[i]]['mscoco_generated_words'])) for i in sel],float))
    t=[by[i]['generated_tokens'] for i in sel]
    return dict(n=len(sel),CHAIR_s=100*m['CHAIRs'],CHAIR_i=100*m['CHAIRi'],recall=100*Rc,F1=100*2*P*Rc/(P+Rc),tok=float(np.mean(t)),at512=int(sum(x>=512 for x in t))),per
def ci(x,b):
    n=len(x['hal']);g=np.random.default_rng(0);ds=[];dr=[]
    for _ in range(2000):
        k=g.integers(0,n,n);ds.append(100*(x['hal'][k].mean()-b['hal'][k].mean()))
        dr.append(100*(x['n_hit'][k].sum()/x['n_gt'][k].sum()-b['n_hit'][k].sum()/b['n_gt'][k].sum()))
    return [float(np.quantile(ds,q)) for q in (.025,.975)],[float(np.quantile(dr,q)) for q in (.025,.975)]
out={}
def line(tag,m,cs=None):
    t=f"  {tag:13s} n {m['n']:3d}  CHAIR_s {m['CHAIR_s']:5.1f}  CHAIR_i {m['CHAIR_i']:5.2f}  recall {m['recall']:5.1f}  F1 {m['F1']:5.1f}  tok {m['tok']:4.0f}  @512 {m['at512']:3d}"
    if cs:t+=f"  dCHAIR_s[{cs[0][0]:+.1f},{cs[0][1]:+.1f}] drecall[{cs[1][0]:+.1f},{cs[1][1]:+.1f}]"
    print(t,flush=True)
print('== full 500 check (reported Base 49.2/9.6/77.2 F1 83.3, RBD 41.8/7.4/75.2 F1 83.0)',flush=True)
mb,pb=score(B,ids);md,pd=score(D,ids);line('vdd Base',mb);line('RBD',md,ci(pd,pb));out['full']={'base':mb,'rbd':md}
for m in METHODS:
    A={r['id']:r for r in runs[m]['0-250']};C={r['id']:r for r in runs[m]['250-500']};all_={**A,**C}
    for name,by in (('all',all_),('0-250',A),('250-500',C)):
        sel=[i for i in ids if i in by];print(f'== {m}, {name}: images {name if name=="all" else ""} n={len(sel)}',flush=True)
        mb,pb=score(B,sel);md,pd=score(D,sel);mx,px=score(by,sel);cd=ci(pd,pb);cx=ci(px,pb)
        line('vdd Base',mb);line('RBD',md,cd);line(m,mx,cx)
        out[f'{m}/{name}']={'base':mb,'rbd':dict(md,ci=cd),m:dict(mx,ci=cx)}
def first_div(a,b):
    x,y=a.split(),b.split()
    for k,(p,q) in enumerate(zip(x,y)):
        if p!=q:return k
    return None if len(x)==len(y) else min(len(x),len(y))
def div(name,P,Q):
    k=[i for i in P if i in Q];f=[first_div(P[i]['prediction'],Q[i]['prediction']) for i in k];d=[v for v in f if v is not None]
    q=np.quantile(d,[.1,.5,.9]) if d else [0,0,0]
    print(f"  {name:44s} n {len(k):4d} identical {100*(len(k)-len(d))/len(k):5.1f}%  first diverging word p10/p50/p90 {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}",flush=True)
print('== caption divergence (word index of the first difference)',flush=True)
rd=lambda p:{r['id']:r for r in map(json.loads,open(p))}
for m in METHODS:div(f'Qwen3 {m} (bench) vs vdd Base',{r['id']:r for s in runs[m].values() for r in s},B)
div('Qwen3 RBD vs vdd Base',D,B)
div('REF path only: Qwen2.5 AMBER bench vs vdd Base',rd(O/'runs/amber_full1004_all/base-0-1004.jsonl'),rd(O/'runs/amber_full1004_ours_all/base-0-1004.jsonl'))
div('REF path only: Gemma CHAIR table vs vdd Base',rd(O/'runs/gchair_test500_tbl/base-0-500.jsonl'),rd(O/'runs/gchair_test500_tbl/vddbase-0-500.jsonl'))
(SNAP/'interim.json').write_text(json.dumps(out,indent=1));print('INTERIM_DONE',flush=True)
