"""Qwen3-VL baselines vs vdd Base and RBD on the same images, with and without class orange. CPU only; runs/qchair_test500_tbl is only read; complete
lines are copied to SNAP. Same scorer and paired bootstrap (2000, seed 0) as score_sets.py.   usage: qchair_report.py SNAP"""
import hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';T=O/'runs/qchair_test500_tbl';SNAP=Path(sys.argv[1]);TMP=SNAP/'tmp';TMP.mkdir(parents=True,exist_ok=True)
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];ids=[r['id'] for r in refs];IM={r['id']:r['metadata']['image_id'] for r in refs}
print('snapshot',time.strftime('%m-%d %H:%M:%S'),SNAP,flush=True)
def snap(name):
    b=(T/name).read_bytes();lines=b.split(b'\n');(SNAP/name).write_bytes(b''.join(x+b'\n' for x in lines[:-1] if x.strip()))
    return [json.loads(x) for x in lines[:-1] if x.strip()],len(lines[-1])>0
B={r['id']:r for r in snap('base-0-500.jsonl')[0]};D={r['id']:r for r in snap('s20a1bl1p05-0-500.jsonl')[0]}
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['built_with_unmodified_constructor'] and c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_rep');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state']);sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
def score(by,sel):
    f=TMP/'c.json';f.write_text(json.dumps([{'image_id':IM[i],'caption':by[i]['prediction']} for i in sel]))
    d=sc.compute_chair(str(f),'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];Rc=m['Recall'];s={x['image_id']:x for x in d['sentences']}
    hw=[{n for _,n in s[IM[i]]['mscoco_hallucinated_words']} for i in sel];t=[by[i]['generated_tokens'] for i in sel]
    p=dict(hal=np.array([bool(x) for x in hw],float),halx=np.array([bool(x-{'orange'}) for x in hw],float),gt=np.array([len(set(s[IM[i]]['mscoco_gt_words'])) for i in sel],float),hit=np.array([len(set(s[IM[i]]['mscoco_gt_words'])&set(s[IM[i]]['mscoco_generated_words'])) for i in sel],float))
    return dict(n=len(sel),CHAIR_s=100*m['CHAIRs'],noorange=100*p['halx'].mean(),CHAIR_i=100*m['CHAIRi'],recall=100*Rc,F1=100*2*P*Rc/(P+Rc),tok=float(np.mean(t)),at512=int(sum(x>=512 for x in t))),p
def ci(a,b):
    n=len(a['hal']);g=np.random.default_rng(0);ds=[];dx=[];dr=[]
    for _ in range(2000):
        k=g.integers(0,n,n);ds.append(100*(a['hal'][k].mean()-b['hal'][k].mean()));dx.append(100*(a['halx'][k].mean()-b['halx'][k].mean()));dr.append(100*(a['hit'][k].sum()/a['gt'][k].sum()-b['hit'][k].sum()/b['gt'][k].sum()))
    q=lambda v:'[%+.1f,%+.1f]'%(np.quantile(v,.025),np.quantile(v,.975));return f"s{q(ds)} no-orange{q(dx)} r{q(dr)}"
def line(t,v,x=''):print(f"  {t:13s} n {v['n']:3d} CHAIR_s {v['CHAIR_s']:5.1f} no-orange {v['noorange']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} F1 {v['F1']:5.1f} tok {v['tok']:4.0f} @512 {v['at512']:3d} {x}",flush=True)
def loop(s):
    w=s.split()
    if len(w)<40:return False
    for n in (3,5,8,12):
        c_={}
        for i in range(len(w)-n+1):
            k=' '.join(w[i:i+n]);c_[k]=c_.get(k,0)+1
        if max(c_.values())>=(6 if n<=5 else 4):return True
    return False
vb,pb=score(B,ids);vd,pd=score(D,ids);print('== all 500');line('vdd Base',vb);line('RBD',vd,'vs Base '+ci(pd,pb))
for m in ('rudder_clone','revis','vti'):
    rows=[]
    for a,e in ((0,250),(250,500)):
        r,torn=snap(f'{m}-{a}-{e}.jsonl');rows+=r
    by={r['id']:r for r in rows};sel=[i for i in ids if i in by]
    print(f"== {m} n={len(sel)} {'FINAL' if len(sel)==500 else 'INTERIM'} dup {len(rows)-len(by)}",flush=True)
    vb,pb=score(B,sel);vd,pd=score(D,sel);vx,px=score(by,sel);line('vdd Base',vb);line('RBD',vd,'vs Base '+ci(pd,pb));line(m,vx,'vs Base '+ci(px,pb)+' | vs RBD '+ci(px,pd))
    same=sum(by[i]['prediction']==B[i]['prediction'] for i in sel)
    print(f"  captions: addCriterion {sum('addCriterion' in by[i]['prediction'] for i in sel)} (Base {sum('addCriterion' in B[i]['prediction'] for i in sel)}), loop {sum(loop(by[i]['prediction']) for i in sel)} (Base {sum(loop(B[i]['prediction']) for i in sel)}), empty {sum(not by[i]['prediction'].strip() for i in sel)}, same as vdd Base {same}",flush=True)
print('REPORT_DONE',flush=True)
