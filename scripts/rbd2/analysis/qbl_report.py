"""Qwen3-VL CHAIR 500: official REVIS and GrAInS (runs/qchair_bl2) next to vdd Base, RBD and the earlier rows, with and without class orange. CPU only; same scorer and paired bootstrap as score_sets.py."""
import hashlib,json
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';T=O/'runs/qchair_test500_tbl';N=O/'runs/qchair_bl2';TMP=N/'tmp';TMP.mkdir(parents=True,exist_ok=True)
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];ids=[r['id'] for r in refs];IM={r['id']:r['metadata']['image_id'] for r in refs}
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['built_with_unmodified_constructor'] and c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_qbl');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state']);sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
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
def line(t,v,x=''):print(f"  {t:15s} n {v['n']:3d} CHAIR_s {v['CHAIR_s']:5.1f} no-orange {v['noorange']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} F1 {v['F1']:5.1f} tok {v['tok']:4.0f} @512 {v['at512']:3d} {x}",flush=True)
rd=lambda p:{r['id']:r for r in map(json.loads,open(p))}
def parts(d,m):
    by={}
    for f in Path(d).glob(f'{m}-*-*.jsonl'):by.update(rd(f))
    return by
B=rd(T/'base-0-500.jsonl');D=rd(T/'s20a1bl1p05-0-500.jsonl');vb,pb=score(B,ids);vd,pd=score(D,ids);print('== Qwen3-VL CHAIR 500');line('vdd Base',vb);line('RBD',vd,'vs Base '+ci(pd,pb))
for name,by in (('REVIS own (old)',parts(T,'revis')),('RUDDER',parts(T,'rudder_clone')),('VTI',parts(T,'vti')),('REVIS official',parts(N,'revis_official')),('GrAInS',parts(N,'grains_clone'))):
    sel=[i for i in ids if i in by]
    if not sel:print('  ',name,'no rows');continue
    bb,qb=score(B,sel);bd,qd=score(D,sel);v,p=score(by,sel);line(name,v,'vs Base '+ci(p,qb)+' | vs RBD '+ci(p,qd))
    if len(sel)<500:print('   INCOMPLETE; Base on these images:',round(bb['CHAIR_s'],1),round(bb['recall'],1))
    fired=[(json.loads(json.dumps(by[i]['author_diagnostics']))) for i in sel]
    if 'fired' in fired[0]:
        a=sum(int(x['fired']) for x in fired);b=sum(int(x['calls']) for x in fired);print(f"   gate open on {100*a/b:.1f}% of steps")
print('REPORT_DONE',flush=True)
