"""CHAIR on the first 100 CHAIR test images: unmodified CHAIR constructor (Maxlinn port, sha256 98728cd2...) on the
COCO 2014 annotations, cached by chair_cache.py; same per-image arrays and paired bootstrap (2000, seed 0) as
scripts/rbd_score.py.
usage: score_chair100.py OUT_JSON name=rows.jsonl [name=rows.jsonl ...]   (first name = Base)"""
import hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module

R=Path('/home/project-b/LJW/vlm-grounding-steering');SRC=R/'upstream/chair/chair.py'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def main(out,specs):
    ids=[r['metadata']['image_id'] for r in read('/workspace/chair_test100.jsonl')]
    cache=json.loads(Path('/workspace/rbd_pilot/chair-test100-scorer-state.json').read_text())   # chair_cache.py
    assert cache['built_with_unmodified_constructor'] and cache['scorer_sha256']==hashlib.sha256(SRC.read_bytes()).hexdigest()
    mod=_load_external_module(SRC,'original_chair_test100');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(cache['state'])
    sc.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
    res={};per={}
    for spec in specs:
        name,path=spec.split('=',1);rows={r['metadata']['image_id']:r for r in read(path)}
        if any(i not in rows for i in ids):print(name,'incomplete',len(rows));continue
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:json.dump([{'image_id':i,'caption':rows[i]['prediction']} for i in ids],f)
        d=sc.compute_chair(f.name,'image_id','caption');m=d['overall_metrics'];P=1-m['CHAIRi'];Rc=m['Recall']
        s={x['image_id']:x for x in d['sentences']}
        per[name]={'hal':np.array([len(s[i]['mscoco_hallucinated_words'])>0 for i in ids],float),
                   'n_gt':np.array([len(set(s[i]['mscoco_gt_words'])) for i in ids],float),
                   'n_hit':np.array([len(set(s[i]['mscoco_gt_words'])&set(s[i]['mscoco_generated_words'])) for i in ids],float)}
        res[name]={'CHAIR_s':100*m['CHAIRs'],'CHAIR_i':100*m['CHAIRi'],'recall':100*Rc,'object_F1':100*2*P*Rc/(P+Rc),
                   'mean_tokens':float(np.mean([rows[i]['generated_tokens'] for i in ids])),
                   'at_512':int(sum(rows[i]['generated_tokens']>=512 for i in ids))}
    base=next(iter(per));rng=np.random.default_rng(0);B=[rng.integers(0,len(ids),len(ids)) for _ in range(2000)]
    for name in per:
        if name==base:continue
        x,b=per[name],per[base]
        ds=[100*(x['hal'][k].mean()-b['hal'][k].mean()) for k in B]
        dr=[100*(x['n_hit'][k].sum()/x['n_gt'][k].sum()-b['n_hit'][k].sum()/b['n_gt'][k].sum()) for k in B]
        res[name]['dCHAIR_s_ci95']=[float(np.quantile(ds,.025)),float(np.quantile(ds,.975))]
        res[name]['drecall_ci95']=[float(np.quantile(dr,.025)),float(np.quantile(dr,.975))]
    Path(out).write_text(json.dumps({'n_images':len(ids),'results':res},indent=1))
    for k,v in res.items():
        ci=f"  dCHs [{v['dCHAIR_s_ci95'][0]:+.1f},{v['dCHAIR_s_ci95'][1]:+.1f}] drec [{v['drecall_ci95'][0]:+.1f},{v['drecall_ci95'][1]:+.1f}]" if 'dCHAIR_s_ci95' in v else ''
        print(f"{k:22s} CHs {v['CHAIR_s']:5.1f} CHi {v['CHAIR_i']:5.2f} rec {v['recall']:5.1f} F1 {v['object_F1']:5.2f} tok {v['mean_tokens']:4.0f} @512 {v['at_512']}{ci}")

if __name__=='__main__':main(sys.argv[1],sys.argv[2:])
