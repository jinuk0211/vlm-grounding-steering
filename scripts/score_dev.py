"""Score caption runs on the train2014 dev split with the unmodified CHAIR scorer (dev ground truth)."""
import glob,json,re,sys
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
from qwen_gpu0_migration_api import save,sha
R=Path('/home/project-b/LJW/vlm-grounding-steering');D=R/'data/chair-dev-train2014-100'
cache=json.loads((D/'chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert cache['built_with_unmodified_constructor'] and sha(src)==cache['scorer_sha256']
mod=_load_external_module(src,'original_chair_devscore');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(cache['state'])
sc.imid_to_objects={int(k):set(v) for k,v in cache['state']['imid_to_objects'].items()}
refs=[json.loads(l) for l in (D/'chair.jsonl').read_text().splitlines()];ids=[r['id'] for r in refs]
outdir=Path(sys.argv[1]);res={}
for tag in sys.argv[2:]:
    rows={}
    for f in glob.glob(str(outdir/f'{tag}-*-*.jsonl')):
        for l in open(f):r=json.loads(l);rows[r['id']]=r
    if set(rows)!=set(ids):print(tag,'incomplete',len(rows));continue
    rows=[rows[i] for i in ids];p=outdir/f'{tag}-dev-captions.json'
    save(p,[{'image_id':r['metadata']['image_id'],'caption':r['prediction']} for r in rows])
    d=sc.compute_chair(str(p),'image_id','caption');m=d['overall_metrics']
    P=1-m['CHAIRi'];Rc=m['Recall']
    res[tag]={'CHAIR_s':m['CHAIRs'],'CHAIR_i':m['CHAIRi'],'recall':Rc,'object_F1':2*P*Rc/(P+Rc),
              'mean_tokens':float(np.mean([r['generated_tokens'] for r in rows])),'rejections':sum(len(r.get('rejections',[])) for r in rows)}
    v=res[tag];print(f"{tag:22s} CHs {100*v['CHAIR_s']:5.1f} CHi {100*v['CHAIR_i']:5.2f} rec {100*v['recall']:5.1f} F1 {100*v['object_F1']:5.2f} tok {v['mean_tokens']:5.0f} rej {v['rejections']}")
save(outdir/'dev-score.json',res)
