R=/home/project-b/LJW/vlm-grounding-steering; cd $R
NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts:src .venv/bin/python - <<'PY' 2>/dev/null
import json,hashlib,tempfile,os
from pathlib import Path
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102'
src=R/'upstream/chair/chair.py';c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_cs');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
refs=[json.loads(l) for l in open(O/'sets/chair_test150.jsonl')];out={r['id']:{'set':'chair_test150','id':r['id'],'image_id':r['metadata']['image_id'],
     'image':r['image'],'prompt':r['prompt'],'gt_objects':sorted(sc.imid_to_objects[r['metadata']['image_id']]),'methods':{}} for r in refs}
for tag,name in (('base','base'),('s16a2bl1p05','ours_s16a2bl1p05')):
    rows={r['id']:r for r in map(json.loads,open(O/f'runs/chair_test150_all/{tag}-0-150.jsonl'))}
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:
        json.dump([{'image_id':r['metadata']['image_id'],'caption':rows[r['id']]['prediction']} for r in refs],f)
    d=sc.compute_chair(f.name,'image_id','caption');os.unlink(f.name);s={x['image_id']:x for x in d['sentences']}
    for r in refs:
        x=s[r['metadata']['image_id']];gt=set(x['mscoco_gt_words']);gen=set(x['mscoco_generated_words'])
        out[r['id']]['methods'][name]={'caption':rows[r['id']]['prediction'],'tokens':rows[r['id']]['generated_tokens'],
            'hallucinated':[list(h) for h in x['mscoco_hallucinated_words']],'correct':sorted(gt&gen),'generated_objects':sorted(gen)}
for v in out.values():print(json.dumps(v))
PY
