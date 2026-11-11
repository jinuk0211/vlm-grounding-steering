"""Development split for caption experiments: first 100 images of the seeded
COCO train2014 probe sample (disjoint from the val2014 CHAIR/POPE test images).
CHAIR ground truth is built with the unmodified CHAIR constructor."""
import hashlib,json
from pathlib import Path
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'data/chair-dev-train2014-100';O.mkdir(exist_ok=False)
items=[json.loads(l) for l in (R/'data/coco-train2014-probe5000/list.jsonl').read_text().splitlines()][:100]
with (O/'chair.jsonl').open('x') as f:
    for it in items:
        f.write(json.dumps({'id':f"chairdev:{it['image_id']}",'group_id':f"coco_train:{it['image_id']}",'task':'caption','benchmark':'chair_dev','split':'dev',
            'image':it['image'],'prompt':'Please describe this image in detail.','max_new_tokens':512,
            'image_sha256':hashlib.sha256(Path(it['image']).read_bytes()).hexdigest(),'metadata':{'image_id':it['image_id']}})+'\n')
src=R/'upstream/chair/chair.py';mod=_load_external_module(src,'original_chair_dev')
sc=mod.CHAIR(str(R/'data/hallucination-suite/annotations'))
ids={it['image_id'] for it in items}
state={'imid_to_objects':{str(k):sorted(set(v)) for k,v in sc.imid_to_objects.items() if k in ids},'coco_path':sc.coco_path,
       'mscoco_objects':sc.mscoco_objects,'inverse_synonym_dict':sc.inverse_synonym_dict,'double_word_dict':sc.double_word_dict}
assert len(state['imid_to_objects'])==100
(O/'chair-scorer-state.json').write_text(json.dumps({'built_with_unmodified_constructor':True,'scorer_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'state':state}))
print('dev images',len(items),'with objects',sum(bool(v) for v in state['imid_to_objects'].values()))
