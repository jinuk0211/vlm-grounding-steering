"""Build the unmodified CHAIR scorer once (it tokenizes every COCO 2014 caption: >10 min) and cache its state for the
first 100 CHAIR test images, in the format make_dev_chair.py / score_dev.py use. Then check that the cached scorer
reproduces the stored official scores of the A5000 Base on the first 50 test images (CHAIR_s 0.28, CHAIR_i 0.12069,
recall 0.55346, from score-pilot-chair50/score.json)."""
import hashlib,json
from pathlib import Path
from vlm_grounding.evaluation import _load_external_module

R=Path('/home/project-b/LJW/vlm-grounding-steering');SRC=R/'upstream/chair/chair.py';OUT=Path('/workspace/rbd_pilot/chair-test100-scorer-state.json')
ids={json.loads(l)['metadata']['image_id'] for l in open('/workspace/chair_test100.jsonl')}
mod=_load_external_module(SRC,'original_chair_test100_build');sc=mod.CHAIR(str(R/'data/hallucination-suite/annotations'))
state={'imid_to_objects':{str(k):sorted(set(v)) for k,v in sc.imid_to_objects.items() if k in ids},'coco_path':sc.coco_path,
       'mscoco_objects':sc.mscoco_objects,'inverse_synonym_dict':sc.inverse_synonym_dict,'double_word_dict':sc.double_word_dict}
assert len(state['imid_to_objects'])==100
OUT.write_text(json.dumps({'built_with_unmodified_constructor':True,'scorer_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'state':state}))

def load():
    c=json.loads(OUT.read_text());s=mod.CHAIR.__new__(mod.CHAIR);s.__dict__.update(c['state'])
    s.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()};return s
m=load().compute_chair('/workspace/rbd_pilot/qwen_test_base50.captions.json','image_id','caption')['overall_metrics']
print({k:round(v,5) for k,v in m.items()})
assert abs(m['CHAIRs']-0.28)<1e-9 and abs(m['CHAIRi']-0.12069)<1e-5 and abs(m['Recall']-0.55346)<1e-5
print('CHAIR_CACHE_OK')
