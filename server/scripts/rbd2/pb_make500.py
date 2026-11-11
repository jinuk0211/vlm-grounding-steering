"""Extend the test sets to 500 images in the same order (the first 150 rows are the *_test150 rows unchanged):
  chair_test500   the whole CHAIR test split (500), in its order
  throne_test500  the 500 smallest COCO val2017 ids (= the first 500 rows of throne_full)
  amber_test500   AMBER ids 1-100 then 201-600; ids 101-200 stay out (dev images come from there)"""
import hashlib,json
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');S=R/'outputs/rbd2-20261102/sets'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def write(name,rows):(S/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
def sha(p):return hashlib.sha256(open(p,'rb').read()).hexdigest()

c150=read(S/'chair_test150.jsonl');keys=list(c150[0].keys())
src=read(R/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl')
assert len(src)==500 and [r['id'] for r in c150]==[r['id'] for r in src[:150]]
c500=c150+[{k:r[k] for k in keys} for r in src[150:]]
for r in c500[150:]:assert sha(r['image'])==r['image_sha256'],r['id']
write('chair_test500',c500)

t150=read(S/'throne_test150.jsonl');full=read(S/'throne_full.jsonl')
assert [(r['id'],r['image_sha256']) for r in t150]==[(r['id'],r['image_sha256']) for r in full[:150]]
write('throne_test500',t150+full[150:500])

a150=read(S/'amber_test150.jsonl');dev={r['metadata']['amber_id'] for r in read(S/'amber_dev60.jsonl')}
gen={r['metadata']['amber_id']:r for r in read(R/'data/amber/amber_generative.jsonl')}
aid=list(range(1,101))+list(range(201,601));assert not dev&set(aid) and len(aid)==500
def amber(i):g=gen[i];return {**a150[0],'id':g['id'],'group_id':g['group_id'],'image':g['image'],'image_sha256':g['image_sha256'],'metadata':g['metadata']}
a500=[amber(i) for i in aid];assert a500[:150]==a150
for r in a500[150:]:assert sha(r['image'])==r['image_sha256'],r['id']
write('amber_test500',a500)
print('chair_test500',len(c500),'| throne_test500',len(t150)+350,'| amber_test500',len(a500),'ids 1-100, 201-600')
