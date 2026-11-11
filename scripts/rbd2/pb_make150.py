"""Extend the test sets to 150 images in the same order (the first rows are copied unchanged):
  chair_test150   first 150 of the CHAIR test order (chair_test120 + the next 30)
  throne_test150  the 150 smallest COCO val2017 ids (throne_test120 + the next 30; new images fetched from
                  images.cocodataset.org, sha256 recorded)
  amber_test150   AMBER ids 1-100 then 201-250 (amber_test60 + the next 90); ids 101-200 are skipped because the dev
                  images come from there (amber_dev60 = ids 101-160)."""
import hashlib,json,urllib.request
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';S=O/'sets'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def write(name,rows):(S/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
def sha(p):return hashlib.sha256(open(p,'rb').read()).hexdigest()

c120=read(S/'chair_test120.jsonl');keys=list(c120[0].keys())
src=read(R/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl')
assert [r['id'] for r in c120]==[r['id'] for r in src[:120]]
c150=c120+[{k:r[k] for k in keys} for r in src[120:150]]
for r in c150[120:]:assert sha(r['image'])==r['image_sha256'],r['id']
write('chair_test150',c150)

t120=read(S/'throne_test120.jsonl')
inst=json.load(open('/home/project-b/LJW/bench/coco2017/annotations/instances_val2017.json'))
ids=sorted(i['id'] for i in inst['images']);assert [r['metadata']['image_id'] for r in t120]==ids[:120]
img=O/'coco2017/val2017';t150=list(t120)
for i in ids[120:150]:
    name=f'{i:012d}.jpg';p=img/name
    if not p.exists():p.write_bytes(urllib.request.urlopen(f'http://images.cocodataset.org/val2017/{name}',timeout=60).read())
    t150.append({**t120[0],'id':f'throne:{i}','group_id':f'throne:{i}','image':str(p),'image_sha256':sha(p),'metadata':{'image_id':i}})
write('throne_test150',t150)

a60=read(S/'amber_test60.jsonl');dev={r['metadata']['amber_id'] for r in read(S/'amber_dev60.jsonl')}
gen={r['metadata']['amber_id']:r for r in read(R/'data/amber/amber_generative.jsonl')}
aid=list(range(1,101))+list(range(201,251));assert not dev&set(aid)
def amber(i):g=gen[i];return {**a60[0],'id':g['id'],'group_id':g['group_id'],'image':g['image'],'image_sha256':g['image_sha256'],'metadata':g['metadata']}
a150=[amber(i) for i in aid];assert a150[:60]==a60,'amber_test60 rows differ from the rebuilt ids 1-60'
for r in a150[60:]:assert sha(r['image'])==r['image_sha256'],r['id']
write('amber_test150',a150)
print('chair_test150',len(c150),'| throne_test150',len(t150),'new ids',ids[120],'..',ids[149],'| amber_test150',len(a150),'ids 1-100, 201-250')
