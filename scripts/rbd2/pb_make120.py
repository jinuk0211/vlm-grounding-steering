"""Extend the 60-image test sets to 120 in the same order: chair_test120 = first 120 of the CHAIR test order
(chair_test60 + the next 60), throne_test120 = the 120 smallest COCO val2017 image ids (throne_test60 + the next 60;
new images fetched from images.cocodataset.org, sha256 recorded). The first 60 rows are copied unchanged."""
import hashlib,json,urllib.request
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';S=O/'sets'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
c60=read(S/'chair_test60.jsonl');keys=list(c60[0].keys())
src=read(R/'outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl')
assert [r['id'] for r in c60]==[r['id'] for r in src[:60]]
c120=c60+[{k:r[k] for k in keys} for r in src[60:120]]
for r in c120[60:]:assert hashlib.sha256(open(r['image'],'rb').read()).hexdigest()==r['image_sha256'],r['id']
(S/'chair_test120.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in c120))
t60=read(S/'throne_test60.jsonl')
inst=json.load(open('/home/project-b/LJW/bench/coco2017/annotations/instances_val2017.json'))
ids=sorted(i['id'] for i in inst['images']);assert [r['metadata']['image_id'] for r in t60]==ids[:60]
img=O/'coco2017/val2017';t120=list(t60)
for i in ids[60:120]:
    name=f'{i:012d}.jpg';p=img/name
    if not p.exists():p.write_bytes(urllib.request.urlopen(f'http://images.cocodataset.org/val2017/{name}',timeout=60).read())
    t120.append({**t60[0],'id':f'throne:{i}','group_id':f'throne:{i}','image':str(p),
                 'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metadata':{'image_id':i}})
(S/'throne_test120.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in t120))
print('chair_test120',len(c120),'throne_test120',len(t120),'| new throne ids',ids[60],'..',ids[119])
