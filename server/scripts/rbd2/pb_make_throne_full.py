"""Full THRONE COCO set: every COCO val2017 image (5000; THRONE's COCO evaluation uses all images of instances_val2017
when no subset is given, throne_generate.py), ascending id, prompt and max_new_tokens as throne_test150. Images from the
official val2017.zip under bench/coco2017/val2017; the first 150 rows are throne_test150 (same ids, same sha256)."""
import hashlib,json
from pathlib import Path
B=Path('/home/project-b/LJW/bench/coco2017');S=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/sets')
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
t150=read(S/'throne_test150.jsonl');tmpl=t150[0]
inst=json.load(open(B/'annotations/instances_val2017.json'));ids=sorted(i['id'] for i in inst['images'])
rows=[]
for i in ids:
    p=B/'val2017'/f'{i:012d}.jpg'
    rows.append({**tmpl,'id':f'throne:{i}','group_id':f'throne:{i}','image':str(p),
                 'image_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'metadata':{'image_id':i}})
assert len(rows)==5000 and [(r['id'],r['image_sha256']) for r in rows[:150]]==[(r['id'],r['image_sha256']) for r in t150]
(S/'throne_full.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('throne_full',len(rows),'ids',ids[0],'..',ids[-1])
