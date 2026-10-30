"""Sample 5000 COCO train2014 images (seed 42) and write the list + 80-class multi-hot labels."""
import json,random
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'data/coco-train2014-probe5000'
inst=json.loads((R/'data/hallucination-suite/annotations/instances_train2014.json').read_text())
cats=sorted(inst['categories'],key=lambda c:c['id']);names=[c['name'] for c in cats];idx={c['id']:i for i,c in enumerate(cats)}
assert len(cats)==80
labels={}
for a in inst['annotations']:labels.setdefault(a['image_id'],set()).add(idx[a['category_id']])
imgs=sorted(inst['images'],key=lambda x:x['id']);random.Random(42).shuffle(imgs);pick=imgs[:5000]
O.mkdir(parents=True,exist_ok=False);(O/'images').mkdir()
with (O/'list.jsonl').open('x') as f:
    for im in pick:
        f.write(json.dumps({'id':f"coco_train:{im['id']}",'image_id':im['id'],'file_name':im['file_name'],
            'image':str(O/'images'/im['file_name']),'url':im['coco_url'],'objects':sorted(labels.get(im['id'],()))})+'\n')
(O/'categories.json').write_text(json.dumps(names))
print('images',len(pick),'with_objects',sum(1 for im in pick if im['id'] in labels))
