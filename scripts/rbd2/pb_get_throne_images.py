"""Fetch the 60 COCO val2017 images of throne_test60 from images.cocodataset.org, check each against the manifest's
image_sha256, and write the manifest with project-b paths. usage: pb_get_throne_images.py IN.jsonl OUT.jsonl IMGDIR"""
import hashlib,json,sys,urllib.request
from pathlib import Path
src,dst,img=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]);img.mkdir(parents=True,exist_ok=True);rows=[]
for l in src.read_text().splitlines():
    r=json.loads(l);name=Path(r['image']).name;p=img/name
    if not (p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256']):
        data=urllib.request.urlopen(f'http://images.cocodataset.org/val2017/{name}',timeout=60).read()
        assert hashlib.sha256(data).hexdigest()==r['image_sha256'],f'sha256 mismatch {name}';p.write_bytes(data)
    rows.append({**r,'image':str(p)})
dst.write_text(''.join(json.dumps(r)+'\n' for r in rows));print('ok',len(rows),'images verified')
