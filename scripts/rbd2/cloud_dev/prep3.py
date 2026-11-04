"""Manifests for the added benchmarks (same row format as the other sets; max_new_tokens 512 everywhere):
  objhal_test60  Object HalBench (RLHF-V eval/data/obj_halbench_300_with_image.jsonl, commit 3863e39d), first 60 rows,
                 each with its own prompt; images decoded from the embedded base64 to files
  amber_dev60L / amber_test60L  AMBER dev/test rows with the CHAIR prompt "Please describe this image in detail."
                 (not the AMBER protocol: a long-caption analysis variant)
  throne_test60  THRONE COCO: first 60 of the sorted val2017 image ids (as throne_generate.py orders them), prompt
                 "Describe this image in detail." (throne_generate.GENERATION_PROMPT); images downloaded and decode-checked
usage: prep3.py objhal | amberL | throne"""
import base64,concurrent.futures as cf,hashlib,io,json,sys,time,urllib.request
from pathlib import Path
from PIL import Image

W=Path('/workspace');S=W/'sets';B=W/'bench'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,rows):Path(p).write_text(''.join(json.dumps(r)+'\n' for r in rows))
def row(i,img,prompt,bench,meta):
    return {'id':i,'group_id':i,'task':'caption','benchmark':bench,'split':'test','image':str(img),'prompt':prompt,'max_new_tokens':512,
            'image_sha256':hashlib.sha256(Path(img).read_bytes()).hexdigest(),'metadata':meta}

def objhal():
    d=B/'objhal_images';d.mkdir(exist_ok=True);out=[]
    for r in read(B/'RLHF-V/eval/data/obj_halbench_300_with_image.jsonl')[:60]:
        p=d/f"{r['org_idx']}_{r['image_id']}.jpg"
        if not p.exists():
            im=Image.open(io.BytesIO(base64.b64decode(r['image'])));im.load();p.write_bytes(base64.b64decode(r['image']))
        out.append(row(f"objhal:{r['org_idx']}",p,r['question'],'object_halbench',{'image_id':r['image_id'],'org_idx':r['org_idx']}))
    write(S/'objhal_test60.jsonl',out);print('objhal_test60',len(out),'distinct images',len({r['metadata']['image_id'] for r in out}))

def amberL():
    for s in ('amber_dev60','amber_test60'):
        rows=[{**r,'prompt':'Please describe this image in detail.','benchmark':'amber_generative_longprompt'} for r in read(S/f'{s}.jsonl')]
        write(S/f'{s}L.jsonl',rows);print(f'{s}L',len(rows))

def get(im):
    p=B/'coco2017/val2017'/im['file_name']
    for k in range(8):
        try:
            if not p.exists():
                data=urllib.request.urlopen(im['coco_url'],timeout=60).read();t=p.with_suffix('.part');t.write_bytes(data);t.rename(p)
            with Image.open(p) as x:x.convert('RGB')
            return p
        except Exception:
            p.unlink(missing_ok=True);time.sleep(2+k)
    raise RuntimeError(im['coco_url'])

def throne():
    ann=json.loads((B/'coco2017/annotations/instances_val2017.json').read_text())
    ims=sorted(ann['images'],key=lambda x:x['id'])[:60]
    with cf.ThreadPoolExecutor(16) as ex:paths=list(ex.map(get,ims))
    out=[row(f"throne:{im['id']}",p,'Describe this image in detail.','throne_coco',{'image_id':im['id']}) for im,p in zip(ims,paths)]
    write(S/'throne_test60.jsonl',out);print('throne_test60',len(out))

if __name__=='__main__':{'objhal':objhal,'amberL':amberL,'throne':throne}[sys.argv[1]]()
