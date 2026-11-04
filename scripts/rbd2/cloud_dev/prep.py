"""Put the inputs rbd_decode.py expects where it expects them (the old server's paths), and fetch the 100+100 images.

- model config.json under read_gap_collect_v2.MODELS (configs/*.json model block, weights already in HF_HOME)
- r(t) source: results/visual_decay_20261030/*-summary.json -> outputs/visual-decay-20261030/
- V_obj: rbd_common.build_vobj on each model's tokenizer -> outputs/rbd-20261030/collect/{model}-vobj2.json
- CHAIR test images (first 100 of the 500 test manifest) and AMBER images, sha256-checked against the manifests
"""
import hashlib,json,shutil,sys,urllib.request,zipfile
from pathlib import Path
import read_gap_collect_v2 as C
from rbd_common import ROOT,build_vobj

W=Path('/workspace')
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

LOCAL={'qwen':'/workspace/models/qwen2.5-vl-7b-instruct','gemma':'/workspace/models/gemma-4-e4b-it'}  # pdl.py, pinned revision

def configs():
    for m,f in (('qwen','qwen2_5_vl_7b.json'),('gemma','gemma4_e4b.json')):
        model=json.loads((ROOT/'configs'/f).read_text())['model'];model['local_files_only']=True
        model['hub_checkpoint']=model['checkpoint'];model['checkpoint']=LOCAL[m]
        C.MODELS[m].mkdir(parents=True,exist_ok=True);(C.MODELS[m]/'config.json').write_text(json.dumps({'model':model},indent=1))
    d=ROOT/'outputs/visual-decay-20261030';d.mkdir(parents=True,exist_ok=True)
    for m in ('qwen','gemma'):shutil.copy(ROOT/f'results/visual_decay_20261030/{m}-visual-decay-summary.json',d)

def vobj():
    from transformers import AutoProcessor
    d=ROOT/'outputs/rbd-20261030/collect';d.mkdir(parents=True,exist_ok=True)
    for m in ('qwen','gemma'):
        if not Path(LOCAL[m],'tokenizer_config.json').exists():print(m,'weights not downloaded; V_obj skipped');continue
        cfg=json.loads((C.MODELS[m]/'config.json').read_text())['model']
        tok=AutoProcessor.from_pretrained(cfg['checkpoint'],revision=cfg['revision'],local_files_only=True).tokenizer
        ids=build_vobj(tok);(d/f'{m}-vobj2.json').write_text(json.dumps(ids))
        print(m,'V_obj',len(ids),[tok.convert_ids_to_tokens(i) for i in ids[::max(1,len(ids)//25)]])

def chair_images():
    for r in read(W/'chair_test100.jsonl'):
        p=Path(r['image']);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():urllib.request.urlretrieve('http://images.cocodataset.org/val2014/'+p.name,p)
        assert sha(p)==r['image_sha256'],p
    print('chair test images ok')

def amber_images():
    z=W/'AMBER_images.zip'
    assert sha(z)=='b4b95686f4f562eab09d449c2e6483df4071e312e73394a2741d5f76f851489c','AMBER zip sha256 mismatch'
    rows=read(W/'amber100.jsonl');root=Path(rows[0]['image']).parents[1]
    if not Path(rows[0]['image']).exists():
        root.mkdir(parents=True,exist_ok=True);zipfile.ZipFile(z).extractall(root)
    for r in rows:assert sha(r['image'])==r['image_sha256'],r['image']
    print('amber images ok')

if __name__=='__main__':
    for step in sys.argv[1:]:globals()[step]()
