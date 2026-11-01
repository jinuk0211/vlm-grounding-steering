"""Downloads for the benchmark setup, through pdl.py's parallel range fetch (this host caps each connection).
  fetchx.py hf REPO REVISION OUTDIR        only the PyTorch-side files (safetensors, configs, tokenizer); LFS sha256 checked
  fetchx.py url URL OUTFILE [SHA256]       one file; size from a HEAD request, sha256 checked when given"""
import json,os,sys,urllib.request
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pdl

KEEP=('config.json','generation_config.json','spiece.model','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
      'model.safetensors.index.json')

def hf(repo,rev,out):
    tree=json.loads(pdl.get(f'https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true'))
    for x in sorted((x for x in tree if x['type']=='file'),key=lambda x:x['size']):
        if not (x['path'] in KEEP or x['path'].endswith('.safetensors')):continue
        p=os.path.join(out,x['path']);os.makedirs(os.path.dirname(p),exist_ok=True)
        oid=(x.get('lfs') or {}).get('oid');url=f'https://huggingface.co/{repo}/resolve/{rev}/{x["path"]}'
        if os.path.exists(p) and os.path.getsize(p)==x['size'] and (not oid or pdl.sha256(p)==oid):continue
        if x['size']<pdl.CHUNK:open(p,'wb').write(pdl.get(url))
        else:pdl.fetch(url,p,x['size'],32)
        assert os.path.getsize(p)==x['size'],p
        if oid:assert pdl.sha256(p)==oid,f'sha256 mismatch {p}'
        print('ok',repo,x['path'],x['size'],flush=True)
    print('HF_OK',repo,rev,flush=True)

def url(u,out,sha=None):
    os.makedirs(os.path.dirname(out) or '.',exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(u,method='HEAD'),timeout=60) as r:size=int(r.headers['Content-Length']);final=r.url
    if not (os.path.exists(out) and os.path.getsize(out)==size):pdl.fetch(final,out,size,32)
    assert os.path.getsize(out)==size,out
    if sha:assert pdl.sha256(out)==sha,f'sha256 mismatch {out}'
    print('URL_OK',out,size,pdl.sha256(out),flush=True)

if __name__=='__main__':{'hf':hf,'url':url}[sys.argv[1]](*sys.argv[2:])
