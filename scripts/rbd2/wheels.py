"""Fetch the torch 2.8.0+cu128 wheel set (versions from `uv pip install --dry-run`) from the PyTorch index with
parallel range requests, checking the sha256 the index publishes. usage: python3 wheels.py OUTDIR name==ver ..."""
import html,os,re,sys,urllib.parse,urllib.request
import pdl

IDX='https://download.pytorch.org/whl/cu128/'

def pick(name,ver):
    page=pdl.get(IDX+name.lower().replace('_','-')+'/').decode()
    for href in re.findall(r'href="([^"]+)"',page):
        url,_,frag=html.unescape(href).partition('#');fn=urllib.parse.unquote(url.rsplit('/',1)[1])
        n,v,*tags=fn[:-4].split('-')
        if n.lower()!=name.lower().replace('-','_') or v!=ver:continue
        py,abi,plat=tags[-3:]
        if py in ('cp312','py3') and (plat=='any' or ('x86_64' in plat and 'manylinux' in plat and 'musl' not in plat)):
            return urllib.parse.urljoin(IDX,url),fn,frag.removeprefix('sha256=')
    raise LookupError(f'{name}=={ver}')

def pypi_sha(name,ver,fn):
    """Index entries without a #sha256= fragment: take PyPI's digest for the same file name, if PyPI has it."""
    import json
    try:d=json.loads(pdl.get(f'https://pypi.org/pypi/{name}/{ver}/json',tries=3))
    except Exception:return ''
    return next((u['digests']['sha256'] for u in d['urls'] if u['filename']==fn),'')

def ok(p,want):
    if want:return pdl.sha256(p)==want
    import zipfile
    with zipfile.ZipFile(p) as z:return z.testzip() is None   # no published hash: per-member CRC32

def size(url):
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:return int(r.headers['Content-Length'])

def main(out,specs):
    os.makedirs(out,exist_ok=True)
    for s in specs:
        name,ver=s.split('==');url,fn,want=pick(name,ver);want=want or pypi_sha(name,ver,fn);p=os.path.join(out,fn)
        if os.path.exists(p) and ok(p,want):continue
        n=size(url)
        if n<pdl.CHUNK:
            with open(p,'wb') as f:f.write(pdl.get(url))
        else:pdl.fetch(url,p,n,32)
        assert ok(p,want),f'integrity check failed {fn}';print('ok',fn,n,'sha256' if want else 'zip-crc',flush=True)
    print('WHEELS_OK',flush=True)

if __name__=='__main__':main(sys.argv[1],sys.argv[2:])
