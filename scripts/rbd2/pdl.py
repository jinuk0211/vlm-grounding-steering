"""Parallel range download of a pinned Hugging Face model revision (this host caps each connection near 150 KB/s).

usage: python3 pdl.py REPO REVISION OUTDIR [connections]
Every file of the revision goes to OUTDIR/<path>; LFS files are checked against their sha256 oid. Resumable: a
file whose size and hash already match is skipped. Standard library only.
"""
import hashlib,json,os,queue,sys,threading,time,urllib.request

CHUNK=4*2**20   # this host drops transfers that run past ~100 s

def get(url,rng=None,tries=20):
    for k in range(tries):
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={rng[0]}-{rng[1]}'} if rng else {})
            with urllib.request.urlopen(req,timeout=120) as r:return r.read()
        except Exception:
            if k==tries-1:raise
            time.sleep(min(30,2**k))

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**24),b''):h.update(b)
    return h.hexdigest()

def fetch(url,path,size,conns):
    with open(path,'wb') as f:f.truncate(size)
    q=queue.Queue()
    for s in range(0,size,CHUNK):q.put((s,min(size,s+CHUNK)-1))
    fd=os.open(path,os.O_WRONLY);done=[0];lock=threading.Lock();err=[]
    def work():
        while not err:
            try:s,e=q.get_nowait()
            except queue.Empty:return
            try:
                b=get(url,(s,e));assert len(b)==e-s+1,(len(b),e-s+1);os.pwrite(fd,b,s)
                with lock:done[0]+=len(b)
            except Exception as x:err.append(x)
    ts=[threading.Thread(target=work,daemon=True) for _ in range(conns)];t0=time.time()
    for t in ts:t.start()
    while any(t.is_alive() for t in ts):
        time.sleep(30);print(f'  {os.path.basename(path)} {done[0]/2**20:.0f}/{size/2**20:.0f} MB {done[0]/2**20/(time.time()-t0):.1f} MB/s',flush=True)
    os.close(fd)
    if err:raise err[0]

def main(repo,rev,out,conns=64):
    tree=json.loads(get(f'https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true'))
    files=sorted((x for x in tree if x['type']=='file'),key=lambda x:x['size'])
    for x in files:
        p=os.path.join(out,x['path']);os.makedirs(os.path.dirname(p),exist_ok=True)
        oid=(x.get('lfs') or {}).get('oid');url=f'https://huggingface.co/{repo}/resolve/{rev}/{x["path"]}'
        if os.path.exists(p) and os.path.getsize(p)==x['size'] and (not oid or sha256(p)==oid):continue
        if x['size']<CHUNK:
            with open(p,'wb') as f:f.write(get(url))
        else:fetch(url,p,x['size'],conns)
        assert os.path.getsize(p)==x['size'],p
        if oid:assert sha256(p)==oid,f'sha256 mismatch {p}'
        print('ok',x['path'],x['size'],flush=True)
    print('ALL_OK',repo,rev,flush=True)

if __name__=='__main__':main(sys.argv[1],sys.argv[2],sys.argv[3],*(int(c) for c in sys.argv[4:]))
