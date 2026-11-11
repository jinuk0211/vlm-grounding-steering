"""rbd5: join: rbd5_score.py join OUTDIR -> runs/OUTDIR_all/{gate80,cfgate80}-0-500.jsonl; exit 1 while rows are missing.
score: rbd5_score.py score DIR -> every candidate vs the current RBD row (paired bootstrap, 2000, seed 0) and CHAIR_s without class orange."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102'
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];ids=[r['id'] for r in refs];IM={r['id']:r['metadata']['image_id'] for r in refs}
def join(od):
    ok=True;S=O/'runs'/od;A=O/'runs'/f'{od}_all';A.mkdir(exist_ok=True)
    for t in ('gate80','cfgate80'):
        by={}
        for b in range(0,500,50):
            f=S/f'{t}-{b}-{b+50}.jsonl'
            if f.exists():by.update({r['id']:r for r in map(json.loads,open(f))})
        n=sum(i in by for i in ids);print('join',od,t,n,'/ 500')
        if n==500:(A/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
        else:ok=False
    sys.exit(0 if ok else 1)
def score(d):
    from vlm_grounding.evaluation import _load_external_module
    c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
    assert c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
    mod=_load_external_module(src,'chair_rbd5s');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
    GT={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
    def per(t):
        by={r['id']:r for r in map(json.loads,open(Path(d)/f'{t}-0-500.jsonl'))};H=[];X=[];N=[];G=[];T=[]
        for i in ids:
            f=set(sc.caption_to_words(by[i]['prediction'])[1]);g=GT[IM[i]];H.append(bool(f-g));X.append(bool(f-g-{'orange'}));N.append(len(f&g));G.append(len(g));T.append(by[i]['generated_tokens'])
        return dict(hal=np.array(H,float),halx=np.array(X,float),hit=np.array(N,float),gt=np.array(G,float),tok=np.array(T))
    P={t:per(t) for t in ('base','rbd','gate80','cfgate80')}
    def ci(a,b):
        g=np.random.default_rng(0);ds=[];dx=[];dr=[]
        for _ in range(2000):
            k=g.integers(0,500,500);ds.append(100*(a['hal'][k].mean()-b['hal'][k].mean()));dx.append(100*(a['halx'][k].mean()-b['halx'][k].mean()));dr.append(100*(a['hit'][k].sum()/a['gt'][k].sum()-b['hit'][k].sum()/b['gt'][k].sum()))
        return [np.quantile(v,[.025,.975]) for v in (ds,dx,dr)]
    for t,a in P.items():
        s=f"{t:9s} CHAIR_s {100*a['hal'].mean():5.1f} no-orange {100*a['halx'].mean():5.1f} recall {100*a['hit'].sum()/a['gt'].sum():5.1f} tok {a['tok'].mean():4.0f} @512 {(a['tok']>=512).sum():3d}"
        for b in ('base','rbd'):
            if t!=b and not (t=='base' and b=='rbd'):
                x=ci(a,P[b]);s+=f" | vs {b}: dCHAIR_s[{x[0][0]:+.1f},{x[0][1]:+.1f}] dNoOrange[{x[1][0]:+.1f},{x[1][1]:+.1f}] drecall[{x[2][0]:+.1f},{x[2][1]:+.1f}]"
        print(s,flush=True)
if __name__=='__main__':(join if sys.argv[1]=='join' else score)(sys.argv[2])
