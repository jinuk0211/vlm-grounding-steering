"""Join the 500-image runs: rows 0-149 from runs/<set>_test150_all/<tag>-0-150.jsonl, rows 150-499 from the 50-image
chunks in runs/<set>_test500, into runs/<set>_test500_all/<tag>-0-500.jsonl. Every piece must match
sets/<set>_test500.jsonl id for id and carry the tag's setting (missing keys count as defaults, as in pb_join150.py).
usage: pb_join500.py [--check]"""
import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');N=150;M=500;CHECK='--check' in sys.argv
# 12:00 user: each set's Base plus its best 150-image setting
RUNS={'chair':['base','s16a2bl1p05'],'throne':['base','s16a1bl2p1'],'amber':['base','mainMed']}
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def setting(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};return json.dumps(s,sort_keys=True)
bad=[]
for s,tags in RUNS.items():
    ref=[r['id'] for r in read(O/f'sets/{s}_test500.jsonl')];assert len(ref)==M
    for t in tags:
        rows=read(O/f'runs/{s}_test150_all/{t}-0-{N}.jsonl');assert [r['id'] for r in rows]==ref[:N],(s,t)
        piece={}
        for f in glob.glob(str(O/f'runs/{s}_test500/{t}-*-*.jsonl')):
            mm=re.fullmatch(re.escape(t)+r'-(\d+)-(\d+)\.jsonl',Path(f).name)
            if not mm:continue
            a,b=int(mm[1]),int(mm[2]);r=read(f)
            if len(r)==b-a and [x['id'] for x in r]==ref[a:b]:piece[a]=(b,r)
        at=N
        while at<M and at in piece:b,r=piece[at];rows+=r;at=b
        if at!=M:bad.append(f'{s} {t}: rows 0-{at}');continue
        if len({setting(r) for r in rows})!=1:bad.append(f'{s} {t}: different settings');continue
        if not CHECK:
            (O/f'runs/{s}_test500_all').mkdir(exist_ok=True)
            (O/f'runs/{s}_test500_all/{t}-0-{M}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(f'{s:7s}{t:12s}{len(rows)} rows')
if bad:print('INCOMPLETE',*bad,sep='\n');sys.exit(1)
