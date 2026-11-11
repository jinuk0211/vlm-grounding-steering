"""Join the row pieces of every setting into runs/<set>_test150_all/<tag>-0-150.jsonl: rows 0-59 from the 60-image runs,
60-149 from the 120- and 150-image extensions, whichever complete pieces cover them. Every piece must match
sets/<set>_test150.jsonl id for id, and all rows of a tag must carry the same setting (older rows predate --bpos and --norm,
so a missing key counts as its default: bpos False, norm 'raw').
usage: pb_join150.py [--check]      (--check: only report coverage, write nothing)"""
import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');N=150;CHECK='--check' in sys.argv
# 21:05 user: only settings with a significant headline gain at 60 images (+ base, main, mainMed); the other 7 were dropped
TAGS='base main mainMed s16a1bl2p1 s16a1 s16a1bl1 s16a2bl1p05 bl1'.split()
DIRS={'chair':['chair_test60','chair_test60_sweep','chair_test120','chair_test150'],'amber':['amber_test60','amber_test150'],
      'throne':['throne_test60','throne_test120','throne_test150']}
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def setting(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};return json.dumps(s,sort_keys=True)
bad=[]
for s,dirs in DIRS.items():
    ref=[r['id'] for r in read(O/f'sets/{s}_test150.jsonl')];assert len(ref)==N
    out=O/f'runs/{s}_test150_all'
    for t in TAGS:
        piece={}
        for d in dirs:
            for f in sorted(glob.glob(str(O/'runs'/d/f'{t}-*-*.jsonl'))):
                m=re.fullmatch(re.escape(t)+r'-(\d+)-(\d+)\.jsonl',Path(f).name)
                if not m:continue
                a,b=int(m[1]),int(m[2]);rows=read(f)
                if len(rows)==b-a and [r['id'] for r in rows]==ref[a:b]:piece.setdefault(a,(b,rows,f'{d}/{Path(f).name}'))
        rows=[];at=0;src=[]
        while at<N and at in piece:b,r,f=piece[at];rows+=r;src.append(f);at=b
        if at!=N:bad.append(f'{s} {t}: rows 0-{at}');continue
        st={setting(r) for r in rows}
        if len(st)!=1:bad.append(f'{s} {t}: {len(st)} different settings');continue
        if not CHECK:
            out.mkdir(exist_ok=True);(out/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(f'{s:7s}{t:12s}',' + '.join(src))
if bad:print('INCOMPLETE',*bad,sep='\n');sys.exit(1)
print('ALL',len(DIRS)*len(TAGS),'complete')
