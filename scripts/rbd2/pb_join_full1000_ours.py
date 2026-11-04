"""Join our method (s16a1bl2p1) on the first 1000 full-THRONE images: rows 0-499 from runs/throne_test500_all (throne_test500
is the first 500 rows of throne_full, same ids and images), rows 500-999 from the 50-image chunks in runs/throne_full1000_ours,
into runs/throne_full1000/s16a1bl2p1-0-1000.jsonl next to Base and the baselines. Checked id for id against
sets/throne_full1000.jsonl, one setting throughout (missing keys count as defaults, as in pb_join150.py)."""
import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');T='s16a1bl2p1';N=1000;H=500
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def setting(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};return json.dumps(s,sort_keys=True)
ref=[r['id'] for r in read(O/'sets/throne_full1000.jsonl')];assert len(ref)==N
assert [r['id'] for r in read(O/'sets/throne_test500.jsonl')]==ref[:H]
rows=read(O/f'runs/throne_test500_all/{T}-0-{H}.jsonl');assert [r['id'] for r in rows]==ref[:H]
piece={}
for f in glob.glob(str(O/f'runs/throne_full1000_ours/{T}-*-*.jsonl')):
    mm=re.fullmatch(re.escape(T)+r'-(\d+)-(\d+)\.jsonl',Path(f).name)
    if not mm:continue
    a,b=int(mm[1]),int(mm[2]);r=read(f)
    if len(r)==b-a and [x['id'] for x in r]==ref[a:b]:piece[a]=(b,r)
at=H
while at<N and at in piece:b,r=piece[at];rows+=r;at=b
if at!=N:print(f'INCOMPLETE {T}: rows 0-{at}');sys.exit(1)
if len({setting(r) for r in rows})!=1:print(f'{T}: different settings');sys.exit(1)
(O/f'runs/throne_full1000/{T}-0-{N}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print(f'{T:13s}{N} rows  addCriterion {sum("addCriterion" in x["prediction"] for x in rows)}')
