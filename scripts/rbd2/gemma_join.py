"""Join one tag of the Gemma test run: runs/gemma_throne1000_parts/<tag>-<a>-<b>.jsonl, checked id for id against
sets/throne_full1000.jsonl, into runs/gemma_throne1000/<tag>-0-1000.jsonl. Exit 1 while rows are missing.
usage: gemma_join.py TAG"""
import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');T=sys.argv[1];N=1000
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
ref=[r['id'] for r in read(O/'sets/throne_full1000.jsonl')];assert len(ref)==N
piece={}
for f in glob.glob(str(O/f'runs/gemma_throne1000_parts/{T}-*-*.jsonl')):
    mm=re.fullmatch(re.escape(T)+r'-(\d+)-(\d+)\.jsonl',Path(f).name)
    if not mm:continue
    a,b=int(mm[1]),int(mm[2]);r=read(f)
    if len(r)==b-a and [x['id'] for x in r]==ref[a:b]:piece[a]=(b,r)
rows=[];at=0
while at<N and at in piece:b,r=piece[at];rows+=r;at=b
if at!=N:sys.exit(1)
assert len({json.dumps(r.get('setting'),sort_keys=True) for r in rows})==1
(O/'runs/gemma_throne1000').mkdir(exist_ok=True)
(O/f'runs/gemma_throne1000/{T}-0-{N}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
tok=[r['generated_tokens'] for r in rows]
print(f'{T:13s}{N} rows  tok {sum(tok)/N:.0f}  @512 {sum(t>=512 for t in tok)}')
