"""Join the full-THRONE chunks of Base and the four baselines for the first N images (N a multiple of the chunk size):
sets/throne_fullN.jsonl = the first N rows of sets/throne_full.jsonl, runs/throne_fullN/<method>-0-N.jsonl from
runs/throne_full_bl/<method>-<a>-<b>.jsonl, every chunk checked id for id.
usage: pb_bl_full_join.py N"""
import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');N=int(sys.argv[1])
M=('base','revis','rudder_clone','grains_clone','vti_author')
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
refs=read(O/'sets/throne_full.jsonl')[:N];ref=[r['id'] for r in refs];assert len(ref)==N
(O/f'sets/throne_full{N}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
out=O/f'runs/throne_full{N}';out.mkdir(exist_ok=True);bad=[]
for m in M:
    piece={}
    for f in glob.glob(str(O/f'runs/throne_full_bl/{m}-*-*.jsonl')):
        mm=re.fullmatch(re.escape(m)+r'-(\d+)-(\d+)\.jsonl',Path(f).name)
        if not mm:continue
        a,b=int(mm[1]),int(mm[2])
        if b>N:continue
        r=read(f)
        if len(r)==b-a and [x['id'] for x in r]==ref[a:b]:piece[a]=(b,r)
    rows=[];at=0
    while at<N and at in piece:b,r=piece[at];rows+=r;at=b
    if at!=N:bad.append(f'{m}: rows 0-{at}');continue
    (out/f'{m}-0-{N}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(f'{m:13s}{N} rows  addCriterion {sum("addCriterion" in x["prediction"] for x in rows)}  tok {sum(x["generated_tokens"] for x in rows)/N:.0f}')
if bad:print('INCOMPLETE',*bad,sep='\n');sys.exit(1)
