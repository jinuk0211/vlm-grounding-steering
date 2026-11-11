"""Join VCD / M3ID chunks of one model into a score dir next to the same model's vdd Base and RBD rows of the
CHAIR 500 table, then score with score_sets.py (vs vdd Base, paired bootstrap).
usage: cdm_join.py MODEL   -> outputs/cdm-20261109/score_MODEL/{base,rbd,vcd,m3id}-0-500.jsonl ; exit 3 if incomplete"""
import json,shutil,sys
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/cdm-20261109';T=R/'outputs/rbd2-20261102/runs'
TABLE={'qwen':(T/'chair_test500_all/base-0-500.jsonl',T/'chair_test500_all/s16a2bl1p05-0-500.jsonl'),
       'gemma':(T/'gchair_test500_all/base-0-500.jsonl',T/'gchair_test500_all/s24a1bl1p05-0-500.jsonl'),
       'qwen3':(T/'qchair_test500_all/base-0-500.jsonl',T/'qchair_test500_all/s20a1bl1p05-0-500.jsonl')}
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
mod=sys.argv[1];refs=read(R/'outputs/rbd2-20261102/sets/chair_test500.jsonl');D=O/f'score_{mod}';D.mkdir(exist_ok=True)
# only the methods still queued for this model (user 11-10 18:40: VCD dropped for qwen3, its queue lines commented out)
Q=[l.split() for l in (O/'queue.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
for met in [m for m in ('vcd','m3id') if any(x[:2]==[mod,m] for x in Q)]:
    rows=[]
    for s in range(0,500,50):
        f=O/f'runs/{mod}/{met}-{s}-{s+50}.jsonl'
        if not f.exists():sys.exit(3)
        r=read(f)
        if len(r)<50:sys.exit(3)
        rows+=r[:50]
    assert [r['id'] for r in rows]==[r['id'] for r in refs]
    (D/f'{met}-0-500.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
b,rb=TABLE[mod];shutil.copyfile(b,D/'base-0-500.jsonl');shutil.copyfile(rb,D/'rbd-0-500.jsonl')
print('joined',mod)
