"""Join the Gemma CHAIR test units (runs/gchair_test500/TAG-B-E.jsonl, 50-image units) into runs/gchair_test500_all/TAG-0-500.jsonl
in chair_test500 order. usage: gchair_join.py TAG..."""
import json,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');S=O/'runs/gchair_test500';D=O/'runs/gchair_test500_all'
D.mkdir(exist_ok=True);ids=[json.loads(l)['id'] for l in open(O/'sets/chair_test500.jsonl')]
for t in sys.argv[1:]:
    by={}
    for b in range(0,500,50):
        f=S/f'{t}-{b}-{b+50}.jsonl'
        if f.exists():by.update({r['id']:r for r in map(json.loads,open(f))})
    ok=all(i in by for i in ids)
    if ok:(D/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
    print('join',t,sum(i in by for i in ids),'/ 500','complete' if ok else 'INCOMPLETE')
