"""Collect one tag's rows for the first N images of a set from runs/qchair_test500/TAG-A-B.jsonl into
OUTDIR/TAG-0-N.jsonl in set order. Exit 1 if a row is missing.
usage: qchair_collect.py SET N OUTDIR TAG..."""
import glob,json,re,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');S=O/'runs/qchair_test500'
SET,N,OUT=sys.argv[1],int(sys.argv[2]),Path(sys.argv[3]);OUT.mkdir(parents=True,exist_ok=True)
ids=[json.loads(l)['id'] for l in open(O/f'sets/{SET}.jsonl')][:N];ok=True
for t in sys.argv[4:]:
    by={}
    for f in glob.glob(str(S/f'{t}-*-*.jsonl')):
        if re.fullmatch(re.escape(t)+r'-\d+-\d+\.jsonl',Path(f).name):by.update({r['id']:r for r in map(json.loads,open(f))})
    n=sum(i in by for i in ids)
    if n==N:(OUT/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
    else:ok=False
    print('collect',t,n,'/',N,'complete' if n==N else 'INCOMPLETE')
sys.exit(0 if ok else 1)
