import sys, json, glob
from pathlib import Path
import numpy as np
import score_sets as S
SET='chair_test100'; S.SET=SET; S.STATE.setdefault(SET, S.STATE['chair_test60'])
refs=S.read(S.SETS/f'{SET}.jsonl'); ids=[r['id'] for r in refs]
P=Path(sys.argv[1]); runs={}
for f in sorted(glob.glob(str(P/'*-0-500.jsonl'))):
    t='paper_'+Path(f).name[:-len('-0-500.jsonl')]; rows={r['id']:r for r in S.read(f)}
    if all(i in rows for i in ids): runs[t]={i:rows[i] for i in ids}
for t,f in [('box_nosteer','runs/steer100/nosteer-0-100.jsonl'),('box_full','runs/steer100/full-0-100.jsonl')]:
    runs[t]={r['id']:r for r in S.read(f)}
S.rng=np.random.default_rng(0)
base=runs.pop('paper_base')
res=S.chair(refs,{'base':base,**runs})
nw=lambda rows: np.mean([len(rows[i]['prediction'].split()) for i in ids])
print('rows 0-99 (n=100); CI = run minus paper base')
print(f"{'paper_base':14s} CHAIR_s {res['base']['CHAIR_s']:5.1f} CHAIR_i {res['base']['CHAIR_i']:5.2f} recall {res['base']['recall']:5.1f} words {nw(base):4.0f}")
for t,v in res.items():
    if t=='base': continue
    c=v['ci']; print(f"{t:14s} CHAIR_s {v['CHAIR_s']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} words {nw(runs[t]):4.0f}  dCHAIR_s[{c['CHAIR_s'][0]:+.1f},{c['CHAIR_s'][1]:+.1f}] drecall[{c['recall'][0]:+.1f},{c['recall'][1]:+.1f}]")
for a,b in [('paper_abl_nosteer','box_nosteer'),('paper_rbd_full','box_full')]:
    if a in runs: print(a,'vs',b,'identical captions:',sum(runs[a][i]['prediction']==runs[b][i]['prediction'] for i in ids),'/100')
