import sys, glob
from pathlib import Path
import numpy as np
import score_sets as S
SET='chair_test500'; S.SET=SET; S.STATE.setdefault(SET, S.STATE['chair_test60'])
refs=S.read(S.SETS/f'{SET}.jsonl'); ids=[r['id'] for r in refs]
P=Path(sys.argv[1]); runs={}
for f in sorted(glob.glob(str(P/'*-0-500.jsonl'))):
    t=Path(f).name[:-len('-0-500.jsonl')]; rows={r['id']:r for r in S.read(f)}; runs[t]={i:rows[i] for i in ids}
S.rng=np.random.default_rng(0); base=runs.pop('base')
res=S.chair(refs,{'base':base,**runs})
tok=lambda rows: np.mean([rows[i]['generated_tokens'] for i in ids])
for sub,sl in (('rows 0-499',slice(0,500)),('rows 0-99',slice(0,100))):
    pass
print('rows 0-499 (n=500); CI = run minus base')
print(f"{'base':12s} CHAIR_s {res['base']['CHAIR_s']:5.1f} CHAIR_i {res['base']['CHAIR_i']:5.2f} recall {res['base']['recall']:5.1f} tok {tok(base):4.0f}")
for t,v in res.items():
    if t=='base': continue
    c=v['ci']; print(f"{t:12s} CHAIR_s {v['CHAIR_s']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} tok {tok(runs[t]):4.0f}  dCHAIR_s[{c['CHAIR_s'][0]:+.1f},{c['CHAIR_s'][1]:+.1f}] drecall[{c['recall'][0]:+.1f},{c['recall'][1]:+.1f}]")
print('settings:'); [print(' ',t, {k:runs[t][ids[0]].get(k) for k in ('method','setting') if k in runs[t][ids[0]]}) for t in runs]
