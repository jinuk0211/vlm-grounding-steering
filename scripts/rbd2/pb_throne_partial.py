"""Provisional THRONE scores while the flan-t5-xl judge is still running: the finished judge files (flan-t5-base, -large)
are copied to diag/<rundir name>_2judges/<tag>/combined and scored with score_throne.per_image at a voting threshold of 6,
i.e. the same unanimity rule as the headline threshold 9 (3 judges x 3 formats), for 2 judges x 3 formats.
Not the official metric; paired bootstrap vs base as in score_throne.py.
usage: pb_throne_partial.py RUNDIR TAG [TAG ...]      (env as score_throne.py: BENCH, RBD2_SETS, RBD2_THRONE_SET, RBD2_N)"""
import glob,json,shutil,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_throne as S
rundir=Path(sys.argv[1]);tags=sys.argv[2:];diag=rundir.parent.parent/'diag'/f'{rundir.name}_2judges'
keys=[r['metadata']['image_id'] for r in S.read(S.SETS/f'{S.TSET}.jsonl')];per={};res={}
for t in tags:
    src=sorted(f for f in glob.glob(str(rundir/'eval'/t/'combined'/'*.json')) if 'flan-t5-xl' not in f)
    assert len(src)==2,(t,src)
    d=diag/t/'combined';shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True)
    for f in src:shutil.copy(f,d)
    P,R=S.per_image(d.parent,thr=6);per[t]=(np.array([P[k] for k in keys]),np.array([R[k] for k in keys]));res[t]=S.summ(*per[t])
rng=np.random.default_rng(0);B=[rng.integers(0,len(keys),len(keys)) for _ in range(2000)];bp,br=per['base']
for t,(p,r) in per.items():
    ci=''
    if t!='base':
        d=np.array([[S.summ(p[k],r[k])[q]-S.summ(bp[k],br[k])[q] for q in ('P','R','F0.5')] for k in B])
        ci=' '.join(f"d{q}[{np.quantile(d[:,j],.025):+.1f},{np.quantile(d[:,j],.975):+.1f}]" for j,q in enumerate(('P','R','F0.5')))
    v=res[t];print(f"{t:13s} P {v['P']:5.1f} R {v['R']:5.1f} F1 {v['F1']:5.1f} F0.5 {v['F0.5']:5.1f}  {ci}")
