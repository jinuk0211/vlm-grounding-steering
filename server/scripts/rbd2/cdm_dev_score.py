"""M3ID on CHAIR dev 100 (round 2): join dev/MODEL/m3id-{0-50,50-100} with the CHAIR dev 100 rows of the table runs
(runs/chair_dev100_MODEL: vdd Base 'vddbase' = base here, bench Base 'base' = bench, RBD at the table setting = rbd) and score with
score_sets.py on the dev scorer state (as cdev_prep.py run_scorer; paired bootstrap vs the vdd Base, same decode path as M3ID).
usage: cdm_dev_score.py MODEL   -> outputs/cdm-20261109/score_dev_MODEL/ ; exit 3 if M3ID rows are incomplete"""
import json,os,sys
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/cdm-20261109';T=R/'outputs/rbd2-20261102/runs';N=100
RBD={'qwen':'s16a2bl1p05','gemma':'s24a1bl1p05','qwen3':'s20a1bl1p05'}
def read(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
mod=sys.argv[1];ids=[r['id'] for r in read(R/'outputs/rbd2-20261102/sets/chair_dev100.jsonl')];D=O/f'score_dev_{mod}';D.mkdir(exist_ok=True)
def rows(files):
    by={x['id']:x for f in files if Path(f).exists() for x in read(f)}
    return [by[i] for i in ids] if all(i in by for i in ids) else None
m=rows([O/f'dev/{mod}/m3id-{s}-{s+50}.jsonl' for s in (0,50)])
if m is None:sys.exit(3)
src={'m3id':m,'base':rows([T/f'chair_dev100_{mod}/vddbase-{s}-{e}.jsonl' for s,e in ((0,60),(60,100))]),
     'bench':rows([T/f'chair_dev100_{mod}/base-{s}-{e}.jsonl' for s,e in ((0,60),(60,100))]),
     'rbd':rows([T/f'chair_dev100_{mod}/{RBD[mod]}-{s}-{e}.jsonl' for s,e in ((0,60),(60,100))])}
for t,r in src.items():
    assert r is not None,(mod,t,'incomplete');(D/f'{t}-0-{N}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in r),encoding='utf-8')
os.environ.setdefault('RBD2_SETS',str(R/'outputs/rbd2-20261102/sets'));sys.path.insert(0,str(R/'scripts/rbd2'))
import score_sets as S
S.STATE['chair_dev100']=S.STATE['chair_dev60'];S.SET='chair_dev100';S.N=N
print(f'=== {mod} CHAIR dev 100 vs vdd Base (bench = bench Base, rbd = table setting)');S.main(D)
