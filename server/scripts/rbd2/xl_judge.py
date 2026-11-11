"""flan-t5-xl judgments for THRONE runs that have only base and large (user 11-08: fill the table's xl columns). score_throne.judge, unchanged,
with JUDGES = [xl], for one tag. Nothing is judged unless eval/TAG/responses.json equals what pb_throne_judge.py would write from the run file
(pb_throne_judge.py would delete combined/ with the base and large judgments otherwise) and combined/ holds base and large; combined/ is copied
to combined.bak-1108 before the first judgment.
usage: xl_judge.py GPU_UUID RUNDIR TAG   |   xl_judge.py --check RUNDIR TAG   (only the checks)"""
import json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_throne as S
S.JUDGES=[str(S.BENCH/'models/flan-t5-xl')]
gpu,rundir,tag=sys.argv[1],Path(sys.argv[2]),sys.argv[3]
refs=S.read(S.SETS/f'{S.TSET}.jsonl');rows={r['id']:r for r in S.read(rundir/f'{tag}-0-{S.N}.jsonl')};assert all(r['id'] in rows for r in refs),tag
d=rundir/'eval'/tag;resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
assert json.loads((d/'responses.json').read_text())==resp,f'{d}/responses.json differs from the run file: not judged'
names=sorted(p.name for p in (d/'combined').glob('*.json'));assert any('flan-t5-base' in n for n in names) and any('flan-t5-large' in n for n in names),names
if gpu=='--check':print('check ok',rundir.name,tag,names);sys.exit(0)
if not (d/'combined.bak-1108').exists():shutil.copytree(d/'combined',d/'combined.bak-1108')
S.judge(gpu,d);print('judged xl',rundir.name,tag,flush=True)
