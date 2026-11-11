"""Run THRONE's judges (score_throne.judge, unchanged) for some tags only, so two GPUs can split the work.
Writes eval/<tag>/responses.json exactly as score_throne.main does; score_throne.py then only scores.
RBD2_JUDGES=base,large (optional) runs only those flan-t5 judges (11-03 user: no flan-t5-xl).
usage: pb_throne_judge.py GPU_UUID RUNDIR TAG [TAG ...]"""
import json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_throne as S
if os.environ.get('RBD2_JUDGES'):S.JUDGES=[str(S.BENCH/f'models/flan-t5-{s}') for s in os.environ['RBD2_JUDGES'].split(',')]
gpu,rundir,tags=sys.argv[1],Path(sys.argv[2]),sys.argv[3:]
refs=S.read(S.SETS/f'{S.TSET}.jsonl');ids=[r['id'] for r in refs]
for tag in tags:
    rows={r['id']:r for r in S.read(rundir/f'{tag}-0-{S.N}.jsonl')};assert all(i in rows for i in ids),tag
    d=rundir/'eval'/tag;d.mkdir(parents=True,exist_ok=True)
    resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
    if not (d/'responses.json').exists() or json.loads((d/'responses.json').read_text())!=resp:
        shutil.rmtree(d/'combined',ignore_errors=True);(d/'responses.json').write_text(json.dumps(resp))
    S.judge(gpu,d);print('judged',tag,flush=True)
