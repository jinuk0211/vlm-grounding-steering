"""THRONE judgment of one 1,000-image block of a t4k run (rows 1000-4999 of throne_full): score_throne.judge, unchanged (THRONE's
throne_aqa_evaluation.py, batch 32, 3 question formats), with one judge. The responses are the block's rows of runs/t4k_parts_MODEL/TAG-*.jsonl in
set order; output runs/t4k_MODEL/eval/TAG/bBLOCK-JUDGE/{responses.json,combined/evaluations_*flan-t5-JUDGE.json}.
usage: t4k_judge.py GPU_UUID MODEL TAG BLOCK JUDGE     (THRONE venv; env BENCH, RBD2_SETS)"""
import json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import score_throne as S
gpu,M,tag,b,J=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
assert J in ('base','large','xl'),J
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
refs=S.read(S.SETS/'throne_r4000.jsonl')[1000*b:1000*(b+1)];assert len(refs)==1000
rows={}
for f in sorted((O/'runs'/f't4k_parts_{M}').glob(f'{tag}-*-*.jsonl')):
    if f.name[:-6].rsplit('-',2)[0]!=tag:continue
    for r in S.read(f):rows[r['id']]=r
miss=[r['id'] for r in refs if r['id'] not in rows];assert not miss,f'{len(miss)} rows of block {b} not decoded'
assert all(rows[r['id']]['image_sha256']==r['image_sha256'] for r in refs)
S.JUDGES=[str(S.BENCH/f'models/flan-t5-{J}')]
d=O/'runs'/f't4k_{M}'/'eval'/tag/f'b{b}-{J}';d.mkdir(parents=True,exist_ok=True)
resp={'prompts':['Describe this image in detail.'],'responses':[[0,r['metadata']['image_id'],rows[r['id']]['prediction']] for r in refs]}
if not (d/'responses.json').exists() or json.loads((d/'responses.json').read_text())!=resp:
    shutil.rmtree(d/'combined',ignore_errors=True);(d/'responses.json').write_text(json.dumps(resp))
S.judge(gpu,d);print('judged',M,tag,b,J,flush=True)
