#!/bin/bash
# User (11-03 14:2x): once our method's runs are done, the baselines on full AMBER generative (1004 images).
# Set amber_full1004: data/amber/amber_generative.jsonl in AMBER id order, rows built as pb_make500.py builds amber_test500
# (prompt "Describe this image.", max_new_tokens 512); every amber_test500 row is identical to its row here.
# Methods as the full-THRONE baselines (baseline_bench.py: base, rudder_clone BF16 sdpa; grains_clone FP16 eager), VTI with
# the whitespace-normalised direction at the setting vti_chair.sh picks on CHAIR dev (vtiws_*), and revis last (under sdpa
# it reproduced Base on all 1000 full-THRONE images). 100-image units, methods interleaved per unit.
# The old Qwen AMBER Base (outputs/amber-20261030) is not used: 248/1004 rows, 37 of them with ' addCriterion'.
# Each GPU joins when its current jobs end: GPU 3 after pool500 worker p3; GPU 5 (x2) after pb_run500c.sh (whose THRONE
# judges use GPU 5); GPU 4 (x2) after pool500 p4a/p4b and pb_bl_chain.sh; GPU 2 after vti_chair.sh. A second worker on a
# GPU starts 150 s after the first, so its free-memory check sees the first model.
# Then join -> score_sets.py amber_full1004 (official AMBER generative matching, paired bootstrap vs base).
# Log: logs/amber_full.log      usage: pb_amber_bl.sh   (the PIDs below are the jobs running at 14:2x)
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/amber_full.log; Q=$O/queues/amber_full.txt; VC=$O/logs/vti_chair.log
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
echo "=== $(date +%T) pb_amber_bl.sh start" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY' || { echo "=== $(date +%T) SET_FAILED" >> $LOG; exit 1; }
import json
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');S=R/'outputs/rbd2-20261102/sets'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
t=read(S/'amber_test150.jsonl')[0];gen=sorted(read(R/'data/amber/amber_generative.jsonl'),key=lambda g:g['metadata']['amber_id'])
rows=[{**t,'id':g['id'],'group_id':g['group_id'],'image':g['image'],'image_sha256':g['image_sha256'],'metadata':g['metadata']} for g in gen]
by={r['id']:r for r in rows};a500=read(S/'amber_test500.jsonl')
assert len(rows)==1004==len(by) and all(by[r['id']]==r for r in a500)
(S/'amber_full1004.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
print('amber_full1004',len(rows),'ids',rows[0]['metadata']['amber_id'],'..',rows[-1]['metadata']['amber_id'],'| amber_test500 rows identical')
PY
while alive 1582071 && ! grep -q 'picked\|PICK_FAILED' $VC; do sleep 60; done
V=$(grep -o 'picked vtiws_[^ ]*' $VC | cut -d' ' -f2); [ -n "$V" ] || V=vtiws_v0.9_t0.9
echo "=== $(date +%T) VTI setting $V" >> $LOG
U(){ b=$1; e=$((b+100)); [ $e -gt 1004 ] && e=1004; echo "amber_full1004 amber_full1004 $b $e"; }
{ echo "# 14:2x user: baselines on full AMBER generative after our method's runs; revis last."
  for b in $(seq 0 100 1000); do
    echo "base sdpa $(U $b)"; echo "rudder_clone sdpa $(U $b)"; echo "grains_clone eager $(U $b)"; echo "$V eager $(U $b)"; done
  for b in $(seq 0 100 1000); do echo "revis sdpa $(U $b)"; done; } > $Q
start(){ G=$1; N=$2; D=$3; shift 3
  ( while alive $@; do sleep 60; done; sleep $D; echo "=== $(date +%T) GPU $G worker $N joins" >> $LOG
    exec bash scripts/rbd2/pb_amber_pool.sh $G $N $Q ) & }
start 3 a3 0 1476645
start 5 a5a 0 1476643
start 5 a5b 150 1476643
start 4 a4a 0 692460 692827 1829740
start 4 a4b 150 692460 692827 1829740
start 2 a2 0 1582071
wait
echo "=== $(date +%T) decoded" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import json,glob
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');S=O/'runs/amber_full1004';D=O/'runs/amber_full1004_all'
D.mkdir(exist_ok=True);ids=[json.loads(l)['id'] for l in open(O/'sets/amber_full1004.jsonl')]
for tag in sorted({Path(f).name.rsplit('-',2)[0] for f in glob.glob(str(S/'*.jsonl'))}):
    rows=[]
    for b in range(0,1004,100):
        f=S/f'{tag}-{b}-{min(b+100,1004)}.jsonl';rows+=[json.loads(l) for l in open(f)] if f.exists() else []
    ok=[r['id'] for r in rows]==ids
    if ok:(D/f'{tag}-0-1004.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print('join',tag,len(rows),'complete' if ok else 'INCOMPLETE')
PY
export RBD2_SETS=$O/sets AMBER_REPO=$R/data/amber/repo NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 RBD2_N=1004
{ echo "=== $(date +%T) amber_full1004 (paired bootstrap vs base)"
  $L/venvs/amber/bin/python scripts/rbd2/score_sets.py amber_full1004 $O/runs/amber_full1004_all 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) AMBER_FULL_DONE"; } >> $LOG
