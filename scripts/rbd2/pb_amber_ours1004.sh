#!/bin/bash
# User (11-03 18:1x, GPUs 2 and 3 idle): our method on the rest of AMBER generative so it covers the same 1004 images as the
# baselines (pb_amber_bl.sh). amber_test500 = AMBER ids 1-100 and 201-600; here ids 101-200 and 601-1004, i.e. rows
# 100-200 and 600-1004 of sets/amber_full1004.jsonl (rows identical to amber_test500's). Base and mainMed exactly as in
# queues/pool500.txt (vdd_decode; mainMed = --alpha 1 --layer 16 --lam 1 --pen 0.5 --norm median). The vdd Base differs from
# the baseline_bench Base (114/500 captions identical), so ours is compared with its own vdd Base, as on the 500 images.
# Workers: pb_pool.sh on GPU 2 and GPU 3 (one Qwen each). Then join with runs/amber_test500_all -> runs/amber_full1004_ours_all,
# score_sets.py amber_full1004 (paired bootstrap vs vdd base).   Log: logs/amber_ours1004.log, AMBER_OURS_DONE
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/amber_ours1004.log; Q=$O/queues/amber_ours1004.txt
{ echo "# 18:1x user: GPUs 2,3 idle -> base + mainMed on AMBER ids 101-200, 601-1004"
  for b in $(seq 100 50 150) $(seq 600 50 1000); do e=$((b+50)); [ $e -gt 1004 ] && e=1004
    echo "amber_full1004 amber_full1004_ours mainMed $b $e --alpha 1 --layer 16 --lam 1 --pen 0.5 --norm median"
    echo "amber_full1004 amber_full1004_ours base $b $e"; done; } > $Q
echo "=== $(date +%T) start, $(grep -vc '^#' $Q) units" >> $LOG
bash scripts/rbd2/pb_pool.sh 2 m2 $Q & sleep 5
bash scripts/rbd2/pb_pool.sh 3 m3 $Q &
wait
echo "=== $(date +%T) decoded" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import json
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');S=O/'runs/amber_full1004_ours';D=O/'runs/amber_full1004_ours_all'
D.mkdir(exist_ok=True);ids=[json.loads(l)['id'] for l in open(O/'sets/amber_full1004.jsonl')]
rd=lambda p:[json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
for t in ['base','mainMed']:
    by={r['id']:r for r in rd(O/f'runs/amber_test500_all/{t}-0-500.jsonl')}
    for b in [100,150]+list(range(600,1004,50)):by.update({r['id']:r for r in rd(S/f'{t}-{b}-{min(b+50,1004)}.jsonl')})
    ok=all(i in by for i in ids)
    if ok:(D/f'{t}-0-1004.jsonl').write_text(''.join(json.dumps(by[i])+'\n' for i in ids))
    print('join',t,sum(i in by for i in ids),'/ 1004','complete' if ok else 'INCOMPLETE')
PY
export RBD2_SETS=$O/sets AMBER_REPO=$R/data/amber/repo NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 RBD2_N=1004
{ echo "=== $(date +%T) amber_full1004 ours (paired bootstrap vs vdd base)"
  $L/venvs/amber/bin/python scripts/rbd2/score_sets.py amber_full1004 $O/runs/amber_full1004_ours_all 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) AMBER_OURS_DONE"; } >> $LOG
