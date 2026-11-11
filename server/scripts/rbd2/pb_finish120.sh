#!/bin/bash
# 120-image check: wait for 60-119 rows and the 0-59 rows of the same configs, join them into runs/<set>_all/<tag>-0-120.jsonl,
# score CHAIR on the CPU as soon as it is complete, judge THRONE on GPU 4 in two parallel halves, then score THRONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export NLTK_DATA=$L/.cache/nltk_data RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json BENCH=$L/bench \
       HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=120
CT="base main s16a1bl1"; TT="base s16a1bl2p1 s16a1"   # base + top 2 per set (user, 17:05)
src0(){ case $1-$2 in chair-base|chair-main) echo $O/runs/chair_test60;; chair-*) echo $O/runs/chair_test60_sweep;; *) echo $O/runs/throne_test60;; esac; }
ok(){ [ -f $1 ] && [ $(wc -l < $1) -ge 60 ]; }
join(){ TAGS=$2; for t in $TAGS; do until ok $O/runs/${1}_test120/$t-60-120.jsonl && ok $(src0 $1 $t)/$t-0-60.jsonl; do sleep 60; done; done
  mkdir -p $O/runs/${1}_test120_all
  for t in $TAGS; do cat $(src0 $1 $t)/$t-0-60.jsonl $O/runs/${1}_test120/$t-60-120.jsonl > $O/runs/${1}_test120_all/$t-0-120.jsonl; done; }
join chair "$CT"
{ echo "=== $(date +%T) chair_test120"; PYTHONPATH=scripts:src .venv/bin/python scripts/rbd2/score_sets.py chair_test120 $O/runs/chair_test120_all 2>&1 | grep -v 'it/s'; } >> $O/logs/score120.log
join throne "$TT"
export RBD2_THRONE_SET=throne_test120; U=$(nvidia-smi --id=4 --query-gpu=uuid --format=csv,noheader)
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $O/runs/throne_test120_all base s16a1 > $O/logs/judge120a.log 2>&1 &
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $O/runs/throne_test120_all s16a1bl2p1 > $O/logs/judge120b.log 2>&1 &
wait
{ echo "=== $(date +%T) throne_test120"; $L/venvs/throne/bin/python scripts/rbd2/score_throne.py $U $O/runs/throne_test120_all 2>&1 | grep -v 'it/s' | tail -10
  echo "=== $(date +%T) SCORE120_DONE"; } >> $O/logs/score120.log
