#!/bin/bash
# Score the 500-image runs: pb_join500.py, CHAIR and AMBER on the CPU, THRONE judges for the two THRONE runs on GPU 5,
# THRONE scores. Everything goes to logs/score500.log.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json AMBER_REPO=$R/data/amber/repo \
       BENCH=$L/bench NLTK_DATA=$L/.cache/nltk_data HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=500
LOG=$O/logs/score500.log
echo "=== $(date +%T) join" >> $LOG
.venv/bin/python scripts/rbd2/pb_join500.py >> $LOG 2>&1 || { echo "=== $(date +%T) JOIN_FAILED" >> $LOG; exit 1; }
{ echo "=== $(date +%T) chair_test500"; PYTHONPATH=scripts:src .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $O/runs/chair_test500_all 2>&1 | grep -v 'it/s'; } >> $LOG
{ echo "=== $(date +%T) amber_test500"; $L/venvs/amber/bin/python scripts/rbd2/score_sets.py amber_test500 $O/runs/amber_test500_all 2>&1 | grep -v 'it/s'; } >> $LOG
export RBD2_THRONE_SET=throne_test500; TD=$O/runs/throne_test500_all; U5=$(nvidia-smi --id=5 --query-gpu=uuid --format=csv,noheader)
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U5 $TD base > $O/logs/judge500-base.log 2>&1 &
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U5 $TD s16a1bl2p1 > $O/logs/judge500-s16a1bl2p1.log 2>&1 &
wait
{ echo "=== $(date +%T) throne_test500"; $L/venvs/throne/bin/python scripts/rbd2/score_throne.py $U5 $TD 2>&1 | grep -v 'it/s' | tail -4
  echo "=== $(date +%T) SCORE500_DONE"; } >> $LOG
