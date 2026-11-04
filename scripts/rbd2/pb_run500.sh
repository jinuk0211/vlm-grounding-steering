#!/bin/bash
# 500-image check (user, 12:00): Base + one setting per set, images 150-499 in 50-image chunks through pb_pool.sh.
# GPU 5 (two workers) starts now; GPUs 3 and 4 (one and two workers) join when the full-THRONE baseline scoring
# (pb_bl_chain.sh, pid $1) has finished with them. Then pb_join500.py, CHAIR and AMBER on the CPU, THRONE judges for the
# two THRONE runs on GPU 5, THRONE scores. Everything goes to logs/score500.log.
# usage: pb_run500.sh BL_CHAIN_PID
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
Q=$O/queues/pool500.txt; P=()
bash scripts/rbd2/pb_pool.sh 5 p5a $Q & P+=($!); sleep 5
bash scripts/rbd2/pb_pool.sh 5 p5b $Q & P+=($!)
while [ -n "$(ps -o pid= -p $1)" ]; do sleep 60; done
if [ -n "$(ps -o pid= -p $(IFS=,; echo "${P[*]}"))" ]; then
  echo "$(date +%T) GPUs 3 and 4 join" >> $O/logs/run500.log
  bash scripts/rbd2/pb_pool.sh 3 p3 $Q & sleep 5
  bash scripts/rbd2/pb_pool.sh 4 p4a $Q & sleep 5
  bash scripts/rbd2/pb_pool.sh 4 p4b $Q &
fi
wait
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
