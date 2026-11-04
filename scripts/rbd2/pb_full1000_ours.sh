#!/bin/bash
# 11-03 16:4x user: no flan-t5-xl judge, and our method (s16a1bl2p1) on the first 1000 full-THRONE images, judged with
# flan-t5-base and -large only. Rows 0-499 are the throne_test500 run (same images); rows 500-999 are decoded here by two
# pool workers on GPU 5. pb_run500c.sh (pid $1) stays stopped until this script ends, because pb_amber_bl.sh starts its
# two GPU 5 AMBER workers when that pid exits; killing it last hands GPU 5 over.
# Then join -> judges base and large in parallel on GPU 5 (separate output files) -> pb_throne_partial.py (2 judges,
# threshold 6) for all methods on the 1000 images and on the first 500. Log: logs/full1000_ours.log, FULL1000_OURS_DONE.
# usage: pb_full1000_ours.sh STOPPED_RUN500C_PID
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/full1000_ours.log; Q=$O/queues/full1000_ours.txt; TD=$O/runs/throne_full1000
trap "kill -9 $1" EXIT
echo "=== $(date +%T) start" >> $LOG
bash scripts/rbd2/pb_pool.sh 5 o5a $Q & P1=$!; sleep 60
bash scripts/rbd2/pb_pool.sh 5 o5b $Q & P2=$!
wait $P1 $P2
echo "=== $(date +%T) decoded" >> $LOG
.venv/bin/python scripts/rbd2/pb_join_full1000_ours.py >> $LOG 2>&1 || { echo "=== $(date +%T) JOIN_FAILED" >> $LOG; exit 1; }
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=1000 RBD2_THRONE_SET=throne_full1000
U5=$(nvidia-smi --id=5 --query-gpu=uuid --format=csv,noheader)
RBD2_JUDGES=base $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U5 $TD s16a1bl2p1 > $O/logs/judge_full1000-ours-base.log 2>&1 &
sleep 30
RBD2_JUDGES=large $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U5 $TD s16a1bl2p1 > $O/logs/judge_full1000-ours-large.log 2>&1 &
wait
echo "=== $(date +%T) judged" >> $LOG
M="base revis rudder_clone grains_clone vti_author s16a1bl2p1"
{ echo "=== $(date +%T) throne_full1000, 2 judges (flan-t5 base+large, threshold 6; not the official 3-judge metric)"
  $L/venvs/throne/bin/python scripts/rbd2/pb_throne_partial.py $TD $M 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) first 500 of them (= throne_test500)"
  RBD2_THRONE_SET=throne_test500 RBD2_N=500 $L/venvs/throne/bin/python scripts/rbd2/pb_throne_partial.py $TD $M 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) FULL1000_OURS_DONE"; } >> $LOG
