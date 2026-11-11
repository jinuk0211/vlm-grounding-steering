#!/bin/bash
# Finish full-THRONE 1000 scoring after the GrAInS flan-t5-xl judge OOM'd on GPU 3 (09:07, next to the VTI xl judge):
# pb_bl_full_score.sh (pid $1) is stopped (SIGSTOP) so it cannot start a second GrAInS judge; GrAInS is judged now on
# GPU 4, then, once the VTI judge (pid $2) has exited too, score_throne.py runs here and the stopped script is killed,
# which lets pb_bl_chain.sh end (and pb_run500b.sh add its GPU 3 worker).
# usage: pb_full1000_finish.sh STOPPED_SCORE_PID VTI_JUDGE_PID
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=1000 RBD2_THRONE_SET=throne_full1000
TD=$O/runs/throne_full1000; U4=$(nvidia-smi --id=4 --query-gpu=uuid --format=csv,noheader)
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U4 $TD grains_clone > $O/logs/judge_full1000-grains-retry.log 2>&1
while [ -n "$(ps -o pid= -p $2)" ]; do sleep 60; done
{ echo "=== $(date +%T) throne_full1000 (GrAInS xl judge re-run on GPU 4 after OOM)"
  $L/venvs/throne/bin/python scripts/rbd2/score_throne.py $U4 $TD 2>&1 | grep -v 'it/s' | tail -8
  echo "=== $(date +%T) THRONE_FULL1000_DONE"; } >> $O/logs/score_throne_full.log
kill -9 $1
