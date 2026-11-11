#!/bin/bash
# Takes over from pb_run500b.sh (13:55): GPU 3 became free when the VTI xl judge ended, while pb_bl_chain.sh still waits
# on the GrAInS re-judge, so the GPU 3 worker starts now instead of at chain exit. Waits for every pool worker (pids $@,
# started by the earlier scripts, plus this one), then pb_score500.sh.
# usage: pb_run500c.sh WORKER_PID [...]
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
P=("$@")
bash scripts/rbd2/pb_pool.sh 3 p3 $O/queues/pool500.txt & P+=($!)
echo "$(date +%T) GPU 3 joins (pb_run500c.sh)" >> $O/logs/run500.log
while [ -n "$(ps -o pid= -p $(IFS=,; echo "${P[*]}"))" ]; do sleep 60; done
bash scripts/rbd2/pb_score500.sh
