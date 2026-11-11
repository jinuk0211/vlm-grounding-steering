#!/bin/bash
# Takes over from pb_run500.sh (12:25): the full-THRONE judges freed GPU 4 early, so two workers start there now; GPU 3
# (one worker) joins when pb_bl_chain.sh (pid $1) exits. Waits for every pool worker, including the GPU 5 ones started by
# pb_run500.sh (pids $2...), then pb_score500.sh.
# usage: pb_run500b.sh BL_CHAIN_PID GPU5_WORKER_PID [...]
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
Q=$O/queues/pool500.txt; C=$1; shift; P=("$@")
bash scripts/rbd2/pb_pool.sh 4 p4a $Q & P+=($!); sleep 5
bash scripts/rbd2/pb_pool.sh 4 p4b $Q & P+=($!)
alive(){ [ -n "$(ps -o pid= -p $(IFS=,; echo "$*"))" ]; }
while alive $C; do sleep 60; done
if alive "${P[@]}"; then echo "$(date +%T) GPU 3 joins" >> $O/logs/run500.log; bash scripts/rbd2/pb_pool.sh 3 p3 $Q & P+=($!); fi
while alive "${P[@]}"; do sleep 60; done
bash scripts/rbd2/pb_score500.sh
