#!/bin/bash
# After the 150-image rbd2 runs are scored (pb_finish150.sh exits): Base and the four existing baselines on the first 1000
# images of full THRONE (COCO val2017 in id order, 100-image chunks, all methods per chunk; user: 1000, not all 5000) on
# GPUs 3 and 4 (pb_bl_pool.sh: GPU 3 x1, GPU 4 x2), then score them (pb_bl_full_score.sh 1000).
# usage: pb_bl_chain.sh FINISHER_PID
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
while [ -n "$(ps -o pid= -p $1)" ]; do sleep 60; done
echo "$(date +%T) full THRONE baselines start" >> $O/logs/bl_chain.log
Q=$O/queues/bl_throne_full.txt
bash scripts/rbd2/pb_bl_pool.sh 3 b3 $Q & sleep 5
bash scripts/rbd2/pb_bl_pool.sh 4 b4a $Q & sleep 5
bash scripts/rbd2/pb_bl_pool.sh 4 b4b $Q &
wait
echo "$(date +%T) full THRONE baselines decoded" >> $O/logs/bl_chain.log
bash scripts/rbd2/pb_bl_full_score.sh 1000 4 4 4 3 3
