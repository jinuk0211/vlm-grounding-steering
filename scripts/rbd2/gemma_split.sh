#!/bin/bash
# User (11-03 20:4x): split GPUs 2, 3, 4, 5 between Gemma CHAIR test (gchair_*) and Gemma THRONE (gemma_chain.sh) and
# finish both. One chain per GPU (two memory-gated pools on one GPU collided at 20:38-20:42):
#   CHAIR : GPU 2 (c2, running) + GPU 4 x2 (c4a, c4b; GPU 4 was empty at 20:47)
#   THRONE: GPU 3 (g3) + GPU 5 x2 (g5a, g5b), as they run now
# When a CHAIR worker exits (every CHAIR test unit done), a THRONE worker (gemma_pool.sh) takes its GPU.
# Log: logs/gemma_split.log
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gemma_split.log
alive(){ [ -n "$(ps -o pid= -p $1)" ]; }
echo "=== $(date +%T) gemma_split.sh start" >> $LOG
bash scripts/rbd2/gchair_pool.sh 4 c4a 18879 & C4A=$!
echo "=== $(date +%T) CHAIR worker c4a on GPU 4 (pid $C4A)" >> $LOG
sleep 90
bash scripts/rbd2/gchair_pool.sh 4 c4b 18879 & C4B=$!
echo "=== $(date +%T) CHAIR worker c4b on GPU 4 (pid $C4B)" >> $LOG
hand(){ P=$1; G=$2; N=$3; D=$4
  ( while alive $P; do sleep 60; done; sleep $D; echo "=== $(date +%T) CHAIR worker pid $P done; THRONE worker $N joins GPU $G" >> $LOG
    exec bash scripts/rbd2/gemma_pool.sh $G $N 21257 ) & }
hand 125200 2 g2 0
hand $C4A 4 g4a 0
hand $C4B 4 g4b 60
wait
echo "=== $(date +%T) GEMMA_SPLIT_DONE" >> $LOG
