#!/bin/bash
# User 11-10 18:16: t4k baselines first (our method dropped from the queues for now), GPUs 0, 1, 2 only.
# Waits until the CDM worker on this GPU has left (POOL_DONE in outputs/cdm-20261109/logs/pool-CDM_WORKER.log), then gives the GPU to t4k:
# a decode slot (t4k_pool.sh) and, when no decode unit is left, a judge slot (t4k_jpool.sh all), like t4k_chain.sh's dslot.
# A5000 rows: the decoders record the GPU name in each row ("gpu").      usage: t4k_after_cdm.sh GPU_INDEX T4K_WORKER CDM_WORKER
G=$1; W=$2; CW=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; C=$R/outputs/cdm-20261109; cd $R
until grep -q POOL_DONE $C/logs/pool-$CW.log 2>/dev/null; do [ -f $O/t4k/STOP ] && exit 0; sleep 60; done
sleep 30
echo "=== $(date '+%m-%d %T') GPU $G joins t4k as $W (CDM worker $CW done)" >> $O/logs/t4k.log
bash scripts/rbd2/t4k_pool.sh $G $W $O/queues/t4k_decode.txt
bash scripts/rbd2/t4k_jpool.sh $G $W all
echo "=== $(date '+%m-%d %T') $W (GPU $G) left t4k" >> $O/logs/t4k.log
