#!/bin/bash
# User, 11-05 19:0x (through session h-20): HAT + ALOHa on GPU 3 only; GPU 2 is the logit lens session's, and rbd6 worker r3 leaves GPU 3
# after its unit. One worker; nothing goes on the GPU until it has the memory free, the CPU steps start at once. Order: HAT reproduction
# (a parser failure stops here), pilot on the existing CHAIR captions, Qwen, Gemma; a model with no pick loses only RBD, its baselines run.
# More GPUs later: bash scripts/rbd2/hat_pool.sh GPU NAME outputs/rbd2-20261102/queues/hat385_{dev,test}.txt joins the running queue.
# Log: logs/hat_gpu3.log, logs/hat385.log, logs/pilot_chair500.log. When this ends, tell session "baseline 실험" that GPU 3 is back.
cd ${HAT_L:-/home/project-b/LJW}/vlm-grounding-steering
export WORKERS=3 PARSE_GPUS=3
HAT_REPRO_ONLY=1 bash scripts/rbd2/hat_chain.sh || { echo "$(date +%T) reproduction failed, stopped"; exit 1; }
echo "$(date +%T) reproduction done"
HAT_PILOT=1 bash scripts/rbd2/hat_chain.sh; echo "$(date +%T) pilot exit $?"
MODELS=qwen bash scripts/rbd2/hat_chain.sh; echo "$(date +%T) qwen exit $?"
MODELS=gemma bash scripts/rbd2/hat_chain.sh; echo "$(date +%T) gemma exit $?"
echo "$(date +%T) HAT_GPU3_DONE"
