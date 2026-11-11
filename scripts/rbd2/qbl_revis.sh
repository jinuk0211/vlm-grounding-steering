#!/bin/bash
# User (11-05 20:2x): run the official REVIS code. Artifacts on GPU 5 (one Qwen3, waits for 18.9 GB free), then its 500 CHAIR units go into the running GrAInS chain's queue (queues/qbl.txt), after the
# GrAInS units, so qbl_chain.sh's workers and report handle them and qbl_after.sh hands the GPUs over only after both. Log: logs/qbl.log. QBL_REVIS_FAILED if a stage stops.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/qbl.log; Q=$O/queues/qbl.txt; C=$1
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=5 --query-gpu=uuid --format=csv,noheader)
until [ $(nvidia-smi --id=5 --query-gpu=memory.free --format=csv,noheader,nounits) -ge 18900 ]; do sleep 20; done
echo "=== $(date +%T) REVIS official: vectors and calibration on GPU 5" >> $LOG
if ! .venv/bin/python -u scripts/rbd2/revis_official_q3.py >> $LOG 2>$O/logs/qbl_revis.err; then echo "=== $(date +%T) QBL_REVIS_FAILED (logs/qbl_revis.err)" >> $LOG; exit 1; fi
until grep -q grains_clone $Q 2>/dev/null; do [ -z "$(ps -o pid= -p $C)" ] && { echo "=== $(date +%T) QBL_REVIS_FAILED: chain $C ended before its queue was written" >> $LOG; exit 1; }; sleep 20; done
[ -n "$(ps -o pid= -p $C)" ] || { echo "=== $(date +%T) QBL_REVIS_FAILED: chain $C has ended" >> $LOG; exit 1; }
for b in $(seq 0 50 450); do echo "revis_official sdpa $b $((b+50))"; done >> $Q
echo "=== $(date +%T) REVIS official artifacts ready; 10 units added to the queue" >> $LOG
