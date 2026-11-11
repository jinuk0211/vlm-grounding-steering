#!/bin/bash
# Hands GPUs 4,5 on when the Qwen3 GrAInS chain has ended, success or failure, with no Claude session needed (user 11-05 evening; the running chain is not edited).
# Waits for the chain pid, then until no Qwen3 GrAInS process is left on GPUs 4,5, then writes the two signal files:
#   hat385/join_gpu45 (h-17: HAT+ALOHa joins GPUs 4,5), gabl_qwen3_done (h-65: Gemma ablation, which also waits for HAT_GPU3_DONE).
# usage: qbl_after.sh CHAIN_PID
O=/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102; LOG=$O/logs/qbl.log; C=$1
while [ -n "$(ps -o pid= -p $C)" ]; do sleep 30; done
echo "=== $(date +%T) chain $C ended; checking GPUs 4,5" >> $LOG
while :; do busy=0
  for p in $(nvidia-smi --id=4,5 --query-compute-apps=pid --format=csv,noheader); do
    case "$(ps -o args= -p $p)" in *qwen3_bl.py*|*extract_qwen3_grains.py*) busy=1;; esac; done
  [ $busy = 0 ] && break; sleep 30; done
touch $O/hat385/join_gpu45 $O/gabl_qwen3_done
echo "=== $(date +%T) QBL_HANDOVER: hat385/join_gpu45 and gabl_qwen3_done written" >> $LOG
