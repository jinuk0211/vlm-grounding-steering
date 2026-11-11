#!/bin/bash
# usage: run_queue.sh QUEUE DATA OUTDIR START END [MODEL]   -- runs each "TAG ARGS" line of QUEUE in order, skipping finished runs
Q=$1; D=$2; O=$3; S=$4; E=$5; M=${6:-qwen}
R=/home/project-b/LJW/vlm-grounding-steering
export HF_HOME=${STEER_HF_HOME:-/workspace/tether-pilot/hf} PYTHONPATH=/workspace/steer:$R/scripts:$R/src CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1
export VDD_GRAY=${VDD_GRAY:-/workspace/steer/gray448.png} HF_HUB_OFFLINE=1
PY=${PY:-/workspace/steer/venv/bin/python}
mkdir -p $O
while read -r T A; do
  [ -z "$T" ] && continue; case $T in \#*) continue;; esac
  F=$O/$T-$S-$E.jsonl
  if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then echo "$(date +%T) skip $T (complete)"; continue; fi
  echo "$(date +%T) start $T $A"
  $PY /workspace/steer/vdd_steer.py --model $M --data $D --output $O --tag $T --start $S --end $E $A >> $O/$T-$S-$E.log 2>&1 \
    && echo "$(date +%T) done $T" || echo "$(date +%T) FAILED $T (see $O/$T-$S-$E.log)"
done < $Q
echo "$(date +%T) QUEUE_DONE"
