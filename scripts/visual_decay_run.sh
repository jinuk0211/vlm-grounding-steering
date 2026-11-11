#!/bin/bash
# Visual-decay analysis: collect on one GPU (shards run one after another), then summarize and plot on CPU.
# usage: bash scripts/visual_decay_run.sh GPU_INDEX MODEL [START END [SHARD] [--regenerate]]
#   bash scripts/visual_decay_run.sh 1 qwen            -> captions 0-500 in shards of 125
#   bash scripts/visual_decay_run.sh 1 gemma 0 300     -> the fixed 300-caption subset
# Refuses to start unless the GPU has NEED_GB free (default 30), so it cannot push other jobs into OOM.
set -euo pipefail
GPU=$1; MODEL=$2; START=${3:-0}; END=${4:-500}; SHARD=${5:-125}; EXTRA=${6:-}
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
O=$R/outputs/visual-decay-20261030; mkdir -p $O/logs
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts"
NEED_GB=${NEED_GB:-30}
free_gb(){ nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
UUID=$(nvidia-smi --id=$GPU --query-gpu=uuid --format=csv,noheader)
for ((s=START; s<END; s+=SHARD)); do
  e=$(( s+SHARD<END ? s+SHARD : END ))
  f=$(free_gb); if (( f < NEED_GB )); then echo "GPU $GPU has ${f} GB free (< ${NEED_GB}); stopping before shard $s-$e" >&2; exit 3; fi
  echo "$(date +%T) $MODEL $s-$e on GPU $GPU (${f} GB free)"
  env $ENV CUDA_VISIBLE_DEVICES=$UUID .venv/bin/python -u scripts/visual_decay_collect.py --model $MODEL --output $O --start $s --end $e $EXTRA > $O/logs/$MODEL-$s-$e.log 2>&1
done
TAG=$MODEL; [[ "$EXTRA" == *--regenerate* ]] && TAG=$MODEL-regen
env $ENV CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/visual_decay_analyze.py $O $TAG > $O/logs/$TAG-analyze.log 2>&1
S=$(ls $O/*-visual-decay-summary.json); env $ENV .venv/bin/python scripts/visual_decay_plot.py $O/visual_decay_report.html $S
echo "$(date +%T) done: $O/visual_decay_report.html"
