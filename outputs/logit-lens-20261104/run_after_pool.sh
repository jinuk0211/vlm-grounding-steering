#!/bin/bash
# Logit lens on GPU 2 once the RBD2 Gemma CHAIR chain (gchair3.sh) and all its pool workers have exited.
# Before each model load, waits until GPU 2 has >= 20 GB free on two checks a minute apart (other users share it).
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/logit-lens-20261104; cd $R
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts OMP_NUM_THREADS=8
U=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
free(){ nvidia-smi --id=2 --query-gpu=memory.free --format=csv,noheader,nounits; }
gate(){ while true; do a=$(free); sleep 60; b=$(free); [ $a -ge 20000 ] && [ $b -ge 20000 ] && return; done; }
busy(){ ps -eo args | awk '$0 ~ /rbd2\/gchair3\.sh|rbd2\/gchair_pool\.sh/ && $1 ~ /bash/' | grep -q .; }
while busy; do sleep 120; done; echo "$(date '+%m-%d %T') gchair chain and workers gone"
for m in qwen gemma; do for mode in image text; do
  gate; echo "$(date '+%m-%d %T') start $m $mode ($(free) MiB free)"
  CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/logit_lens.py $mode --model $m --output $O > $O/logs/$m-$mode.log 2>&1 \
    && echo "$(date '+%m-%d %T') done $m $mode" || echo "$(date '+%m-%d %T') FAILED $m $mode"
done; CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/logit_lens.py analyze --model $m --output $O > $O/logs/$m-analyze.log 2>&1; echo "$(date '+%m-%d %T') analyzed $m"; done
echo "$(date '+%m-%d %T') ALL_DONE"
