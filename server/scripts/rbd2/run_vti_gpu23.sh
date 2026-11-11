#!/bin/bash
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
source scripts/ljw_env.sh
export PYTHONPATH=scripts:scripts/rbd2:src:$PYTHONPATH
OUT=$R/outputs/rbd2-20261102/runs/qchair_test500_tbl
SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl

GPU=$1; S=$2; E=$3
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$GPU --query-gpu=uuid --format=csv,noheader)
echo "$(date +%T) Starting VTI on GPU $GPU ($S-$E)" >> $OUT/vti_gpu$GPU.log
.venv/bin/python -u scripts/rbd2/qwen3_bench.py --method vti --set $SET --output $OUT --start $S --end $E --attn sdpa >> $OUT/vti_gpu$GPU.log 2>&1
echo "$(date +%T) Finished VTI on GPU $GPU ($S-$E)" >> $OUT/vti_gpu$GPU.log
