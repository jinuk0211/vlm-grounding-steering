#!/bin/bash
# usage: na_run.sh TAG START END ARGS...   (Qwen2.5-VL, sdpa, chair_test500, project-b layout replicated at the same paths)
T=$1; B=$2; E=$3; shift 3
R=/home/project-b/LJW/vlm-grounding-steering; O=/workspace/na/qwen; mkdir -p $O /workspace/tmp
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/workspace/tmp \
       VDD_GRAY=$R/outputs/rbd2-20261102/gray448.png PYTHONPATH=$R/scripts/steer_vast:$R/scripts:$R/src RBD2_SETS=$R/outputs/rbd2-20261102/sets \
       OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
cd $R && /workspace/na-venv/bin/python $R/scripts/steer_vast/vdd_steer.py --model qwen --data $R/outputs/rbd2-20261102/sets/chair_test500.jsonl \
   --output $O --attn sdpa --tag $T --start $B --end $E "$@" >> $O/$T-$B-$E.log 2>&1
echo "$(date "+%m-%d %T") exit $? $T $B-$E" >> /workspace/logs/na.log
