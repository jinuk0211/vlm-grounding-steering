#!/bin/bash
# usage: vrun.sh GPU DATA OUTDIR TAG END [vdd_decode args...]
G=$1; D=$2; O=$3; T=$4; E=$5; shift 5
R=/home/project-b/LJW/vlm-grounding-steering
export HF_HOME=/workspace/.hf_home PYTHONPATH=/workspace/rbd_pilot:$R/scripts:$R/src CUDA_VISIBLE_DEVICES=$G
source /venv/main/bin/activate >/dev/null
mkdir -p $O /workspace/logs/vdd
python /workspace/rbd_pilot/vdd_decode.py --model qwen --data $D --output $O --tag $T --start 0 --end $E "$@" >> /workspace/logs/vdd/$(basename $O)-$T.log 2>&1
