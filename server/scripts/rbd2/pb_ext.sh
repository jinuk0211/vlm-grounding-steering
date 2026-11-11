#!/bin/bash
# Decode images 60-119 of a 120-image set for the configs in LISTFILE ("TAG args..."), in order, on one GPU.
# usage: pb_ext.sh GPU_INDEX SET120 LISTFILE     (rows -> runs/SET120/TAG-60-120.jsonl; partial rows are resumed)
G=$1; S=$2; Q=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; OUT=$O/runs/$S; mkdir -p $OUT
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/scripts VDD_GRAY=$O/gray448.png \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
while read -r T A; do
  [ -z "$T" ] && continue
  if [ -f $OUT/$T-60-120.jsonl ] && [ $(wc -l < $OUT/$T-60-120.jsonl) -ge 60 ]; then continue; fi
  echo "$(date +%T) start $S $T $A" >> $L/ext$G-$S.log
  .venv/bin/python -u scripts/rbd2/vdd_decode.py --model qwen --data $O/sets/$S.jsonl --output $OUT --tag $T --start 60 --end 120 $A \
    < /dev/null >> $L/$S-$T.log 2>&1 || echo "$(date +%T) FAILED $T" >> $L/ext$G-$S.log
  echo "$(date +%T) end $S $T" >> $L/ext$G-$S.log
done < $Q
echo "$(date +%T) EXT_DONE $Q" >> $L/ext$G-$S.log
