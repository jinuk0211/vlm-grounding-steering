#!/bin/bash
# project-b port of rbd2 worker.sh + sweep.sh: run a queue in order on one GPU, skipping complete runs.
# usage: pb_worker.sh GPU_INDEX QUEUE MODE      MODE=test  lines "SET TAG args..."  (vdd_decode.py, 60 images)
#                                               MODE=sweep lines "TAG args..."      (sweep_decode.py on chair_test60)
G=$1; Q=$2; MODE=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; mkdir -p $L
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/scripts VDD_GRAY=$O/gray448.png \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
while read -r a b c; do
  [ -z "$a" ] && continue; case $a in \#*) continue;; esac
  if [ "$MODE" = test ]; then S=$a; T=$b; A=$c; D=$O/sets/$S.jsonl; OUT=$O/runs/$S; PYF=scripts/rbd2/vdd_decode.py
  else S=chair_test60; T=$a; A="$b $c"; D=$O/sets/chair_test60.jsonl; OUT=$O/runs/chair_test60_sweep; PYF=scripts/rbd2/sweep_decode.py; fi
  mkdir -p $OUT
  if [ -f $OUT/$T-0-60.jsonl ] && [ $(wc -l < $OUT/$T-0-60.jsonl) -ge 60 ]; then continue; fi
  echo "$(date +%T) start $S $T $A" >> $L/worker$G.log
  .venv/bin/python -u $PYF --model qwen --data $D --output $OUT --tag $T --start 0 --end 60 $A < /dev/null >> $L/$S-$T.log 2>&1 \
    || echo "$(date +%T) FAILED $S $T" >> $L/worker$G.log
  echo "$(date +%T) end $S $T" >> $L/worker$G.log
done < $Q
echo "$(date +%T) QUEUE_DONE $Q" >> $L/worker$G.log
