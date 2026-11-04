#!/bin/bash
# 150-image extension: any number of workers share one queue; a unit is claimed with mkdir (atomic), so each runs once.
# Queue lines "DATA OUTDIR TAG START END args..." -> vdd_decode on sets/DATA.jsonl rows START-END into runs/OUTDIR/TAG-START-END.jsonl
# (partial rows are resumed). A failed unit drops its claim and is retried once in the second pass.
# usage: pb_pool.sh GPU_INDEX WORKER_NAME QUEUE
G=$1; W=$2; Q=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims150; mkdir -p $L $C
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/scripts VDD_GRAY=$O/gray448.png \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
for pass in 1 2; do
  while read -r D OD T S E A; do
    [ -z "$D" ] && continue; case $D in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; K=$C/$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    mkdir $K 2>/dev/null || continue
    echo $W > $K/worker; mkdir -p $OUT
    echo "$(date +%T) start $OD $T $S-$E" >> $L/pool-$W.log
    if .venv/bin/python -u scripts/rbd2/vdd_decode.py --model qwen --data $O/sets/$D.jsonl --output $OUT --tag $T --start $S --end $E $A \
         < /dev/null >> $L/$OD-$T.log 2>&1; then echo "$(date +%T) end $OD $T $S-$E" >> $L/pool-$W.log
    else echo "$(date +%T) FAILED $OD $T $S-$E" >> $L/pool-$W.log; rm -r $K; fi
  done < $Q
done
echo "$(date +%T) POOL_DONE" >> $L/pool-$W.log
