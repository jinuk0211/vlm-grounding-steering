#!/bin/bash
# Existing baselines (and Base) through baseline_bench.py: workers share one queue, a unit is claimed with mkdir (atomic).
# Queue lines "METHOD ATTN SET OUTDIR START END" -> sets/SET.jsonl rows START-END into runs/OUTDIR/METHOD-START-END.jsonl
# (resumable). A failed unit drops its claim and is retried in the second pass; after two failures of a method (any
# workers) its remaining units are skipped, so a broken method does not reload the model for every chunk.
# usage: pb_bl_pool.sh GPU_INDEX WORKER_NAME QUEUE
G=$1; W=$2; Q=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_bl; mkdir -p $L $C
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
for pass in 1 2; do
  while read -r M A S OD B E; do
    [ -z "$M" ] && continue; case $M in \#*) continue;; esac
    [ $(cat $C/fail-$M 2>/dev/null | wc -l) -ge 2 ] && continue
    OUT=$O/runs/$OD; F=$OUT/$M-$B-$E.jsonl; K=$C/$OD-$M-$B-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    mkdir $K 2>/dev/null || continue
    echo $W > $K/worker; mkdir -p $OUT
    echo "$(date +%T) start $OD $M $B-$E $A" >> $L/blpool-$W.log
    if .venv/bin/python -u scripts/rbd2/baseline_bench.py --method $M --attn $A --set $O/sets/$S.jsonl --output $OUT --start $B --end $E \
         < /dev/null >> $L/$OD-$M.log 2>&1; then echo "$(date +%T) end $OD $M $B-$E" >> $L/blpool-$W.log
    else echo "$(date +%T) FAILED $OD $M $B-$E" >> $L/blpool-$W.log; echo "$W $B-$E" >> $C/fail-$M; rm -r $K; fi
  done < $Q
done
echo "$(date +%T) BLPOOL_DONE" >> $L/blpool-$W.log
