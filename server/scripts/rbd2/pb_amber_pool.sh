#!/bin/bash
# Worker for pb_amber_bl.sh: pb_bl_pool.sh plus VTI units through vti_ws_run.py (tag vtiws_v{av}_t{at}), its own claims
# dir, and a free-memory check before each unit (other users share these GPUs; one Qwen needs ~18 GB).
# Queue lines "METHOD ATTN SET OUTDIR START END" -> runs/OUTDIR/METHOD-START-END.jsonl (resumable).
# usage: pb_amber_pool.sh GPU_INDEX WORKER_NAME QUEUE
G=$1; W=$2; Q=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_amber; mkdir -p $L $C
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
for pass in 1 2; do
  while read -r M A S OD B E; do
    [ -z "$M" ] && continue; case $M in \#*) continue;; esac
    [ $(cat $C/fail-$M 2>/dev/null | wc -l) -ge 2 ] && continue
    OUT=$O/runs/$OD; F=$OUT/$M-$B-$E.jsonl; K=$C/$OD-$M-$B-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    mkdir $K 2>/dev/null || continue
    echo $W > $K/worker; mkdir -p $OUT
    until [ $(free) -ge 19000 ]; do sleep 60; done
    case $M in
      vtiws_*) X=(scripts/rbd2/vti_ws_run.py --av $(echo $M | cut -d_ -f2 | cut -c2-) --at $(echo $M | cut -d_ -f3 | cut -c2-) --tag $M);;
      *) X=(scripts/rbd2/baseline_bench.py --method $M --attn $A);;
    esac
    echo "$(date +%T) start $OD $M $B-$E $A" >> $L/amberpool-$W.log
    if .venv/bin/python -u "${X[@]}" --set $O/sets/$S.jsonl --output $OUT --start $B --end $E \
         < /dev/null >> $L/$OD-$M.log 2>&1; then echo "$(date +%T) end $OD $M $B-$E" >> $L/amberpool-$W.log
    else echo "$(date +%T) FAILED $OD $M $B-$E" >> $L/amberpool-$W.log; echo "$W $B-$E" >> $C/fail-$M; rm -r $K; fi
  done < $Q
done
echo "$(date +%T) AMBERPOOL_DONE" >> $L/amberpool-$W.log
