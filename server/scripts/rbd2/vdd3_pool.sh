#!/bin/bash
# Worker for vdd3_chain.sh. Queue lines "DATA OUTDIR TAG START END args..." -> vdd3_decode.py --model qwen on
# sets/DATA.jsonl rows START-END into runs/OUTDIR/TAG-START-END.jsonl (resumable). A unit is claimed with mkdir in
# claims_vdd3 only when the GPU has MIN_FREE MiB free twice, 10-30 s apart (other jobs share these GPUs). CUDA
# out-of-memory failures are not counted (back off 1-3 min); other failures twice -> skipped. Exits when every unit is
# complete or failed twice.
# usage: vdd3_pool.sh GPU_INDEX WORKER_NAME QUEUE [MIN_FREE_MIB]
G=$1; W=$2; Q=$3; MF=${4:-19500}
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_vdd3; mkdir -p $L $C
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       VDD_GRAY=$O/gray448.png CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
run_one(){
  while read -r D OD T S E A; do
    [ -z "$D" ] && continue; case $D in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/v3pool-$W.log; UL=$L/v3unit-$W.log
    .venv/bin/python -u scripts/rbd2/vdd3_decode.py --model qwen --data $O/sets/$D.jsonl --tag $T --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $L/v3-$OD-$T.log
    if [ $RC -eq 0 ]; then echo "$(date +%T) end $U" >> $L/v3pool-$W.log; return 0; fi
    rm -r $C/$U
    if grep -q OutOfMemoryError $UL; then echo "$(date +%T) OOM $U (not counted)" >> $L/v3pool-$W.log; sleep $((RANDOM%120+60))
    else echo "$(date +%T) FAILED $U" >> $L/v3pool-$W.log; echo $W >> $C/fail-$U; fi
    return 0
  done < $Q
  return 1
}
while true; do
  OPEN=0
  run_one && continue
  [ $OPEN -eq 0 ] && break
  sleep 120
done
echo "$(date +%T) POOL_DONE" >> $L/v3pool-$W.log
