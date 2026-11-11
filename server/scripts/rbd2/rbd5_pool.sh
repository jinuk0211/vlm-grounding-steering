#!/bin/bash
# Worker for rbd5_chain.sh. Queue lines "MODEL ATTN SET OUTDIR TAG START END args..." -> vdd4_decode.py into runs/OUTDIR/TAG-START-END.jsonl
# (resumable). A unit is claimed with mkdir in claims_rbd5, after two free-memory checks 10-30 s apart (other jobs share these GPUs).
# OOM is not counted as a failure; a unit that fails otherwise twice is skipped.
# usage: rbd5_pool.sh GPU_INDEX WORKER_NAME QUEUE MIN_FREE_MIB
G=$1; W=$2; Q=$3; MF=$4
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_rbd5; mkdir -p $L $C
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       VDD_GRAY=$O/gray448.png CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
run_one(){
  while read -r M AT D OD T S E A; do
    [ -z "$M" ] && continue; case $M in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/r5pool-$W.log
    UL=$L/r5unit-$W.log
    .venv/bin/python -u scripts/rbd2/vdd4_decode.py --model $M --attn $AT --data $O/sets/$D.jsonl --tag $T --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $L/r5-$OD-$T.log
    if [ $RC -eq 0 ]; then echo "$(date +%T) end $U" >> $L/r5pool-$W.log; return 0; fi
    rm -r "${C:?}/${U:?}"
    if grep -q OutOfMemoryError $UL; then echo "$(date +%T) OOM $U (not counted)" >> $L/r5pool-$W.log; sleep $((RANDOM%120+60))
    else echo "$(date +%T) FAILED $U" >> $L/r5pool-$W.log; echo $W >> $C/fail-$U; fi
    return 0
  done < $Q
  return 1
}
while true; do
  OPEN=0
  run_one && continue
  [ $OPEN -eq 0 ] && break
  sleep 60
done
echo "$(date +%T) POOL_DONE" >> $L/r5pool-$W.log
