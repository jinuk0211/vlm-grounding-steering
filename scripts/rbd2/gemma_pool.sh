#!/bin/bash
# Worker for gemma_chain.sh. Queues in priority order: dev -> ours on test (written after the dev pick) -> baselines.
# Lines "KIND SET OUTDIR TAG START END args..." -> runs/OUTDIR/TAG-START-END.jsonl (resumable); KIND vdd = vdd_decode.py
# --model gemma --attn eager, KIND bench = gemma_bench.py --method TAG. A unit is claimed with mkdir in claims_gemma, only
# when the GPU has MIN_FREE MiB free (other users share these GPUs) twice, 10-30 s apart, so a model another job has just
# started loading is seen. A failed unit drops its claim; after 2 failures it is skipped, except CUDA out-of-memory
# failures (another job took the memory: the other session's Gemma CHAIR workers share GPUs 3 and 5), which are not
# counted and back off 1-3 min. The worker waits while units are still open (running elsewhere, or the ours queue not yet
# written) and exits when every unit is complete or failed twice.
# usage: gemma_pool.sh GPU_INDEX WORKER_NAME [MIN_FREE_MIB]
G=$1; W=$2; MF=${3:-20000}
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_gemma; mkdir -p $L $C
QD=$O/queues/gemma_dev.txt; QO=$O/queues/gemma_ours.txt; QB=$O/queues/gemma_bl.txt
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       VDD_GRAY=$O/gray448.png CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
# run the first claimable unit of queue $1 (return 0), or count its open units into OPEN (return 1)
run_one(){
  while read -r K D OD T S E A; do
    [ -z "$K" ] && continue; case $K in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/gpool-$W.log
    if [ $K = vdd ]; then X=(scripts/rbd2/vdd_decode.py --model gemma --attn eager --data $O/sets/$D.jsonl --tag $T)
    else X=(scripts/rbd2/gemma_bench.py --method $T --set $O/sets/$D.jsonl); fi
    UL=$L/gunit-$W.log
    .venv/bin/python -u "${X[@]}" --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $L/g-$OD-$T.log
    if [ $RC -eq 0 ]; then echo "$(date +%T) end $U" >> $L/gpool-$W.log; return 0; fi
    rm -r $C/$U
    if grep -q OutOfMemoryError $UL; then echo "$(date +%T) OOM $U (not counted)" >> $L/gpool-$W.log; sleep $((RANDOM%120+60))
    else echo "$(date +%T) FAILED $U" >> $L/gpool-$W.log; echo $W >> $C/fail-$U; fi
    return 0
  done < $1
  return 1
}
while true; do
  OPEN=0
  run_one $QD && continue
  [ -f $QO ] && run_one $QO && continue
  run_one $QB && continue
  if [ $OPEN -eq 0 ] && { [ -f $QO ] || [ -f $O/runs/gemma_throne_dev60/PICK_FAILED ]; }; then break; fi
  sleep 120
done
echo "$(date +%T) POOL_DONE" >> $L/gpool-$W.log
