#!/bin/bash
# Worker for hat_chain.sh. Queue lines "MODEL KIND TAG START END args..." -> runs/${HAT_RUN}_MODEL/TAG-START-END.jsonl (resumable).
# KIND vdd = vdd4_decode.py (RBD and vdd Base; --vobj), bench = baseline_bench.py (qwen) / qwen3_bench.py (qwen3) / gemma_bench.py (gemma),
# grains = qwen3_bl.py (qwen3), vti = vti_ws_run.py at 0.9/0.9 (qwen). A unit is claimed with mkdir in claims_$HAT_RUN only when the GPU has the model's MiB free twice, 10-30 s apart
# (other jobs share these GPUs). CUDA out-of-memory failures are not counted (back off 1-3 min); other failures twice -> skipped.
# Exits when every unit is complete or failed twice.    usage: hat_pool.sh GPU_INDEX WORKER_NAME QUEUE
G=$1; W=$2; Q=$3; RUN=${HAT_RUN:-hat385}
R=${HAT_L:-/home/project-b/LJW}/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_$RUN; mkdir -p $L $C
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$R/../.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       VDD_GRAY=$O/gray448.png TMPDIR=$R/../.cache/tmp XDG_CACHE_HOME=$R/../.cache CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
D=$O/sets/$RUN.jsonl
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
run_one(){
  while read -r M K T S E A; do
    [ -z "$M" ] && continue; case $M in \#*) continue;; esac
    OUT=$O/runs/${RUN}_$M; F=$OUT/$T-$S-$E.jsonl; U=$M-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    if [ $M = gemma ]; then MF=21257; else MF=19500; fi
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/hatpool-$W.log; UL=$L/hatunit-$W.log
    case $M/$K in
      */vdd)      X=(scripts/rbd2/vdd4_decode.py --model $M --data $D --tag $T);;
      qwen/vti)   X=(scripts/rbd2/vti_ws_run.py --av 0.9 --at 0.9 --tag $T --set $D);;
      qwen/bench) X=(scripts/rbd2/baseline_bench.py --method $T --set $D);;
      qwen3/bench) X=(scripts/rbd2/qwen3_bench.py --method $T --set $D);;
      qwen3/grains) X=(scripts/rbd2/qwen3_bl.py --method $T --set $D);;
      *)          X=(scripts/rbd2/gemma_bench.py --method $T --set $D);;
    esac
    .venv/bin/python -u "${X[@]}" --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $L/hat-$M-$T.log
    if [ $RC -eq 0 ]; then echo "$(date +%T) end $U" >> $L/hatpool-$W.log; return 0; fi
    rm -r $C/$U
    if grep -q OutOfMemoryError $UL; then echo "$(date +%T) OOM $U (not counted)" >> $L/hatpool-$W.log; sleep $((RANDOM%120+60))
    else echo "$(date +%T) FAILED $U" >> $L/hatpool-$W.log; echo $W >> $C/fail-$U; fi
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
echo "$(date +%T) POOL_DONE" >> $L/hatpool-$W.log
