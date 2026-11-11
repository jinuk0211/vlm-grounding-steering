#!/bin/bash
# Worker pool for Qwen3-VL CHAIR.
# usage: qchair_pool.sh GPU_INDEX WORKER_NAME MIN_FREE_MIB
G=$1; W=$2; MF=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; L=$O/logs; C=$O/claims_qchair; mkdir -p $L $C
QD=$O/queues/qchair_dev.txt; QO=$O/queues/qchair_ours.txt; QB=$O/queues/qchair_base.txt; PF=$O/runs/qchair_t60/PICK_FAILED
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
    OPEN=$((OPEN+1)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date +%T) start $U" >> $L/qcpool-$W.log
    if .venv/bin/python -u scripts/rbd2/vdd_decode.py --model qwen3 --attn sdpa --data $O/sets/$D.jsonl --tag $T \
         --output $OUT --start $S --end $E $A < /dev/null >> $L/qc-$OD-$T.log 2>&1
    then echo "$(date +%T) end $U" >> $L/qcpool-$W.log
    else echo "$(date +%T) FAILED $U" >> $L/qcpool-$W.log; echo $W >> $C/fail-$U; rm -r "${C:?}/${U:?}"; fi
    return 0
  done < $1
  return 1
}
while true; do
  OPEN=0
  run_one $QD && continue
  [ -f $QO ] && run_one $QO && continue
  [ -f $QB ] && run_one $QB && continue
  if [ $OPEN -eq 0 ] && { [ -f $QO ] || [ -f $PF ]; }; then break; fi
  sleep 20
done
echo "$(date +%T) POOL_DONE" >> $L/qcpool-$W.log
