#!/bin/bash
# VCD / M3ID CHAIR 500 worker: takes units "MODEL METHOD START END" from $O/queue.txt (claims by mkdir).
# usage: cdm_pool.sh GPU_INDEX WORKER_NAME MIN_FREE_MIB
# The GPU lock is held from the memory check until the model is loaded, so two workers never race the gate.
G=$1; W=$2; MF=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/cdm-20261109; Q=$O/queue.txt; C=$O/claims
SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl
mkdir -p $O/logs $C $L/.cache/tmp
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
attn(){ case $1 in gemma) echo eager;; *) echo sdpa;; esac; }   # attention of each model's vdd Base rows
log(){ echo "$(date '+%m-%d %T') $*" >> $O/logs/pool-$W.log; }
log "worker $W GPU $G gate $MF MiB start"
while true; do
  [ -f $O/STOP ] && { log STOP; break; }
  OPEN=0; RAN=0
  while read -r MOD MET S E; do
    [ -z "$MOD" ] && continue; case $MOD in \#*) continue;; esac
    OUT=$O/runs/$MOD; F=$OUT/$MET-$S-$E.jsonl; U=$MOD-$MET-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    exec 9>$C/gpu$G.lock; flock 9
    if [ $(free) -lt $MF ] || ! mkdir $C/$U 2>/dev/null; then flock -u 9; exec 9>&-; continue; fi
    echo $W > $C/$U/worker; mkdir -p $OUT; ST=$OUT/$MET-$S-$E-status.json; rm -f $ST.loaded
    log "start $U"
    nice -n 5 .venv/bin/python -u scripts/rbd2/cdm_decode.py --model $MOD --method $MET --attn $(attn $MOD) --data $SET \
       --tag $MET --output $OUT --start $S --end $E < /dev/null 9>&- >> $O/logs/$MOD-$MET.log 2>&1 &
    P=$!; sleep 20
    for i in $(seq 1 60); do grep -q '"stage": "running"' $ST 2>/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 5; done
    flock -u 9; exec 9>&-
    if wait $P; then log "end $U"; else log "FAILED $U"; echo $W >> $C/fail-$U; fi
    rm -r "${C:?}/${U:?}"; RAN=1; break
  done < $Q
  [ $RAN -eq 1 ] && continue
  [ $OPEN -eq 0 ] && break
  sleep 30
done
log POOL_DONE
