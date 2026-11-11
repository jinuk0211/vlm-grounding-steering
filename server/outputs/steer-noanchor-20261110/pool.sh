#!/bin/bash
# usage: pool.sh GPU_INDEX WORKER [MIN_FREE_MIB=21000]; queue lines: MODEL TAG START END ARGS...  (re-read every pass)
G=$1; W=$2; MF=${3:-21000}; source /home/project-b/LJW/vlm-grounding-steering/scripts/steer_vast/steer_env.sh
export PYTHONPATH=$PYTHONPATH:$R/scripts/rbd2
O=$R/outputs/steer-noanchor-20261110; Q=$O/queue.txt; C=$O/claims; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/pool.log; }
log "worker start GPU $G gate $MF"
while [ ! -f $O/STOP ]; do
  RAN=0
  while read -r M T B E X; do
    [ -z "$M" ] && continue; case $M in \#*) continue;; esac
    D=$O/$M; mkdir -p $D; F=$D/$T-$B-$E.jsonl; K=$M-$T-$B-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    [ $(cat $C/fail-$K 2>/dev/null | wc -l) -ge 2 ] && continue
    [ -d $C/$K ] && continue
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%15+5)); [ $(free) -ge $MF ] || continue
    mkdir $C/$K 2>/dev/null || continue
    echo $W > $C/$K/worker; AT=sdpa; [ $M = gemma ] && AT=eager
    log "start $K $X"
    if nice -n 5 .venv/bin/python $S/vdd_steer.py --model $M --data $SET --output $D --attn $AT --tag $T --start $B --end $E $X \
         < /dev/null >> $D/$T-$B-$E.log 2>&1; then log "done $K"; else log "FAILED $K"; echo $W >> $C/fail-$K; fi
    rm -r $C/$K; RAN=1; sleep 40; break
  done < $Q
  [ $RAN -eq 0 ] && sleep 60
done
log "STOP"
