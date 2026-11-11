#!/bin/bash
# Generic vdd_steer.py worker: units "TAG START END ARGS..." from UNITS_FILE (re-read every pass), claimed by mkdir in OUTDIR/claims,
# partial files resumed by vdd_steer.py; a unit that failed twice is skipped. Several workers may share one GPU and one units file.
# usage: steer_pool.sh GPU_INDEX WORKER UNITS_FILE OUTDIR MODEL ATTN SET.jsonl     log OUTDIR/pool.log (POOL_DONE per worker)
G=$1; W=$2; U=$3; O=$4; M=$5; AT=$6; SET=$7; source $(dirname $0)/steer_env.sh
mkdir -p $O/claims
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/pool.log; }
log "worker start GPU $G ($CUDA_VISIBLE_DEVICES) units $U"
while true; do
  RAN=0
  while read -r T B E X; do
    [ -z "$T" ] && continue; case $T in \#*) continue;; esac
    F=$O/$T-$B-$E.jsonl; K=$T-$B-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    [ $(cat $O/claims/fail-$K 2>/dev/null | wc -l) -ge 2 ] && continue
    mkdir $O/claims/$K 2>/dev/null || continue
    log "start $K $X"
    if nice -n 5 .venv/bin/python $S/vdd_steer.py --model $M --data $SET --output $O --attn $AT --tag $T --start $B --end $E $X \
         < /dev/null >> $O/$K.log 2>&1; then log "done $K"; else log "FAILED $K"; echo $W >> $O/claims/fail-$K; fi
    rm -r $O/claims/$K; RAN=1; break
  done < $U
  [ $RAN -eq 0 ] && break
done
log POOL_DONE
