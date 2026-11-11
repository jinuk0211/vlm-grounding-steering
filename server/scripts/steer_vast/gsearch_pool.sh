#!/bin/bash
# User 11-10 22:05 (/loop until 11-11 10:00): make TETHER's steering work on Gemma 4 E4B, CHAIR test set only. Same framework (Eq. steer_cur
# resid alpha s(t) d_t + Eq. anchor/steer anchornorm rho s(t) |h'| v/|v|, belief + penalty); only layer / strengths / K / schedule vary.
# Candidates on test rows 0-99 first; the best goes to 500 rows. Belief lam 1, pen 0.5, eager (as the stored Gemma rows, so the stored
# -steering and Base rows of gabl_chair500_all are the references).
# Persistent worker: units "TAG START END ARGS..." from $Q (re-read every pass, appended by hand), claimed by mkdir; a unit that failed
# twice is skipped; waits when nothing is left; leaves on $O/STOP.   usage: gsearch_pool.sh GPU_INDEX WORKER [MIN_FREE_MIB=19000]
G=$1; W=$2; MF=${3:-19000}; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-gemma-search-20261111; Q=$O/queue.txt; C=$O/claims; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl
mkdir -p $C; touch $Q
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/pool.log; }
log "worker start GPU $G ($CUDA_VISIBLE_DEVICES) gate $MF"
while [ ! -f $O/STOP ]; do
  RAN=0
  while read -r T B E X; do
    [ -z "$T" ] && continue; case $T in \#*) continue;; esac
    F=$O/$T-$B-$E.jsonl; K=$T-$B-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    [ $(cat $C/fail-$K 2>/dev/null | wc -l) -ge 2 ] && continue
    [ -d $C/$K ] && continue
    [ $(free) -ge $MF ] || continue
    mkdir $C/$K 2>/dev/null || continue
    echo $W > $C/$K/worker; log "start $K $X"
    if nice -n 5 .venv/bin/python $S/vdd_steer.py --model gemma --data $SET --output $O --attn eager --tag $T --start $B --end $E $X \
         < /dev/null >> $O/$K.log 2>&1; then log "done $K"; else log "FAILED $K"; echo $W >> $C/fail-$K; fi
    rm -r $C/$K; RAN=1; break
  done < $Q
  [ $RAN -eq 0 ] && sleep 60
done
log "STOP"
