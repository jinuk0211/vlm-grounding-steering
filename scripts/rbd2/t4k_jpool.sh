#!/bin/bash
# THRONE rows 1000-4999 (t4k_chain.sh): judge worker. Queue lines "MODEL TAG BLOCK JUDGE S-E,S-E,..." (the decode units of that block); a unit is
# ready when those part files are complete, done when runs/t4k_MODEL/eval/TAG/bBLOCK-JUDGE/combined holds the judge's file. Claimed with mkdir in
# claims_t4kj only when the GPU has the judge's MiB free twice, 10-30 s apart (t4k/jgate: lines "JUDGE MIB"). MODE light = only the judges listed
# in t4k/light (default "base large"; small enough to share a GPU with two decoders), all = every judge. Both files are re-read every pass.
# OOM not counted (back off 1-3 min); other failures twice -> skipped. Leaves when no unit is left that it may claim (done, failed twice, or held by
# another worker); waits while such units are not ready yet. t4k/STOP: leave after the current unit.   usage: t4k_jpool.sh GPU_INDEX WORKER MODE
G=$1; W=$2; MODE=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; LG=$O/logs; C=$O/claims_t4kj; Q=$O/queues/t4k_judge.txt; mkdir -p $LG $C $O/t4k
export BENCH=$L/bench RBD2_SETS=$O/sets HF_HOME=$L/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache
UU=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
gate(){ local g; g=$(awk -v j=$1 '$1==j{print $2}' $O/t4k/jgate 2>/dev/null); echo ${g:-16000}; }
mine(){ [ $MODE = all ] && return 0; local l; l=$(cat $O/t4k/light 2>/dev/null || echo base large); [[ " $l " == *" $1 "* ]]; }
ready(){ local se; for se in ${5//,/ }; do f=$O/runs/t4k_parts_$1/$2-$se.jsonl; [ -f $f ] && [ $(wc -l < $f) -ge $(( ${se#*-} - ${se%-*} )) ] || return 1; done; }
while [ ! -f $O/t4k/STOP ]; do
  RAN=; LEFT=0
  while read -r M T B J N; do
    [ -z "$M" ] && continue; case $M in \#*) continue;; esac
    mine $J || continue
    U=$M-$T-b$B-$J
    ls $O/runs/t4k_$M/eval/$T/b$B-$J/combined/*flan-t5-$J.json > /dev/null 2>&1 && continue
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    [ -d $C/$U ] && continue
    LEFT=$((LEFT+1))
    ready $M $T $B $J $N || continue
    MF=$(gate $J); [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; echo "$(date '+%m-%d %T') start $U" >> $LG/t4kj-$W.log; UL=$LG/t4kjunit-$W.log
    $L/venvs/throne/bin/python -u scripts/rbd2/t4k_judge.py $UU $M $T $B $J < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $LG/t4kj-$M-$T.log
    if [ $RC -eq 0 ] && ls $O/runs/t4k_$M/eval/$T/b$B-$J/combined/*flan-t5-$J.json > /dev/null 2>&1; then echo "$(date '+%m-%d %T') end $U" >> $LG/t4kj-$W.log
    else
      rm -r $C/$U
      if grep -q 'OutOfMemoryError\|CUDA out of memory' $UL; then echo "$(date '+%m-%d %T') OOM $U (not counted)" >> $LG/t4kj-$W.log; sleep $((RANDOM%120+60))
      else echo "$(date '+%m-%d %T') FAILED $U rc $RC" >> $LG/t4kj-$W.log; echo $W >> $C/fail-$U; fi
    fi
    RAN=1; break
  done < $Q
  [ -n "$RAN" ] && continue
  [ $LEFT -eq 0 ] && break
  sleep 120
done
echo "$(date '+%m-%d %T') JPOOL_DONE $MODE" >> $LG/t4kj-$W.log
