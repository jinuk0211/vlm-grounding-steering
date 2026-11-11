#!/bin/bash
# User 11-10 21:27: "belief 효과를 하이퍼파라미터 약한거 써서 줄여봐" -> Gemma CHAIR 500 ablation with belief strength lam 0.5 (was 1;
# one a-priori value, half, no search). Same anchor steering as gemma_abl.sh (layer 24 resid alpha 1 + anchornorm layer 24 rho 0.3 K 16,
# pen 0.5, eager, GPU 0 A6000). Decoded: Full, -steering (alpha 0), -penalty (pen 0), all at lam 0.5. -belief (lam 0) does not depend on
# lam: abl_nobelief_gemma from gemma_abl.sh is reused (its decoder keeps running). Base = stored gabl_chair500_all/base.
# The lam 1 run on GPU 0 is NOT touched (the user likes it); this runs in parallel on another A6000 (GPU 5, same type as GPU 0), and
# claims a unit only when the GPU has MIN_FREE MiB free twice, 10-30 s apart (GPU 5 is shared with the other session's Qwen2.5 dev).
# usage: gemma_l05.sh GPU_INDEX WORKER [MIN_FREE_MIB=21500]
G=$1; W=$2; MF=${3:-21500}; source $(dirname $0)/steer_env.sh
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
O=$R/outputs/steer-anchor-20261110/gemma; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl; P=$R/outputs/rbd2-20261102/runs/gabl_chair500_all
C=$O/claims_l05; mkdir -p $C
ST="--layers 24 --alpha 1 --layers2 24 $NEW"
A="--model gemma --data $SET --output $O --attn eager --lam 0.5"
UNITS="anchor_gemma_l05:0:500:STEER abl_nosteer_gemma_l05:0:500:--steer_none abl_nopen_gemma_l05:0:250:STEER_--pen_0 abl_nopen_gemma_l05:250:500:STEER_--pen_0"
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/abl_l05.log; }
log "worker start GPU $G ($CUDA_VISIBLE_DEVICES) gate $MF MiB"
while true; do
  RAN=0; LEFT=0
  for u in $UNITS; do
    IFS=: read T B E X <<< "$u"; X=${X//_/ }; X=${X//STEER/$ST}; F=$O/$T-$B-$E.jsonl      # B = first row ($S is the scripts dir)
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    [ -d $C/$T-$B-$E ] && continue
    LEFT=1
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$T-$B-$E 2>/dev/null || continue
    log "start $T $B-$E $X"
    nice -n 5 .venv/bin/python $S/vdd_steer.py $A --tag $T --start $B --end $E $X < /dev/null >> $O/$T-$B-$E.log 2>&1 \
      && log "done $T $B-$E" || log "FAILED $T $B-$E"
    rm -r $C/$T-$B-$E; RAN=1; break
  done
  [ $RAN -eq 1 ] && continue
  [ $LEFT -eq 0 ] && break
  sleep 60
done
for u in $UNITS; do IFS=: read T B E X <<< "$u"; F=$O/$T-$B-$E.jsonl; [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ] || { log "leaves (units left to the other worker)"; exit 0; }; done
NB=$O/abl_nobelief_gemma-0-500.jsonl; while [ ! -f $NB ] || [ $(wc -l < $NB) -lt 500 ]; do sleep 120; done
mkdir $C/table 2>/dev/null || exit 0
cat $O/abl_nopen_gemma_l05-0-250.jsonl $O/abl_nopen_gemma_l05-250-500.jsonl > $O/abl_nopen_gemma_l05-0-500.jsonl
.venv/bin/python $S/steer_table.py chair_test500 $R/data/hallucination-suite/chair-scorer-state.json \
  "Full (lam .5)=$O/anchor_gemma_l05-0-500.jsonl" "- steering=$O/abl_nosteer_gemma_l05-0-500.jsonl" "- penalty=$O/abl_nopen_gemma_l05-0-500.jsonl" \
  "- belief=$NB" "Base=$P/base-0-500.jsonl" "old Full=$P/rbd_full-0-500.jsonl" > $O/abl_table_l05.txt 2>> $O/abl_l05.log
log "GEMMA_L05_DONE ($O/abl_table_l05.txt)"
