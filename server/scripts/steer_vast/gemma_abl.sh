#!/bin/bash
# User 11-10 20:20: Gemma 4 E4B CHAIR 500 ablation with the anchor steering, as the Qwen2.5 table. Full = layer 24 resid alpha 1 (the
# stored Gemma Full) + anchornorm layer 24 rho 0.3 K 16, lam 1, pen 0.5, eager. Decoded here: Full, -penalty (pen 0), -belief (lam 0), on
# GPU 0 (A6000: vdd_steer reproduced the stored Gemma Full rows there, 2 of 2). -steering (alpha 0, lam 1, pen 0.5) and Base do not depend
# on the steering variant: the stored rows outputs/rbd2-20261102/runs/gabl_chair500_all/{abl_nosteer,base}-0-500.jsonl are used.
# Workers pull units in order (claim = mkdir); the last one to find every unit complete writes table.txt.
# usage: gemma_abl.sh GPU_INDEX WORKER        log outputs/steer-anchor-20261110/gemma/abl.log (GEMMA_ABL_DONE)
G=$1; W=$2; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-anchor-20261110/gemma; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl; P=$R/outputs/rbd2-20261102/runs/gabl_chair500_all
C=$O/claims; mkdir -p $C
A="--model gemma --data $SET --output $O --attn eager --layers 24 --alpha 1 --layers2 24 $NEW"
UNITS="anchor_gemma:0:500: abl_nobelief_gemma:0:500:--lam_0 abl_nopen_gemma:0:250:--pen_0 abl_nopen_gemma:250:500:--pen_0"
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/abl.log; }
log "worker start GPU $G ($CUDA_VISIBLE_DEVICES)"
while true; do
  RAN=0
  for u in $UNITS; do
    IFS=: read T B E X <<< "$u"; X=${X//_/ }; F=$O/$T-$B-$E.jsonl      # B = first row ($S is the scripts dir)
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    mkdir $C/$T-$B-$E 2>/dev/null || continue
    log "start $T $B-$E $X"
    nice -n 5 .venv/bin/python $S/vdd_steer.py $A --tag $T --start $B --end $E $X < /dev/null >> $O/$T-$B-$E.log 2>&1 \
      && log "done $T $B-$E" || log "FAILED $T $B-$E"
    rm -r $C/$T-$B-$E; RAN=1; break
  done
  [ $RAN -eq 0 ] && break
done
# every unit complete -> one worker joins the -penalty halves and writes the table
for u in $UNITS; do IFS=: read T B E X <<< "$u"; F=$O/$T-$B-$E.jsonl; [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ] || { log "leaves (units left to the other worker)"; exit 0; }; done
mkdir $C/table 2>/dev/null || exit 0
cat $O/abl_nopen_gemma-0-250.jsonl $O/abl_nopen_gemma-250-500.jsonl > $O/abl_nopen_gemma-0-500.jsonl
.venv/bin/python $S/steer_table.py chair_test500 $R/data/hallucination-suite/chair-scorer-state.json \
  "Full (anchor)=$O/anchor_gemma-0-500.jsonl" "- steering=$P/abl_nosteer-0-500.jsonl" "- penalty=$O/abl_nopen_gemma-0-500.jsonl" \
  "- belief=$O/abl_nobelief_gemma-0-500.jsonl" "Base=$P/base-0-500.jsonl" "old Full=$P/rbd_full-0-500.jsonl" \
  "old - penalty=$P/abl_nopen-0-500.jsonl" "old - belief=$P/abl_nobelief-0-500.jsonl" > $O/abl_table.txt 2>> $O/abl.log
log "GEMMA_ABL_DONE ($O/abl_table.txt)"
