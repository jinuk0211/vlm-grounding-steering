#!/bin/bash
# User 11-10 ~19:40: can belief win back the recall the anchor steering costs? Qwen2.5, chair_dev100 (train2014 dev, not test), one GPU,
# steering of the box pick full_an16_r03 (resid L16 a2 + anchornorm L16 rho 0.3 K16). Variants: Base, -steering, Full, -belief,
# belief that only raises objects (--bpos: negative belief clipped to 0) at lam 1/2/3, and lam 2.
# Then steer_table.py -> outputs/steer-beliefdev-20261110/table.txt.
# usage: belief_dev.sh GPU_INDEX [WAIT_FILE]   (first waits until WAIT_FILE contains STEER_NOBELIEF_DONE, if given)
G=$1; W=$2; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-beliefdev-20261110; SET=$R/outputs/rbd2-20261102/sets/chair_dev100.jsonl; mkdir -p $O
log(){ echo "$(date '+%m-%d %T') $*" >> $O/run.log; }
if [ -n "$W" ]; then until grep -q STEER_NOBELIEF_DONE $W 2>/dev/null; do sleep 60; done; fi
log "start GPU $G ($CUDA_VISIBLE_DEVICES)"
F="--layers 16 --alpha 2 --layers2 16 $NEW"
while read -r T X; do
  [ -z "$T" ] && continue
  if [ -f $O/$T-0-100.jsonl ] && [ $(wc -l < $O/$T-0-100.jsonl) -ge 100 ]; then continue; fi
  log "start $T $X"
  nice -n 5 .venv/bin/python $S/vdd_steer.py --model qwen --data $SET --output $O --tag $T --start 0 --end 100 $X < /dev/null >> $O/$T-0-100.log 2>&1 \
    && log "done $T" || log "FAILED $T"
done <<LIST
dev_base --steer none --lam 0 --pen 0
dev_nosteer --steer none
dev_full $F
dev_nobelief $F --lam 0
dev_bpos1 $F --bpos
dev_bpos2 $F --bpos --lam 2
dev_bpos3 $F --bpos --lam 3
dev_lam2 $F --lam 2
LIST
.venv/bin/python $S/steer_table.py chair_dev100 $R/data/chair-dev-train2014-100/chair-scorer-state.json \
  "Base=$O/dev_base-0-100.jsonl" "- steering=$O/dev_nosteer-0-100.jsonl" "Full=$O/dev_full-0-100.jsonl" "- belief=$O/dev_nobelief-0-100.jsonl" \
  "bpos lam1=$O/dev_bpos1-0-100.jsonl" "bpos lam2=$O/dev_bpos2-0-100.jsonl" "bpos lam3=$O/dev_bpos3-0-100.jsonl" "lam2=$O/dev_lam2-0-100.jsonl" \
  > $O/table.txt 2>> $O/run.log
log "BELIEF_DEV_DONE ($O/table.txt)"
