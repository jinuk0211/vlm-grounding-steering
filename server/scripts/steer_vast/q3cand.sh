#!/bin/bash
# User 11-10 21:45 "test 100장 후보써서 가보자": Qwen3 steering candidates on CHAIR test rows 0-99 (selection on test rows: to be stated
# in the paper, as the earlier Qwen3 pick on rows 0-60). All with Qwen3's measured r(t) (vdd_steer_rc.py), lam 1, pen 0.5, sdpa, A5000.
#   cand_L12/L16/L24/L28 : resid alpha 1 + anchornorm rho 0.3 K 16 at that layer
#   cand_anc20           : anchornorm only at layer 20 (no alpha d_t term; the paper's Eq. steer)
#   anchor_qwen3_rc (L20 current Full) and abl_nosteer_qwen3_rc (-steering): resumed from the stopped 0-500 runs' first rows
# Pick (told the user): largest CHAIR_s drop vs -steering with CHAIR_i not higher and recall within 1 point; none -> Qwen3 has no
# steering effect.     usage: q3cand.sh GPU_INDEX WORKER      log outputs/steer-anchor-20261110/qwen3_cand/cand.log (Q3CAND_DONE)
G=$1; W=$2; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-anchor-20261110/qwen3_cand; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl; Q=$R/outputs/rbd2-20261102/runs/qchair_test500_all
export RCURVE_JSON=$R/outputs/steer-qwen3diag-20261110/qwen3-rcurve-measured.json
C=$O/claims; mkdir -p $C
an(){ echo "--steer2 anchornorm --layers2 $1 --alpha2 1 --rho2 0.3 --anchor-steps 16"; }
args(){ case $1 in
  anchor_qwen3_rc) echo "--steer resid --layers 20 --alpha 1 $(an 20)";;
  abl_nosteer_qwen3_rc) echo "--steer none";;
  cand_L12) echo "--steer resid --layers 12 --alpha 1 $(an 12)";; cand_L16) echo "--steer resid --layers 16 --alpha 1 $(an 16)";;
  cand_L24) echo "--steer resid --layers 24 --alpha 1 $(an 24)";; cand_L28) echo "--steer resid --layers 28 --alpha 1 $(an 28)";;
  cand_anc20) echo "--steer anchornorm --layers 20 --alpha 1 --rho 0.3 --anchor-steps 16";; esac; }
TAGS="anchor_qwen3_rc abl_nosteer_qwen3_rc cand_L16 cand_L24 cand_anc20 cand_L12 cand_L28"
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/cand.log; }
log "worker start GPU $G ($CUDA_VISIBLE_DEVICES) rcurve $RCURVE_JSON"
while true; do
  RAN=0
  for T in $TAGS; do
    F=$O/$T-0-100.jsonl
    if [ -f $F ] && [ $(wc -l < $F) -ge 100 ]; then continue; fi
    mkdir $C/$T 2>/dev/null || continue
    X=$(args $T); log "start $T $X"
    nice -n 5 .venv/bin/python $S/vdd_steer_rc.py --model qwen3 --data $SET --output $O --attn sdpa --tag $T --start 0 --end 100 $X \
      < /dev/null >> $O/$T-0-100.log 2>&1 && log "done $T" || log "FAILED $T"
    rm -r $C/$T; RAN=1; break
  done
  [ $RAN -eq 0 ] && break
done
for T in $TAGS; do [ -f $O/$T-0-100.jsonl ] && [ $(wc -l < $O/$T-0-100.jsonl) -ge 100 ] || { log "leaves"; exit 0; }; done
mkdir $C/table 2>/dev/null || exit 0
A=(); for T in abl_nosteer_qwen3_rc anchor_qwen3_rc cand_anc20 cand_L12 cand_L16 cand_L24 cand_L28; do A+=("$T=$O/$T-0-100.jsonl"); done
.venv/bin/python $S/interim.py chair_test500 $R/data/hallucination-suite/chair-scorer-state.json "${A[@]}" "Base=$Q/base-0-500.jsonl" \
  "old Full=$Q/s20a1bl1p05-0-500.jsonl" > $O/cand_table.txt 2>> $O/cand.log
log "Q3CAND_DONE ($O/cand_table.txt)"
