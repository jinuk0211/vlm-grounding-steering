#!/bin/bash
# User 11-10 21:05: Qwen3 is the one backbone where the anchor steering did not help; find why, fix, run. Finding: Qwen3's stored r(t)
# was a byte copy of Qwen2.5's (never measured); its measured curve (rcurve_collect.py, 500 Base captions) matches Qwen2.5 up to
# t~127 and stays higher later (0.154 vs 0.068 at 256-383, 0.080 vs 0.019 at 384+). Every Qwen3 row here uses the measured curve
# (vdd_steer_rc.py; same code otherwise, checked: the stored curve reproduces the earlier A5000 rows). Full = layer 20 resid alpha 1
# + anchornorm layer 20 rho 0.3 K 16, lam 1, pen 0.5, sdpa. GPUs 1, 2 (A5000). Base = stored qchair_test500_all/base.
# usage: q3rc_abl.sh GPU_INDEX WORKER     log outputs/steer-anchor-20261110/qwen3_rc/abl.log (QWEN3_RC_ABL_DONE)
G=$1; W=$2; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-anchor-20261110/qwen3_rc; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl; Q=$R/outputs/rbd2-20261102/runs/qchair_test500_all
export RCURVE_JSON=$R/outputs/steer-qwen3diag-20261110/qwen3-rcurve-measured.json
C=$O/claims; mkdir -p $C; echo "$RCURVE_JSON" > $O/RCURVE.txt
F="--steer resid --layers 20 --alpha 1 --steer2 anchornorm --layers2 20 --alpha2 1 --rho2 0.3 --anchor-steps 16"
log(){ echo "$(date '+%m-%d %T') $W $*" >> $O/abl.log; }
log "worker start GPU $G ($CUDA_VISIBLE_DEVICES) rcurve $RCURVE_JSON"
run_unit(){ T=$1; X=$2; Fl=$O/$T-0-500.jsonl
  if [ -f $Fl ] && [ $(wc -l < $Fl) -ge 500 ]; then return 1; fi
  mkdir $C/$T 2>/dev/null || return 1
  log "start $T $X"
  nice -n 5 .venv/bin/python $S/vdd_steer_rc.py --model qwen3 --data $SET --output $O --attn sdpa --tag $T --start 0 --end 500 $X \
    < /dev/null >> $O/$T-0-500.log 2>&1 && log "done $T" || log "FAILED $T"
  rm -r $C/$T; return 0; }
while true; do
  RAN=0
  for T in anchor_qwen3_rc abl_nosteer_qwen3_rc abl_nobelief_qwen3_rc abl_nopen_qwen3_rc; do
    case $T in anchor_qwen3_rc) X="$F";; abl_nosteer_qwen3_rc) X="--steer none";; abl_nobelief_qwen3_rc) X="$F --lam 0";; abl_nopen_qwen3_rc) X="$F --pen 0";; esac
    if run_unit $T "$X"; then RAN=1; break; fi
  done
  [ $RAN -eq 0 ] && break
done
for T in anchor_qwen3_rc abl_nosteer_qwen3_rc abl_nobelief_qwen3_rc abl_nopen_qwen3_rc; do
  [ -f $O/$T-0-500.jsonl ] && [ $(wc -l < $O/$T-0-500.jsonl) -ge 500 ] || { log "leaves (units left to the other worker)"; exit 0; }; done
mkdir $C/table 2>/dev/null || exit 0
.venv/bin/python $S/steer_table.py chair_test500 $R/data/hallucination-suite/chair-scorer-state.json \
  "Full (anchor)=$O/anchor_qwen3_rc-0-500.jsonl" "- steering=$O/abl_nosteer_qwen3_rc-0-500.jsonl" "- penalty=$O/abl_nopen_qwen3_rc-0-500.jsonl" \
  "- belief=$O/abl_nobelief_qwen3_rc-0-500.jsonl" "Base=$Q/base-0-500.jsonl" "old Full=$Q/s20a1bl1p05-0-500.jsonl" > $O/abl_table.txt 2>> $O/abl.log
log "QWEN3_RC_ABL_DONE ($O/abl_table.txt)"
