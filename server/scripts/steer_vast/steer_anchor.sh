#!/bin/bash
# User 11-10 ~19:40: try the anchor steering on Gemma 4 E4B and Qwen3-VL-8B, CHAIR test 500. No tuning: each model's existing Full
# (vdd_decode.py s24a1bl1p05 / s20a1bl1p05: layer, alpha, lam 1, pen 0.5, attn) + the anchornorm term of the Qwen2.5 pick
# (rho 0.3, K 16, same layer). 1) port check: vdd_steer.py --steer resid alone on rows 0-1 vs the stored Full rows;
# 2) 500 rows (vdd_steer.py resumes a partial file; it asserts that the steering hooks fire).
# usage: steer_anchor.sh MODEL GPU_INDEX      log outputs/steer-anchor-20261110/MODEL/run.log (ANCHOR_DONE)
M=$1; G=$2; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-anchor-20261110/$M; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl; T=$R/outputs/rbd2-20261102/runs; mkdir -p $O
case $M in
  gemma) LY=24; AL=1; AT=eager; OLD=$T/gchair_test500_all/s24a1bl1p05-0-500.jsonl;;
  qwen3) LY=20; AL=1; AT=sdpa;  OLD=$T/qchair_test500_all/s20a1bl1p05-0-500.jsonl;;
  *) echo "bad model $M"; exit 1;;
esac
A="--model $M --data $SET --output $O --attn $AT --layers $LY --alpha $AL --lam 1 --pen 0.5"
log(){ echo "$(date '+%m-%d %T') $*" >> $O/run.log; }
log "start $M GPU $G ($CUDA_VISIBLE_DEVICES) layer $LY alpha $AL attn $AT"
nice -n 5 .venv/bin/python $S/vdd_steer.py $A --tag port_resid --start 0 --end 2 --steer resid < /dev/null >> $O/port_resid-0-2.log 2>&1 \
  || { log "FAILED port check run"; exit 1; }
.venv/bin/python $S/port_check.py $O/port_resid-0-2.jsonl $OLD >> $O/run.log 2>&1
nice -n 5 .venv/bin/python $S/vdd_steer.py $A --tag anchor_$M --start 0 --end 500 $NEW --layers2 $LY < /dev/null >> $O/anchor_$M-0-500.log 2>&1 \
  && log "ANCHOR_DONE ($O/anchor_$M-0-500.jsonl)" || log "FAILED anchor_$M"
