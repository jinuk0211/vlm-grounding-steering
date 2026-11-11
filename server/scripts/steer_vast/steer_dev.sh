#!/bin/bash
# User 11-10 ~19:50: "한번에 실수없이, hal 줄이고 recall 손실 없이" -> the belief setting is chosen on chair_dev100 (train2014 dev) before the
# single test run. Gemma 4 E4B / Qwen3-VL-8B: each model's existing steering (layer, alpha, attn of s24a1bl1p05 / s20a1bl1p05) + the
# anchornorm term of the Qwen2.5 pick (rho 0.3, K 16, same layer); Base rerun on the same GPU so every dev row shares the hardware.
# usage: steer_dev.sh MODEL GPU_INDEX VARIANT ...   (VARIANT in base nosteer full nobelief bpos1 bpos2 bpos3 lam2 nosteer_bpos1
#        nosteer_bpos2 bpos2_r45 full_r45; skips finished ones; OUTSUB env = output subdirectory, default MODEL)
# 21:30 user: steering must be the main effect; belief = recall defence, with an effect smaller than steering's.
# -> outputs/steer-dev-20261110/OUTSUB/dev_VARIANT-0-100.jsonl, table.txt (steer_table.py), run.log (STEER_DEV_DONE)
M=$1; G=$2; shift 2; source $(dirname $0)/steer_env.sh
O=$R/outputs/steer-dev-20261110/${OUTSUB:-$M}; SET=$R/outputs/rbd2-20261102/sets/chair_dev100.jsonl; mkdir -p $O
case $M in
  gemma) LY=24; AL=1; AT=eager;;
  qwen3) LY=20; AL=1; AT=sdpa;;
  qwen)  LY=16; AL=2; AT=sdpa;;
  *) echo "bad model $M"; exit 1;;
esac
F="--layers $LY --alpha $AL --layers2 $LY $NEW"
log(){ echo "$(date '+%m-%d %T') $*" >> $O/run.log; }
log "start $M GPU $G ($CUDA_VISIBLE_DEVICES) variants: $*"
for V in "$@"; do
  case $V in
    base) X="--steer none --lam 0 --pen 0";; nosteer) X="--steer none";; full) X="$F";; nobelief) X="$F --lam 0";;
    bpos1) X="$F --bpos";; bpos2) X="$F --bpos --lam 2";; bpos3) X="$F --bpos --lam 3";; lam2) X="$F --lam 2";;
    nosteer_bpos1) X="--steer none --bpos";; nosteer_bpos2) X="--steer none --bpos --lam 2";;
    bpos2_r45) X="$F --bpos --lam 2 --rho2 0.45";; full_r45) X="$F --rho2 0.45";;
    *) log "unknown variant $V"; continue;;
  esac
  T=dev_$V
  if [ -f $O/$T-0-100.jsonl ] && [ $(wc -l < $O/$T-0-100.jsonl) -ge 100 ]; then continue; fi
  log "start $T $X"
  nice -n 5 .venv/bin/python $S/vdd_steer.py --model $M --data $SET --output $O --tag $T --start 0 --end 100 --attn $AT $X < /dev/null >> $O/$T-0-100.log 2>&1 \
    && log "done $T" || log "FAILED $T"
done
A=(); for V in base nosteer full nobelief bpos1 bpos2 bpos3 lam2 nosteer_bpos1 nosteer_bpos2 bpos2_r45 full_r45; do [ -f $O/dev_$V-0-100.jsonl ] && A+=("$V=$O/dev_$V-0-100.jsonl"); done
.venv/bin/python $S/steer_table.py chair_dev100 $R/data/chair-dev-train2014-100/chair-scorer-state.json "${A[@]}" > $O/table.txt 2>> $O/run.log
log "STEER_DEV_DONE ($O/table.txt)"
