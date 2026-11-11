#!/bin/bash
# RBD v2.1 THRONE (h-35 / user, 11-04): colour-fixed V_obj (orange etc. removed; vdd4_setup.py). User 16:5x: the setting is
# chosen on the FIRST 200 TEST IMAGES (throne_full1000 rows 0-199) and only then run on the other 800 (the paper states
# this); Qwen first, then Gemma (layer 24). Starts after the AMBER vdd3 chain (user's order). GPUs 3 and 4 only (user):
# workers 3 x1, 4 x2, memory-gated; judges (flan-t5 base+large, threshold 6) on 3 or 4 with room.
# Selection rows on the 200: bench Base and the old-V_obj config (existing judge files of the 1000-image runs), vdd Base
# (Qwen only, existing), and the 4 colour-fixed candidates (new). Guard + pick: vdd4_prep.py pick (a candidate must beat
# bench Base AND the old config on F0.5_CLS, else that backbone stops there and the next stage goes on).
# Test: the pick on rows 200-999, joined with its rows 0-199, judged on the 1000, scored vs bench Base (ALL, CLS, CIs,
# F0.5_ALL without orange), with the old config as an extra row. Log: logs/vdd4.log, VDD4_DONE / VDD4_STOPPED.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/vdd4.log
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "VDD4_STOPPED: $*"; exit 1; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
export BENCH=$L/bench RBD2_SETS=$O/sets HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
TP=$L/venvs/throne/bin/python
log "vdd4_chain.sh start (selection on the first 200 test images); waiting for VDD3_DONE / VDD3_STOPPED"
until grep -qE 'VDD3_DONE|VDD3_STOPPED' $O/logs/vdd3.log 2>/dev/null; do sleep 120; done
log "vdd3 finished; starting"
.venv/bin/python scripts/rbd2/vdd4_setup.py >> $LOG 2>&1 || stop "setup failed"
.venv/bin/python scripts/rbd2/vdd4_prep.py sets >> $LOG 2>&1 || stop "sets failed"
phase(){ Q=$1; M=$2; MF=$3; PIDS=()
  for spec in "3 w3" "4 w4a" "4 w4b"; do set -- $spec; bash scripts/rbd2/vdd4_pool.sh $1 ${2}_$M $Q $MF $M & PIDS+=($!); sleep 60; done
  wait "${PIDS[@]}"; }
judge_group(){ RD=$1; SET=$2; N=$3; shift 3
  until G=$(for x in 3 4; do [ $(fre $x) -ge 12000 ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done
  for t in "$@"; do
    RBD2_THRONE_SET=$SET RBD2_N=$N RBD2_JUDGES=base $TP scripts/rbd2/pb_throne_judge.py $(uuid $G) $RD $t > $O/logs/v4judge-$t-base.log 2>&1 & JB=$!
    sleep 30
    RBD2_THRONE_SET=$SET RBD2_N=$N RBD2_JUDGES=large $TP scripts/rbd2/pb_throne_judge.py $(uuid $G) $RD $t > $O/logs/v4judge-$t-large.log 2>&1
    wait $JB
  done; }
judge_all(){ RD=$1; SET=$2; N=$3; shift 3; T=("$@"); h=$(( (${#T[@]}+1)/2 )); J2=
  judge_group $RD $SET $N "${T[@]:0:$h}" & J1=$!
  if [ ${#T[@]} -gt $h ]; then sleep 90; judge_group $RD $SET $N "${T[@]:$h}" & J2=$!; fi
  wait $J1 $J2; }
for M in qwen gemma; do
  if [ $M = qwen ]; then MF=19500; BD=$O/diag/throne_full1000_2judges; OLDT=s16a1bl2p1; else MF=21257; BD=$O/diag/gemma_throne1000_2judges; OLDT=s24a1bl2p1; fi
  CT="cf_a1bl2p1 cf_a2bl2p1 cf_a3bl2p1 cf_a1bl3p1"
  .venv/bin/python scripts/rbd2/vdd4_prep.py queue_sel $M >> $LOG 2>&1 || stop "$M queue failed"
  log "$M selection decode on the first 200 test images (16 units)"; phase $O/queues/v4_${M}_sel.txt $M $MF
  .venv/bin/python scripts/rbd2/vdd4_prep.py join $M sel $CT >> $LOG 2>&1 || { log "$M selection incomplete (claims_vdd4/fail-*, logs/v4pool-*.log); skipping $M"; continue; }
  SR=$O/runs/${M}_cf_t200_all; judge_all $SR throne_t200 200 $CT
  SPEC=(base=$BD/base/combined old_a1bl2p1=$BD/$OLDT/combined)
  [ $M = qwen ] && SPEC+=(vddbase=$O/runs/throne_test500_all/eval/base/combined)
  for t in $CT; do SPEC+=("$t=$SR/eval/$t/combined"); done
  $TP scripts/rbd2/vdd4_score.py throne_t200 "${SPEC[@]}" > $O/logs/v4_${M}_t200_table.txt 2>&1 || { log "$M selection scoring failed (logs/v4_${M}_t200_table.txt); skipping $M"; continue; }
  { echo "=== $(date +%T) $M selection table (first 200 test images, 2 judges thr 6; reference = bench Base)"; cat $O/logs/v4_${M}_t200_table.txt; } >> $LOG
  .venv/bin/python scripts/rbd2/vdd4_prep.py pick $M $O/logs/v4_${M}_t200_table.txt >> $LOG 2>&1 || { log "$M pick failed; skipping $M"; continue; }
  PICK=$(head -1 $O/runs/${M}_cf_pick.txt)
  if [ "$PICK" = none ]; then log "${M}: no colour-fixed candidate beats bench Base and the old config on F0.5_CLS (200); NOT continued to 1000"; continue; fi
  log "$M test decode: $PICK on throne_full1000 rows 200-999 (16 units)"; phase $O/queues/v4_${M}_test.txt $M $MF
  .venv/bin/python scripts/rbd2/vdd4_prep.py join $M test $PICK >> $LOG 2>&1 || { log "$M test incomplete; skipping"; continue; }
  TR=$O/runs/${M}_cf_test_all; judge_all $TR throne_full1000 1000 $PICK
  $TP scripts/rbd2/vdd4_score.py throne_full1000 base=$BD/base/combined $PICK=$TR/eval/$PICK/combined old_$OLDT=$BD/$OLDT/combined > $O/logs/v4_${M}_test_table.txt 2>&1 || { log "$M test scoring failed"; continue; }
  { echo "=== $(date +%T) $M test table (throne_full1000; setting chosen on rows 0-199 of these; 2 judges thr 6; reference = bench Base; old_$OLDT = earlier test config)"; cat $O/logs/v4_${M}_test_table.txt; } >> $LOG
  log "${M}_DONE"
done
log VDD4_DONE
