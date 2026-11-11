#!/bin/bash
# Resume of the THRONE colour-fix run (vdd4_chain.sh, left running for Gemma on GPUs 3, 4). At 19:07 the Qwen pick failed on a
# parser bug in vdd4_prep.py (fixed 20:28; rule unchanged), so the chain skipped Qwen's 800 images. User 20:4x: use GPUs 2-5.
# 1. Qwen: pick on the existing table, the pick on rows 200-999 on GPU 2 x1 and GPU 5 x2 (memory-gated), judges on GPU 5 or 2,
#    then a 1000-image table and an 800-image table (rows 200-999, not used for selection; request of session h-35) with the
#    baselines re-scored from their existing judge files.
# 2. Gemma: once the chain has picked, adds workers on GPU 2 x1 and GPU 5 x2 to its test queue (units are claimed with mkdir,
#    and every pool stays until all units are complete, so the chain's join still waits for them); after gemma_DONE, the same
#    800-image table for Gemma.
# Log: logs/vdd4.log, VDD4Q_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/vdd4.log; TP=$L/venvs/throne/bin/python
log(){ echo "=== $(date +%T) $*" >> $LOG; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
export BENCH=$L/bench RBD2_SETS=$O/sets HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
pools(){ Q=$1; M=$2; MF=$3; PIDS=()
  for spec in "2 x2" "5 x5a" "5 x5b"; do set -- $spec; bash scripts/rbd2/vdd4_pool.sh $1 ${2}_$M $Q $MF $M & PIDS+=($!); sleep 60; done
  wait "${PIDS[@]}"; }
t800(){ M=$1; BD=$2; OLDT=$3; VTI=$4; P=$5; TR=$O/runs/${M}_cf_test_all
  $TP scripts/rbd2/vdd4_score.py throne_t800 base=$BD/base/combined $P=$TR/eval/$P/combined old_$OLDT=$BD/$OLDT/combined revis=$BD/revis/combined \
      rudder_clone=$BD/rudder_clone/combined grains_clone=$BD/grains_clone/combined $VTI=$BD/$VTI/combined > $O/logs/v4_${M}_t800_table.txt 2>&1
  { echo "=== $(date +%T) $M 800-image table (throne_full1000 rows 200-999, NOT used for selection; 2 judges thr 6; reference = bench Base)"; cat $O/logs/v4_${M}_t800_table.txt; } >> $LOG; }
log "vdd4_qwen_resume.sh start (GPUs 2, 5)"
.venv/bin/python scripts/rbd2/vdd4_prep.py sets >> $LOG 2>&1
{ echo "=== $(date +%T) qwen pick on the existing table (rerun after the parser fix)"; .venv/bin/python scripts/rbd2/vdd4_prep.py pick qwen $O/logs/v4_qwen_t200_table.txt 2>&1; } >> $LOG
P=$(head -1 $O/runs/qwen_cf_pick.txt)
BD=$O/diag/throne_full1000_2judges
if [ -n "$P" ] && [ "$P" != none ]; then
  log "qwen test decode: $P on rows 200-999 (16 units; GPU 2, GPU 5 x2)"; pools $O/queues/v4_qwen_test.txt qwen 19500
  if .venv/bin/python scripts/rbd2/vdd4_prep.py join qwen test $P >> $LOG 2>&1; then
    until G=$(for x in 5 2; do [ $(fre $x) -ge 12000 ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done
    TR=$O/runs/qwen_cf_test_all
    RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 RBD2_JUDGES=base $TP scripts/rbd2/pb_throne_judge.py $(uuid $G) $TR $P > $O/logs/v4judge-q-$P-base.log 2>&1 & JB=$!
    sleep 30
    RBD2_THRONE_SET=throne_full1000 RBD2_N=1000 RBD2_JUDGES=large $TP scripts/rbd2/pb_throne_judge.py $(uuid $G) $TR $P > $O/logs/v4judge-q-$P-large.log 2>&1
    wait $JB
    $TP scripts/rbd2/vdd4_score.py throne_full1000 base=$BD/base/combined $P=$TR/eval/$P/combined old_s16a1bl2p1=$BD/s16a1bl2p1/combined > $O/logs/v4_qwen_test_table.txt 2>&1
    { echo "=== $(date +%T) qwen test table (throne_full1000; setting chosen on rows 0-199 of these; 2 judges thr 6; reference = bench Base; old_s16a1bl2p1 = earlier test config)"; cat $O/logs/v4_qwen_test_table.txt; } >> $LOG
    t800 qwen $BD s16a1bl2p1 vti_author $P
    log "qwen_DONE"
  else log "qwen test incomplete (claims_vdd4/fail-*, logs/v4pool-x*.log)"; fi
else log "qwen: no candidate passed; not continued to 1000"; fi
# Gemma: help with the test units, then the 800-image table
until [ -f $O/runs/gemma_cf_pick.txt ] || grep -q VDD4_DONE $LOG; do sleep 60; done
G4=$(head -1 $O/runs/gemma_cf_pick.txt 2>/dev/null)
if [ -n "$G4" ] && [ "$G4" != none ]; then
  until [ -f $O/queues/v4_gemma_test.txt ]; do sleep 30; done
  log "gemma test: adding workers on GPU 2, GPU 5 x2"; pools $O/queues/v4_gemma_test.txt gemma 21257
  until grep -qE 'gemma_DONE|VDD4_DONE' $LOG; do sleep 120; done
  [ -d $O/runs/gemma_cf_test_all/eval/$G4/combined ] && t800 gemma $O/diag/gemma_throne1000_2judges s24a1bl2p1 vti $G4
fi
log VDD4Q_DONE
