#!/bin/bash
# Qwen2.5-VL CHAIR ablation of the paper's RBD row (user 11-04 16:3x: "after all of these"; design in abl_prep.py).
# Starts when every queued job has ended: logit lens (ALL_DONE), AMBER vdd3 (VDD3_DONE/STOPPED), THRONE colour fix
# (VDD4_DONE/STOPPED). Workers on GPUs 2, 3, 4 (x2), 5, memory-gated (vdd4_pool.sh, Qwen; vdd4_decode.py without --vobj
# = vdd_decode.py with the old V_obj). Scored with the CHAIR scorer of the paper table: vs vdd Base and vs the full RBD row.
# Log: logs/abl_chair.log, ABL_DONE / ABL_STOPPED.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/abl_chair.log
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "ABL_STOPPED: $*"; exit 1; }
log "abl_chain.sh start; waiting for ALL_DONE (logit lens), VDD3_DONE/STOPPED, VDD4_DONE/STOPPED"
until grep -q ALL_DONE $R/outputs/logit-lens-20261104/logs/run.log 2>/dev/null && grep -qE 'VDD3_DONE|VDD3_STOPPED' $O/logs/vdd3.log 2>/dev/null \
      && grep -qE 'VDD4_DONE|VDD4_STOPPED' $O/logs/vdd4.log 2>/dev/null; do sleep 120; done
log "start conditions met"
.venv/bin/python scripts/rbd2/abl_prep.py queue >> $LOG 2>&1 || stop "queue failed"
PIDS=(); for spec in "2 a2" "3 a3" "4 a4a" "4 a4b" "5 a5"; do set -- $spec
  bash scripts/rbd2/vdd4_pool.sh $1 $2 $O/queues/abl_chair.txt 19500 qwen & PIDS+=($!); sleep 60; done
wait "${PIDS[@]}"
.venv/bin/python scripts/rbd2/abl_prep.py join >> $LOG 2>&1 || stop "decode incomplete (claims_vdd4/fail-*, logs/v4pool-a*.log)"
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 RBD2_N=500
{ echo "=== $(date +%T) CHAIR 500, ablation; paired bootstrap vs vdd Base (base)"
  PYTHONPATH=scripts:src .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $O/runs/chair_abl500_all 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) CHAIR 500, ablation; paired bootstrap vs the full RBD row (base = rbd_full)"
  PYTHONPATH=scripts:src .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $O/runs/chair_abl500_vsfull 2>&1 | grep -v 'it/s'; } >> $LOG
log ABL_DONE
