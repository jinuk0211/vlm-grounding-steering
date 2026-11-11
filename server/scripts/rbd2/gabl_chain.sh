#!/bin/bash
# Gemma 4 E4B CHAIR ablation of the paper's RBD row s24a1bl1p05 (user 11-05 20:0x, after the Qwen2.5 table in logs/abl_chair.log; design in gabl_prep.py).
# Starts when HAT + ALOHa (HAT_GPU3_DONE in logs/hat_gpu3.log) and the Qwen3 baselines (file gabl_qwen3_done, written by session "baseline 실험") have
# ended. No process or memory check decides the start: a GPU that looks free is not therefore free. `touch outputs/rbd2-20261102/gabl_go` starts it at once.
# HAT's main pid gone without that line is an abnormal end (h-17): logged, not counted as done, the chain keeps waiting.
#  0 at scheduling: gabl_prep.py check (reused rows complete, settings as they should be), so a wrong path shows now and not after the wait
#  1 the full setting on rows 0-5 vs the stored full row; none of the 5 identical stops here, before the 3440 rows are spent
#  2 workers on GPUs 2, 3, 4 x2, 5 x2, memory-gated (vdd4_pool.sh, Gemma gate as vdd4_chain.sh; KJH's small jobs on GPUs 2 and 4)
#  3 CHAIR scorer of the paper table: vs vdd Base and vs the full row; LaTeX in logs/gabl_chair_table.tex
# Restart after a kill: rm -r claims_vdd4/gabl_* (finished units are kept; a unit with partial rows resumes).
# Everything stays under LJW (TMPDIR, XDG_CACHE_HOME); CPU steps run at nice 10 with 8 threads. Log: logs/gabl_chair.log, GABL_DONE / GABL_STOPPED.
L=${GABL_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gabl_chair.log; PY=.venv/bin/python; MF=21257; HP=${HAT_MAIN_PID:-1258987}; WARNED=
export TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache PYTHONDONTWRITEBYTECODE=1; mkdir -p $TMPDIR $O/logs $O/queues
CPU="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "GABL_STOPPED: $*"; exit 1; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
go(){ [ -f $O/gabl_go ] || { grep -q HAT_GPU3_DONE $O/logs/hat_gpu3.log 2>/dev/null && [ -f $O/gabl_qwen3_done ]; }; }
log "gabl_chain.sh start; waiting for HAT_GPU3_DONE and gabl_qwen3_done"
$CPU $PY scripts/rbd2/gabl_prep.py check >> $LOG 2>&1 || stop "check failed"
until go; do
  if [ -z "$WARNED" ] && ! kill -0 $HP 2>/dev/null && ! grep -q HAT_GPU3_DONE $O/logs/hat_gpu3.log 2>/dev/null; then
    log "HAT main pid $HP gone without HAT_GPU3_DONE: abnormal end, not counted as done; still waiting (gabl_go starts the chain)"; WARNED=1; fi
  sleep 120; done
log "start conditions met"
$CPU $PY scripts/rbd2/gabl_prep.py queue >> $LOG 2>&1 || stop "queue failed"
until G=$(for x in 5 4 3 2; do [ $(fre $x) -ge $MF ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done
log "reproduction check on GPU $G: full setting, rows 0-5"
bash scripts/rbd2/vdd4_pool.sh $G garepro $O/queues/gabl_repro.txt $MF gemma
$CPU $PY scripts/rbd2/gabl_prep.py repro >> $LOG 2>&1 || stop "reproduction check failed (logs/v4-gabl_repro-*.log, claims_vdd4/fail-*)"
log "decode: 7 variants, 3440 rows"
PIDS=(); for spec in "2 ga2" "3 ga3" "4 ga4a" "4 ga4b" "5 ga5a" "5 ga5b"; do set -- $spec
  bash scripts/rbd2/vdd4_pool.sh $1 $2 $O/queues/gabl_chair.txt $MF gemma & PIDS+=($!); sleep 60; done
wait "${PIDS[@]}"
$CPU $PY scripts/rbd2/gabl_prep.py join >> $LOG 2>&1 || stop "decode incomplete (claims_vdd4/fail-*, logs/v4pool-ga*.log)"
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data RBD2_N=500
{ echo "=== $(date +%T) CHAIR 500, Gemma ablation; paired bootstrap vs vdd Base (base)"
  PYTHONPATH=scripts:src $CPU $PY scripts/rbd2/score_sets.py chair_test500 $O/runs/gabl_chair500_all 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) CHAIR 500, Gemma ablation; paired bootstrap vs the full RBD row (base = rbd full; vddbase, tblbase)"
  PYTHONPATH=scripts:src $CPU $PY scripts/rbd2/score_sets.py chair_test500 $O/runs/gabl_chair500_vsfull 2>&1 | grep -v 'it/s'; } >> $LOG
[ -s $O/runs/gabl_chair500_all/score.json ] || stop "scoring vs vdd Base failed"
$CPU $PY scripts/rbd2/gabl_prep.py table $O/runs/gabl_chair500_vsfull > $O/logs/gabl_chair_table.tex 2>> $LOG || stop "table failed (scoring vs the full row)"
log GABL_DONE
