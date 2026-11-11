#!/bin/bash
# User 11-08: when the HAT v7 chain (hat_v7_chain.sh) has ended, on GPUs 0 and 1 only:
#  1 REVIS on CHAIR dev 100, the implementation of each backbone's CHAIR row: own (baselines.py) on Qwen2.5 (sdpa) and Gemma, official code on
#    Qwen3 (qwen3_bl.py revis_official, sdpa); then the dev tables with REVIS (cdev_prep.py score) -> logs/cdev_table_MODEL.tex, logs/cdev.log
#  2 the flan-t5-xl judgments the THRONE table's xl columns lack (Qwen RBD first and final configuration, the 7 Gemma rows; xl_judge.py), then
#    xl_score.py -> logs/throne_xl_table.txt
#  3 the rest of the Gemma CHAIR ablation (gabl_chain.sh, stopped by the 11-06 outage: claims of its dead workers removed, finished units kept,
#    partial ones resume), then join, scoring and LaTeX table exactly as in gabl_chain.sh -> logs/gabl_chair.log GABL_DONE, logs/gabl_chair_table.tex
# Three slots, GPU 0 x2 (A6000) and GPU 1 (A5000). Each runs the REVIS units, then xl judgments, then ablation units, all claim-based and memory-gated,
# so no slot idles while another phase still has work. Log: logs/after_v7.log, AFTER_V7_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
V7=${V7_PID:-1472680}; LOG=$O/logs/after_v7.log; PY=.venv/bin/python; TPY=$L/venvs/throne/bin/python
export TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src; mkdir -p $TMPDIR
CPU="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="
log(){ echo "=== $(date +%T) $*" >> $LOG; }
QR=$O/queues/chair_dev100_revis.txt; QX=$O/queues/throne_xl.txt; QA=$O/queues/gabl_chair.txt
log "after_v7.sh start; waiting for hat_v7_chain.sh ($V7)"
while kill -0 $V7 2>/dev/null; do sleep 60; done
log "v7 chain ended: $(grep -oE 'V7_DONE.*|V7_STOPPED.*' $O/logs/hat_v7.log | tail -1)"
printf '%s\n' 'gemma bench revis 0 60' 'gemma bench revis 60 100' 'qwen3 grains revis_official 0 60 --attn sdpa' 'qwen3 grains revis_official 60 100 --attn sdpa' \
              'qwen bench revis 0 60 --attn sdpa' 'qwen bench revis 60 100 --attn sdpa' > $QR
printf '%s\n' 'gemma_throne1000 base' 'gemma_throne1000 revis' 'gemma_throne1000 rudder_clone' 'gemma_throne1000 grains_clone' 'gemma_throne1000 vti' \
              'gemma_throne1000 s24a1bl2p1' 'gemma_cf_test_all cf_a3bl2p1' 'throne_full1000 s16a1bl2p1' 'qwen_cf_test_all cf_a3bl2p1' > $QX
rm -rf $O/claims_vdd4/gabl_chair500-*   # left by the workers the 11-06 outage killed (no gabl process runs)
rm -f $O/after_v7.*_left
slot(){ G=$1; W=$2
  HAT_RUN=chair_dev100 bash scripts/rbd2/hat_pool.sh $G $W $QR; echo $W >> $O/after_v7.revis_left
  bash scripts/rbd2/xl_pool.sh $G $W $QX; echo $W >> $O/after_v7.xl_left
  bash scripts/rbd2/vdd4_pool.sh $G $W $QA 21257 gemma; echo $W >> $O/after_v7.abl_left; }
n(){ cat $O/after_v7.$1_left 2>/dev/null | wc -l; }
PIDS=(); for spec in "0 av0a" "0 av0b" "1 av1"; do set -- $spec; slot $1 $2 & PIDS+=($!); sleep 60; done
log "slots on GPU 0 x2 and GPU 1 started"
# hat_pool.sh leaves only when no REVIS unit is open, so the first slot out means every REVIS unit is complete or failed twice
until [ $(n revis) -ge 1 ]; do sleep 60; done
log "REVIS dev decoded; dev tables with REVIS"
for M in qwen qwen3 gemma; do $CPU $PY scripts/rbd2/cdev_prep.py score $M >> $O/logs/cdev.log 2>&1 && log "$M: logs/cdev_table_$M.tex" || log "$M dev scoring failed (logs/cdev.log)"; done
# xl_pool.sh leaves when nothing is left to claim, so all xl judgments have ended once every slot is out of that phase
until [ $(n xl) -ge 3 ]; do sleep 60; done
log "xl judgments ended: $(ls $O/runs/*/eval/*/combined/*flan-t5-xl.json | wc -l) xl files; $(ls $O/claims_xl | grep -c fail-) failed units"
BENCH=$L/bench $CPU $TPY scripts/rbd2/xl_score.py > $O/logs/throne_xl_table.txt 2>> $LOG && log "xl table: logs/throne_xl_table.txt" || log "xl scoring failed"
wait "${PIDS[@]}"
GL=$O/logs/gabl_chair.log; glog(){ echo "=== $(date +%T) $*" >> $GL; }
glog "decode (resumed by after_v7.sh on GPUs 0, 1 after the 11-06 outage) ended"
if $CPU $PY scripts/rbd2/gabl_prep.py join >> $GL 2>&1; then
  export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json RBD2_N=500
  { echo "=== $(date +%T) CHAIR 500, Gemma ablation; paired bootstrap vs vdd Base (base)"
    PYTHONPATH=scripts:src $CPU $PY scripts/rbd2/score_sets.py chair_test500 $O/runs/gabl_chair500_all 2>&1 | grep -v 'it/s'
    echo "=== $(date +%T) CHAIR 500, Gemma ablation; paired bootstrap vs the full RBD row (base = rbd full; vddbase, tblbase)"
    PYTHONPATH=scripts:src $CPU $PY scripts/rbd2/score_sets.py chair_test500 $O/runs/gabl_chair500_vsfull 2>&1 | grep -v 'it/s'; } >> $GL
  if [ -s $O/runs/gabl_chair500_all/score.json ] && $CPU $PY scripts/rbd2/gabl_prep.py table $O/runs/gabl_chair500_vsfull > $O/logs/gabl_chair_table.tex 2>> $GL
  then glog GABL_DONE; log "ablation: logs/gabl_chair_table.tex"; else glog "GABL_STOPPED: scoring or table failed"; log "ablation scoring failed"; fi
else glog "GABL_STOPPED: decode incomplete (claims_vdd4/fail-*, logs/v4pool-av*.log)"; log "ablation decode incomplete"; fi
log AFTER_V7_DONE
