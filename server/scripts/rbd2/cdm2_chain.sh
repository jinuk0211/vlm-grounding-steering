#!/bin/bash
# M3ID round 2 (user 11-11): wait until the Qwen ablation on GPU 5 (qbel.sh pids given) has ended, then two workers on GPU 5,
# then score: CHAIR dev 100 per model (cdm_dev_score.py) and Qwen3 CHAIR test 500 (cdm_join.py + score_sets.py). Log logs/cdm2.log (CDM2_DONE).
# usage: cdm2_chain.sh PID...      stop: touch outputs/cdm-20261109/STOP2
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/cdm-20261109; LOG=$O/logs/cdm2.log; cd $R
export PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
       PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src RBD2_SETS=$R/outputs/rbd2-20261102/sets \
       RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json
log(){ echo "=== $(date '+%m-%d %T') $*" >> $LOG; }
alive(){ for p in "$@"; do grep -q qbel /proc/$p/cmdline 2>/dev/null && return 0; done; return 1; }
log "cdm2_chain start, waiting for the Qwen ablation on GPU 5 (pids $*)"
while alive "$@"; do [ -f $O/STOP2 ] && { log CDM2_STOPPED; exit 1; }; sleep 60; done
log "ablation ended; workers w5a w5b on GPU 5"
nohup bash scripts/rbd2/cdm2_pool.sh 5 w5a 20000 > /dev/null 2>&1 < /dev/null &
sleep 90
nohup bash scripts/rbd2/cdm2_pool.sh 5 w5b 20000 > /dev/null 2>&1 < /dev/null &
for M in qwen gemma qwen3; do
  until nice -n 10 .venv/bin/python scripts/rbd2/cdm_dev_score.py $M > $O/logs/.dev_$M.out 2>&1; do
    [ $? -ne 3 ] && { cat $O/logs/.dev_$M.out >> $LOG; log "dev score $M FAILED"; break; }
    [ -f $O/STOP2 ] && { log CDM2_STOPPED; exit 1; }; sleep 120; done
  grep -v 'it/s' $O/logs/.dev_$M.out >> $LOG; echo "CDM2_dev_${M}_DONE" >> $LOG
done
until nice -n 10 .venv/bin/python scripts/rbd2/cdm_join.py qwen3 >> $LOG 2>&1; do
  [ -f $O/STOP2 ] && { log CDM2_STOPPED; exit 1; }; sleep 120; done
{ log "qwen3 CHAIR 500 vs vdd Base (rbd = table row)"
  RBD2_N=500 nice -n 10 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $O/score_qwen3 2>&1 | grep -v 'it/s'
  echo CDM2_qwen3_test_DONE; } >> $LOG
log CDM2_DONE
