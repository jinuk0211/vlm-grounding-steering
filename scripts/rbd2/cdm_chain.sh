#!/bin/bash
# VCD + M3ID on CHAIR 500 (chair_test500), Qwen2.5-VL-7B -> Gemma 4 E4B -> Qwen3-VL-8B, GPUs 1 and 2 only.
# usage: cdm_chain.sh "GPU:WORKER:GATE ..."   e.g. "1:g1a:22000 1:g1b:22000 2:g2:20000"
# Log $O/logs/cdm.log: CDM_<model>_DONE per model, CDM_DONE at the end. Stop: touch $O/STOP (workers finish their unit).
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/cdm-20261109; LOG=$O/logs/cdm.log; cd $R
mkdir -p $O/logs $O/runs $O/claims
export PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp \
       PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src RBD2_SETS=$R/outputs/rbd2-20261102/sets \
       RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json RBD2_N=500 OMP_NUM_THREADS=4
MODELS="qwen gemma qwen3"
if [ ! -f $O/queue.txt ]; then
  { for M in $MODELS; do echo "# $M"; for s in $(seq 0 50 450); do for X in vcd m3id; do echo "$M $X $s $((s+50))"; done; done; done; } > $O/queue.txt
fi
echo "=== $(date '+%m-%d %T') cdm_chain start, workers: $1" >> $LOG
for w in $1; do
  IFS=: read G W MF <<< "$w"
  nohup bash scripts/rbd2/cdm_pool.sh $G $W $MF > /dev/null 2>&1 &
  echo "$(date +%T) worker $W GPU $G pid $!" >> $LOG; sleep 90
done
for M in $MODELS; do
  until nice -n 10 .venv/bin/python scripts/rbd2/cdm_join.py $M >> $LOG 2>&1; do
    [ -f $O/STOP ] && { echo "=== $(date +%T) CDM_STOPPED" >> $LOG; exit 1; }; sleep 120; done
  { echo "=== $(date '+%m-%d %T') $M CHAIR 500 vs vdd Base (rbd = table row)"
    nice -n 10 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $O/score_$M 2>&1 | grep -v 'it/s'
    echo "CDM_${M}_DONE"; } >> $LOG
done
echo "=== $(date '+%m-%d %T') CDM_DONE" >> $LOG
