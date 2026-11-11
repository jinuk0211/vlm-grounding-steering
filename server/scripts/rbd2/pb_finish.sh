#!/bin/bash
# After the sweep queues: GPU g judges THRONE for its own half of the configs as soon as its decoding queue ends;
# when both halves are judged, score every set (chair sweep, amber sweep, throne) into logs/final_score.log.
G=$1; CHAIN=$2; TAGS=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
while kill -0 $CHAIN 2>/dev/null; do sleep 30; done
U=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
echo "$(date +%T) gpu$G judging: $TAGS" >> $O/logs/throne_judge$G.log
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $O/runs/throne_test60 $TAGS >> $O/logs/throne_judge$G.log 2>&1 || echo FAILED >> $O/logs/throne_judge$G.log
echo "$(date +%T) JUDGE_DONE" >> $O/logs/throne_judge$G.log
if [ "$G" = 2 ]; then
  while ! grep -q JUDGE_DONE $O/logs/throne_judge3.log 2>/dev/null; do sleep 30; done
  { for s in chair_test60_sweep amber_test60 throne_test60; do echo "=== $(date +%T) $s"; bash scripts/rbd2/pb_score.sh $s 2 2>&1 | grep -v 'it/s' | tail -16; done
    echo "=== $(date +%T) FINAL_SCORED"; } > $O/logs/final_score.log 2>&1
fi
