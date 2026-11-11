#!/bin/bash
# 13:00 reorder (user): pilot first, ablations last.
R=/home/project-b/LJW/vlm-grounding-steering; O=/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102
if [ "$1" = 2 ]; then
  while kill -0 1801412 2>/dev/null || kill -0 1800702 2>/dev/null; do sleep 15; done
  { echo "=== $(date +%T) amber_test60L"; bash $R/scripts/rbd2/pb_score.sh amber_test60L 2>&1 | grep -v 'it/s' | tail -4
    echo "=== $(date +%T) throne_test60"; bash $R/scripts/rbd2/pb_score.sh throne_test60 2 2>&1 | grep -v 'it/s' | tail -6
    echo "=== $(date +%T) PILOT_SCORED"; } >> $O/logs/pilot_score.log
  bash $R/scripts/rbd2/pb_worker.sh 2 $O/queues/sweep_q0.txt sweep
  bash $R/scripts/rbd2/pb_worker.sh 2 $O/queues/q0d.txt test
else
  while kill -0 1800702 2>/dev/null; do sleep 15; done
  bash $R/scripts/rbd2/pb_worker.sh 3 $O/queues/sweep_q1.txt sweep
  bash $R/scripts/rbd2/pb_worker.sh 3 $O/queues/q1d.txt test
fi
