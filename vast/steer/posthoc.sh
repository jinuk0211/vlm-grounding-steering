#!/bin/bash
# Post-hoc confirmation of an16_r05 on chair_test500 rows 100-499 (PROTOCOL.md, "Post-hoc confirmation"); starts after screen2.sh exits
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering
while pgrep -f screen2.sh > /dev/null; do sleep 60; done
echo "$(date +%T) screen2 finished; post-hoc confirmation: nosteer, an16_r05, full on rows 100-499"
{ grep -E "^nosteer " queue_qwen100.txt; grep -E "^an16_r05 " queue_qwen100_r2.txt; grep -E "^full " queue_qwen100.txt; } > queue_posthoc.txt
cat queue_posthoc.txt
./run_queue.sh queue_posthoc.txt sets/chair_test500.jsonl runs/confirm400 100 500
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json RBD2_N=500
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
C=runs/confirm400
venv/bin/python steer_score.py chair_test500 $C 100 500 > $C/score_posthoc.txt 2> logs/score_posthoc.err
venv/bin/python lenctl.py chair_test500 100 500 nosteer=$C/nosteer-100-500.jsonl full=$C/full-100-500.jsonl \
  an16_r05=$C/an16_r05-100-500.jsonl > $C/lenctl_posthoc.txt 2> logs/lenctl_posthoc.err
echo "$(date +%T) POSTHOC_DONE"
