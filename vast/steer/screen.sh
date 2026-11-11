#!/bin/bash
# Screen: 12 conditions x chair_test100, then score with the rule fixed in PROTOCOL.md
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering
./run_queue.sh queue_qwen100.txt sets/chair_test100.jsonl runs/steer100 0 100
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json RBD2_N=100
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
venv/bin/python steer_score.py chair_test100 runs/steer100 0 100 > runs/steer100/score.txt 2> logs/score.err
echo "$(date +%T) SCREEN_DONE"
