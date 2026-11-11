#!/bin/bash
# Round 2 screen: 10 variants x chair_test100 (round-1 full / nosteer as references), then the PROTOCOL.md rule
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering
./run_queue.sh queue_qwen100_r2.txt sets/chair_test100.jsonl runs/steer100b 0 100
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json RBD2_N=100
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
venv/bin/python steer_score.py chair_test100 runs/steer100b 0 100 > runs/steer100b/score.txt 2> logs/score2.err
echo "$(date +%T) SCREEN2_DONE"
PICK=$(python3 -c "import json;s=json.load(open('runs/steer100b/steer_score-0-100.json'))['selection'];print(s['picked'] if s['carry_to_confirmation'] else '')" 2>/dev/null)
if [ -z "$PICK" ]; then echo "$(date +%T) no round-2 variant carried to confirmation"; exit 0; fi
echo "$(date +%T) picked $PICK; confirmation on rows 100-499"
{ grep -E "^(full|nosteer) " queue_qwen100.txt; grep -E "^$PICK " queue_qwen100_r2.txt; } > queue_confirm2.txt
./run_queue.sh queue_confirm2.txt sets/chair_test500.jsonl runs/confirm400 100 500
export RBD2_N=500
venv/bin/python steer_score.py chair_test500 runs/confirm400 100 500 > runs/confirm400/score.txt 2> logs/score_confirm2.err
echo "$(date +%T) CONFIRM2_DONE ($PICK)"
