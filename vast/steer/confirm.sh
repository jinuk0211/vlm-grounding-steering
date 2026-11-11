#!/bin/bash
# Confirmation (PROTOCOL.md): after the screen, if the fixed rule carries a variant forward, decode it with full and
# nosteer on chair_test500 rows 100-499 (not used for selection) and score; otherwise stop.
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering
while ! grep -q SCREEN_DONE logs/screen.log; do sleep 60; done
PICK=$(python3 -c "import json;s=json.load(open('runs/steer100/steer_score-0-100.json'))['selection'];print(s['picked'] if s['carry_to_confirmation'] else '')" 2>/dev/null)
if [ -z "$PICK" ]; then echo "$(date +%T) no variant carried to confirmation (see runs/steer100/score.txt)"; exit 0; fi
echo "$(date +%T) picked $PICK; confirmation on rows 100-499"
grep -E "^(full|nosteer|$PICK) " queue_qwen100.txt > queue_confirm.txt
./run_queue.sh queue_confirm.txt sets/chair_test500.jsonl runs/confirm400 100 500
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json RBD2_N=500
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
venv/bin/python steer_score.py chair_test500 runs/confirm400 100 500 > runs/confirm400/score.txt 2> logs/score_confirm.err
echo "$(date +%T) CONFIRM_DONE ($PICK)"
