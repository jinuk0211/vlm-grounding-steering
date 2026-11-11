#!/bin/bash
# Round 3 screen (rows 0-99), then confirmation on rows 100-499: nosteer, round-3 pick, full, an16_r05 (PROTOCOL.md, round 3).
# Starts after screen2.sh exits. Scores the confirmation after every run (runs/confirm400/cscore.txt).
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
while pgrep -f screen2.sh > /dev/null; do sleep 60; done
echo "$(date +%T) screen2 finished; round 3 screen"
D=runs/steer100c; mkdir -p $D
for t in full nosteer; do ln -sfn /workspace/steer/runs/steer100/$t-0-100.jsonl $D/$t-0-100.jsonl; done
./run_queue.sh queue_qwen100_r3.txt sets/chair_test100.jsonl $D 0 100
RBD2_N=100 venv/bin/python steer_score.py chair_test100 $D 0 100 > $D/score.txt 2> logs/score3.err
for t in full_an16_r02 full_an16_r03; do
  python3 -c "import json;r=[json.loads(l) for l in open('$D/$t-0-100.jsonl')];print('$t steer2_calls min',min(x['steer2_calls'] for x in r),'steer_calls min',min(x['steer_calls'] for x in r))" 2>/dev/null
done
PICK=$(python3 -c "
import json;s=json.load(open('$D/steer_score-0-100.json'));V=s['vs_nosteer'];ns=V['ref:nosteer']
c=[t for t in V if t not in ('ref:nosteer','full') and V[t]['recall']>=ns['recall']-1.0]
p=min(c,key=lambda t:(V[t]['CHAIR_s'],V[t]['CHAIR_i'])) if c else ''
print(p if p and V[p]['CHAIR_s']<ns['CHAIR_s'] else '')" 2>/dev/null)
echo "$(date +%T) SCREEN3_DONE; round-3 pick: ${PICK:-none}"
{ grep -E "^nosteer " queue_qwen100.txt; [ -n "$PICK" ] && grep -E "^$PICK " queue_qwen100_r3.txt
  grep -E "^full " queue_qwen100.txt; grep -E "^an16_r05 " queue_qwen100_r2.txt; } > queue_confirm3.txt
cat queue_confirm3.txt
C=runs/confirm400; mkdir -p $C
mapfile -t LINES < queue_confirm3.txt
for line in "${LINES[@]}"; do
  echo "$line" > queue_one.txt
  ./run_queue.sh queue_one.txt sets/chair_test500.jsonl $C 100 500 < /dev/null
  RBD2_N=500 venv/bin/python cscore.py chair_test500 $C 100 500 --pooled runs/steer100 > $C/cscore.txt 2> logs/cscore.err
  echo "$(date +%T) scored after ${line%% *}"
done
echo "$(date +%T) CONFIRM3_DONE"
