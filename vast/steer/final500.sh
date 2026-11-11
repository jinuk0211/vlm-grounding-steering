#!/bin/bash
# Author's decision (PROTOCOL.md, "Final run"): one pre-specified variant, full_an16_r03, on all 500 CHAIR test rows,
# compared with the paper's 500-row runs (nosteer = abl_nosteer, full = rbd_full, base); then same-box nosteer rows 100-499.
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering; P=$R/outputs/rbd2-20261102/runs/chair_abl500_all
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
TAG=full_an16_r03; ARGS="--steer resid --layers 16 --alpha 2 --steer2 anchornorm --layers2 16 --alpha2 1 --rho2 0.3 --anchor-steps 16"
while pgrep -f vdd_steer.py > /dev/null; do sleep 30; done
echo "$(date +%T) GPU free"
# round-2 record (scores only; its confirmation step is not run, by the author's decision)
RBD2_N=100 venv/bin/python steer_score.py chair_test100 runs/steer100b 0 100 > runs/steer100b/score.txt 2> logs/score2.err
echo "$(date +%T) round-2 scored: runs/steer100b/score.txt"
# smoke test: 1 image, both hooks must fire on every caption
rm -rf runs/smoke_final; echo "$TAG $ARGS" > queue_one.txt
./run_queue.sh queue_one.txt sets/chair_test500.jsonl runs/smoke_final 0 1 < /dev/null
python3 -c "
import json; r=[json.loads(l) for l in open('runs/smoke_final/$TAG-0-1.jsonl')]
assert r and all(x['steer_calls']>0 and x['steer2_calls']>0 for x in r), r
print('smoke ok: steer_calls', r[0]['steer_calls'], 'steer2_calls', r[0]['steer2_calls'], 'tokens', r[0]['generated_tokens'])" || { echo "$(date +%T) SMOKE_FAILED"; exit 1; }
# main run: 500 images
F=runs/final500; mkdir -p $F
./run_queue.sh queue_one.txt sets/chair_test500.jsonl $F 0 500 < /dev/null
# score: vs the paper's 500-row runs, all 500 rows and rows 100-499 (rows 0-99 informed the choice of the an16 family)
for X in all500 rows100; do mkdir -p $F/$X; done
for t in nosteer:abl_nosteer full:rbd_full base:base; do ln -sfn $P/${t#*:}-0-500.jsonl $F/all500/${t%%:*}-0-500.jsonl; done
ln -sfn /workspace/steer/$F/$TAG-0-500.jsonl $F/all500/$TAG-0-500.jsonl
for f in $F/all500/*-0-500.jsonl; do b=$(basename $f -0-500.jsonl); tail -n +101 $f > $F/rows100/$b-100-500.jsonl; done
RBD2_N=500 venv/bin/python cscore.py chair_test500 $F/all500 0 500 > $F/score_all500.txt 2> logs/final_score.err
RBD2_N=500 venv/bin/python cscore.py chair_test500 $F/rows100 100 500 > $F/score_rows100.txt 2>> logs/final_score.err
echo "$(date +%T) FINAL500_SCORED"
# hardware check: same-box nosteer on rows 100-499, joined with the existing same-box rows 0-99
echo "nosteer --steer none" > queue_one.txt
./run_queue.sh queue_one.txt sets/chair_test500.jsonl runs/confirm400 100 500 < /dev/null
mkdir -p $F/samebox
cat runs/steer100/nosteer-0-100.jsonl runs/confirm400/nosteer-100-500.jsonl > $F/samebox/nosteer-0-500.jsonl
ln -sfn /workspace/steer/$F/$TAG-0-500.jsonl $F/samebox/$TAG-0-500.jsonl
RBD2_N=500 venv/bin/python cscore.py chair_test500 $F/samebox 0 500 > $F/score_samebox.txt 2>> logs/final_score.err
echo "$(date +%T) SAMEBOX_SCORED"
