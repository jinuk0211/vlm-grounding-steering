#!/bin/bash
# User 11-10 ~21:00 KST: run the belief-recall dev sweep (stopped on project-b at 20:45) on this box, after abl_new.sh and
# recheck_nosteer.sh. Qwen2.5, chair_dev100 (train2014 dev; set, images and scorer state copied from project-b), the steering of
# full_an16_r03; every variant decoded here so the dev table shares the hardware. Log logs/bposdev.log (BPOSDEV_DONE).
cd /workspace/steer
while pgrep -f abl_new.sh > /dev/null || pgrep -f recheck_nosteer.sh > /dev/null || pgrep -f vdd_steer.py > /dev/null; do sleep 60; done
echo "$(date +%T) start (UTC)" >> logs/bposdev.log
export NLTK_DATA=/workspace/tether-pilot/nltk_data
./run_queue.sh queue_bposdev.txt sets/chair_dev100.jsonl runs/bposdev 0 100 < /dev/null >> logs/bposdev.log 2>&1
R=/home/project-b/LJW/vlm-grounding-steering; D=runs/bposdev
export RBD2_SETS=/workspace/steer/sets PYTHONPATH=/workspace/steer:$R/scripts/rbd2:$R/scripts:$R/src
venv/bin/python steer_table.py chair_dev100 /workspace/steer/devbundle/chair-dev100-scorer-state.json \
  "Base=$D/dev_base-0-100.jsonl" "Full=$D/dev_full-0-100.jsonl" "- belief=$D/dev_nobelief-0-100.jsonl" "bpos lam1=$D/dev_bpos1-0-100.jsonl" \
  "bpos lam2=$D/dev_bpos2-0-100.jsonl" "bpos lam3=$D/dev_bpos3-0-100.jsonl" "lam2=$D/dev_lam2-0-100.jsonl" > $D/table.txt 2>> logs/bposdev.log
echo "$(date +%T) BPOSDEV_DONE ($D/table.txt)" >> logs/bposdev.log
