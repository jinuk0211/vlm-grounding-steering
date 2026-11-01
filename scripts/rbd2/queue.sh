#!/bin/bash
# usage: queue.sh GPU MODEL BENCH   -- RBD pilot runs for one benchmark on one GPU, one after another
# (resumable: rbd_decode appends rows and skips the ones already written)
set -u
G=$1; M=$2; B=$3
R=/home/project-b/LJW/vlm-grounding-steering; O=$R/outputs/rbd-test-pilot-20261101
export HF_HOME=/workspace/.hf_home PYTHONPATH=$R/scripts:$R/src CUDA_VISIBLE_DEVICES=$G
source /venv/main/bin/activate
cd $R/scripts
case $B in chair) D=/workspace/chair_test100.jsonl;; amber) D=/workspace/amber100.jsonl;; esac
run() {  # tag rbd_decode-args...
  t=$1; shift
  echo "$(date +%T) start $B $t" >> /workspace/logs/queue-$M-$B.log
  python rbd_decode.py --model $M --data $D --output $O/$B --tag $M-$t --start 0 --end 100 "$@" \
    >> /workspace/logs/run-$M-$B-$t.log 2>&1 || echo "$(date +%T) FAILED $B $t" >> /workspace/logs/queue-$M-$B.log
  echo "$(date +%T) end $B $t" >> /workspace/logs/queue-$M-$B.log
}
# Base and the dev-selected setting first, then the two controls that decide whether belief itself matters
run base    --lam 0
run rbd01   --lam 0.1
run const01 --lam 0.1 --r const
run unif01  --lam 0.1 --uniform
echo "$(date +%T) QUEUE_DONE" >> /workspace/logs/queue-$M-$B.log
