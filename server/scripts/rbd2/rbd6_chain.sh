#!/bin/bash
# RBD layer / hyperparameter search (rbd6_ctl.py has the protocol). Workers join a GPU only after the Qwen3 baselines on it have ended; nothing
# running is touched. GPU 2, GPU 3, GPU 4 x2, GPU 5 x2. Log: logs/rbd6.log, RBD6_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/rbd6.log; Q=$O/queues/rbd6.txt
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
CUDA_VISIBLE_DEVICES= NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
  .venv/bin/python -u scripts/rbd2/rbd6_ctl.py run >> $LOG 2> $O/logs/rbd6_ctl.err &
until [ -f $Q ]; do sleep 5; done
start(){ G=$1; N=$2; D=$3; shift 3
  ( while alive $@; do sleep 30; done; sleep $D; echo "=== $(date +%T) GPU $G worker $N joins" >> $LOG
    exec bash scripts/rbd2/rbd6_pool.sh $G $N $Q 18900 ) & }
start 2 r2 0 3877612 3877590
start 3 r3 0 3877614 3877591
start 4 r4a 0 3750721 3750746 3795174
start 4 r4b 150 3750721 3750746 3795174
start 5 r5a 0 3750722 3750744 3795175
start 5 r5b 150 3750722 3750744 3795175
wait
echo "=== $(date +%T) RBD6_CHAIN_END" >> $LOG
