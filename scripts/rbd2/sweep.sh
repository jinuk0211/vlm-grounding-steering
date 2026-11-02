#!/bin/bash
# usage: sweep.sh GPU QUEUEFILE -- wait until GPU is free of the rbd queue, then run "TAG args..." lines on chair_test60
G=$1; Q=$2
R=/home/project-b/LJW/vlm-grounding-steering; O=/workspace/vdd/chair_test60_sweep; L=/workspace/logs/sweep
mkdir -p $O $L
export HF_HOME=/workspace/.hf_home PYTHONPATH=/workspace/rbd_pilot:$R/scripts:$R/src CUDA_VISIBLE_DEVICES=$G
source /venv/main/bin/activate >/dev/null
echo "$(date +%T) waiting for GPU $G" >> $L/sweep$G.log
while pgrep -f "[w]orker.sh $G " >/dev/null || pgrep -f "[v]run.sh $G " >/dev/null \
      || [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1000 ]; do sleep 20; done
while read -r T A; do
  [ -z "$T" ] && continue
  if [ -f $O/$T-0-60.jsonl ] && [ $(wc -l < $O/$T-0-60.jsonl) -ge 60 ]; then continue; fi
  echo "$(date +%T) start $T $A" >> $L/sweep$G.log
  python /workspace/sweep/sweep_decode.py --model qwen --data /workspace/sets/chair_test60.jsonl --output $O --tag $T \
    --start 0 --end 60 $A < /dev/null >> $L/$T.log 2>&1 || echo "$(date +%T) FAILED $T" >> $L/sweep$G.log
  echo "$(date +%T) end $T" >> $L/sweep$G.log
done < $Q
echo "$(date +%T) SWEEP_DONE" >> $L/sweep$G.log
