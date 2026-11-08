#!/bin/bash
# Launch each baseline on the first of GPU 1/2/3 with >= 20 GB free; 90 s between launches so memory readings settle.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/rbd-20261030/dev
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts:src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for m in revis rudder_clone grains_clone vti; do
  while true; do
    for g in 1 2 3; do f=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits); if (( f >= 20480 )); then break 2; fi; done
    sleep 30
  done
  U=$(nvidia-smi --id=$g --query-gpu=uuid --format=csv,noheader)
  echo "$(date +%T) $m -> gpu$g (${f} MiB free)"
  nohup env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/rbd_baselines.py --method $m --output $O > $O/logs/qwen-$m.log 2>&1 &
  sleep 90
done
echo "$(date +%T) all launched"
