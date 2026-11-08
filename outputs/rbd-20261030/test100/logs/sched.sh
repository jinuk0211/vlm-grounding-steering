#!/bin/bash
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; T=$R/outputs/rbd-20261030/test100
D=$R/outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386/original-vti-gpu1-final/chair/base.jsonl
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for spec in "qwen-b0.3:--lam 0.3" "qwen-b0.1:--lam 0.1"; do tag=${spec%%:*}; args=${spec#*:}
  while true; do for g in 1 2 3; do f=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits); (( f >= 20480 )) && break 2; done; sleep 20; done
  U=$(nvidia-smi --id=$g --query-gpu=uuid --format=csv,noheader); echo "$(date +%T) $tag -> gpu$g"
  nohup env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/rbd_decode.py --model qwen --data $D --output $T --tag $tag $args --end 100 > $T/logs/$tag.log 2>&1 &
  sleep 90
done
