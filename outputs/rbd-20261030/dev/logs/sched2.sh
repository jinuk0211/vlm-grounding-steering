#!/bin/bash
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/rbd-20261030/dev
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for spec in "qwen-b0.05:--lam 0.05" "qwen-b0.1s1.0:--lam 0.1 --alpha 1.0 --layer 16"; do t=${spec%%:*}; args=${spec#*:}
  while true; do for g in 1 2 3; do f=$(nvidia-smi --id=$g --query-gpu=memory.free --format=csv,noheader,nounits); (( f >= 20480 )) && break 2; done; sleep 20; done
  U=$(nvidia-smi --id=$g --query-gpu=uuid --format=csv,noheader); echo "$(date +%T) $t -> gpu$g"
  nohup env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/rbd_decode.py --model qwen --output $O --tag $t $args > $O/logs/$t.log 2>&1 &
  sleep 90
done
