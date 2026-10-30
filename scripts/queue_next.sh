#!/bin/bash
# Launch follow-up jobs as each GPU's current Gemma test shards exit.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
V=$R/outputs/verified-decoding-20261030; VR=$R/outputs/visual-reliance-20261030; mkdir -p $VR/logs $V/dev $V/dev/logs
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts"
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
waitpids(){ for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 30; done; done; }
waitdev(){ while [ ! -f $R/data/chair-dev-train2014-100/chair-scorer-state.json ]; do sleep 30; done; }
dev(){ # gpu tag key tau
  env $ENV CUDA_VISIBLE_DEVICES=$(uuid $1) .venv/bin/python -u scripts/verified_decoding_v2.py --model qwen --data $R/data/chair-dev-train2014-100/chair.jsonl \
    --output $V/dev --tag $2 --key $3 --tau=$4 --start 0 --end 100 > $V/dev/logs/$2.log 2>&1; }
( waitpids 2132977; waitdev; echo "gpu2 start $(date)"; dev 2 qwen-dev-base node -inf; dev 2 qwen-dev-word-t0 word 0 ) > $V/dev/logs/queue-gpu2.log 2>&1 &
( waitpids 2051999; waitdev; echo "gpu3 start $(date)"; dev 3 qwen-dev-node-t0 node 0; dev 3 qwen-dev-word-tm1 word -1 ) > $V/dev/logs/queue-gpu3.log 2>&1 &
( waitpids 2132978 2132979 2132982 2052000; echo "gpu1 start $(date)"
  env $ENV CUDA_VISIBLE_DEVICES=$(uuid 1) .venv/bin/python -u scripts/visual_reliance.py --model qwen --output $VR --start 0 --end 200 > $VR/logs/qwen.log 2>&1 &
  sleep 60
  env $ENV CUDA_VISIBLE_DEVICES=$(uuid 1) .venv/bin/python -u scripts/visual_reliance.py --model gemma --output $VR --start 0 --end 200 > $VR/logs/gemma.log 2>&1 &
  wait ) > $VR/logs/queue-gpu1.log 2>&1 &
wait
