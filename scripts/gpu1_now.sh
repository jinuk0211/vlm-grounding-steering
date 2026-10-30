#!/bin/bash
# GPU 1 has room now: run the WordNet dev variants and Qwen reliance immediately;
# Gemma reliance follows once the dev runs and the last Gemma CHAIR shard (pid 2052000) exit.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
V=$R/outputs/verified-decoding-20261030; VR=$R/outputs/visual-reliance-20261030
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts"
U=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)
dev(){ env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/verified_decoding_v3.py --model qwen --data $R/data/chair-dev-train2014-100/chair.jsonl \
    --output $V/dev --tag $1 --detector wordnet --key word --tau=$2 --start 0 --end 100 > $V/dev/logs/$1.log 2>&1; }
echo "start $(date)"
dev qwen-dev-wn-t0 0 & A=$!; sleep 20
dev qwen-dev-wn-tm1 -1 & B=$!; sleep 20
env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/visual_reliance.py --model qwen --output $VR --start 0 --end 200 > $VR/logs/qwen.log 2>&1 &
wait $A $B; while kill -0 2052000 2>/dev/null; do sleep 30; done
env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/visual_reliance.py --model gemma --output $VR --start 0 --end 200 > $VR/logs/gemma.log 2>&1
wait; echo "done $(date)"
