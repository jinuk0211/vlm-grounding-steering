#!/bin/bash
# Fill GPU 1: Gemma reliance + AMBER Base now; more AMBER Base shards when the WordNet dev runs exit.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R
VR=$R/outputs/visual-reliance-20261030; AB=$R/outputs/amber-20261030; mkdir -p $AB/logs
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts"
U=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader); D=$R/data/amber/amber_generative.jsonl
base(){ env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/plain_generate.py --model $1 --data $D --output $AB --tag $1-base --start $2 --end $3 > $AB/logs/$1-base-$2.log 2>&1; }
env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/visual_reliance.py --model gemma --output $VR --start 0 --end 200 > $VR/logs/gemma.log 2>&1 &
sleep 60
base qwen 0 502 &
while kill -0 2226357 2>/dev/null || kill -0 2227555 2>/dev/null; do sleep 30; done
base qwen 502 1004 & sleep 30
base gemma 0 502 & sleep 30
base gemma 502 1004 &
wait
