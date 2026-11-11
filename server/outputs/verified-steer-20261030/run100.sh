#!/bin/bash
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/verified-steer-20261030
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
U=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)
until [ $(wc -l < $O/qwen-ask-0-250.jsonl) -ge 100 ]; do sleep 3; done
kill 3346028; sleep 3; echo "qwen ask stopped at $(wc -l < $O/qwen-ask-0-250.jsonl) $(date +%T)"
QP=""; for c in base boost_3 both_3 both_6; do env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/verify_probe_steer.py gen --model qwen --output $O --cond $c --start 0 --end 100 --batch 16 > $O/logs/qwen-gen-$c.log 2>&1 & QP="$QP $!"; sleep 20; done
echo "qwen gens $QP $(date +%T)"
# gemma ask once one qwen gen has finished
while [ $(for p in $QP; do kill -0 $p 2>/dev/null && echo x; done | wc -l) -ge 4 ]; do sleep 5; done
env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/verify_probe_steer.py ask --model gemma --output $O --start 0 --end 100 --ask-batch 9 > $O/logs/gemma-ask-0-100.log 2>&1
echo "gemma ask done $(date +%T)"
for p in $QP; do while kill -0 $p 2>/dev/null; do sleep 5; done; done; echo "qwen gens done $(date +%T)"
env $ENV CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/verify_probe_steer.py summarize --model qwen --output $O --start 0 --end 100 > $O/logs/qwen-summary-100.log 2>&1; echo "qwen summary $(date +%T)"
GP=""; for c in base boost_3 both_3 both_6; do env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/verify_probe_steer.py gen --model gemma --output $O --cond $c --start 0 --end 100 --batch 8 > $O/logs/gemma-gen-$c.log 2>&1 & GP="$GP $!"; sleep 20; done
for p in $GP; do while kill -0 $p 2>/dev/null; do sleep 5; done; done
env $ENV CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/verify_probe_steer.py summarize --model gemma --output $O --start 0 --end 100 > $O/logs/gemma-summary-100.log 2>&1; echo "ALL_DONE $(date +%T)"
