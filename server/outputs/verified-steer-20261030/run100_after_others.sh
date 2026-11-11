#!/bin/bash
# Start only after every other project GPU job (anything under scripts/ that is not verify_probe_*) has been gone for 2 minutes.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/verified-steer-20261030
others(){ ps -eo args | awk '$1 ~ /\.venv\/bin\/python/ && $0 ~ /scripts\// && $0 !~ /verify_probe_/' | wc -l; }
quiet=0
while [ $quiet -lt 4 ]; do if [ $(others) -eq 0 ]; then quiet=$((quiet+1)); else quiet=0; fi; sleep 30; done
echo "other sessions idle since $(date +%T)"
ENV="HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
U=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)
QP=""; for c in base boost_3 both_3 both_6; do env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/verify_probe_steer.py gen --model qwen --output $O --cond $c --start 0 --end 100 --batch 16 > $O/logs/qwen-gen-$c.log 2>&1 & QP="$QP $!"; sleep 20; done
env $ENV CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader) .venv/bin/python -u scripts/verify_probe_steer.py ask --model gemma --output $O --start 0 --end 100 --ask-batch 9 > $O/logs/gemma-ask-0-100.log 2>&1
echo "gemma ask done $(date +%T)"
GP=""; for x in "2 base" "3 boost_3"; do set -- $x; env $ENV CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader) .venv/bin/python -u scripts/verify_probe_steer.py gen --model gemma --output $O --cond $2 --start 0 --end 100 --batch 8 > $O/logs/gemma-gen-$2.log 2>&1 & GP="$GP $!"; sleep 20; done
for p in $QP; do while kill -0 $p 2>/dev/null; do sleep 5; done; done; echo "qwen gens done $(date +%T)"
env $ENV CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/verify_probe_steer.py summarize --model qwen --output $O --start 0 --end 100 > $O/logs/qwen-summary-100.log 2>&1; echo "qwen summary $(date +%T)"
for c in both_3 both_6; do env $ENV CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/verify_probe_steer.py gen --model gemma --output $O --cond $c --start 0 --end 100 --batch 8 > $O/logs/gemma-gen-$c.log 2>&1 & GP="$GP $!"; sleep 20; done
for p in $GP; do while kill -0 $p 2>/dev/null; do sleep 5; done; done
env $ENV CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/verify_probe_steer.py summarize --model gemma --output $O --start 0 --end 100 > $O/logs/gemma-summary-100.log 2>&1; echo "ALL_DONE $(date +%T)"
