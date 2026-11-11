#!/bin/bash
# Top-100 caption lens on GPU 2. Waits until the rbd6 pool worker r2 is gone and GPU 2 has >= 20 GB free on two checks a minute apart.
R=/home/project-b/LJW/vlm-grounding-steering; cd $R; O=$R/outputs/logit-lens-20261104
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=/home/project-b/LJW/.cache/nltk_data PYTHONPATH=scripts OMP_NUM_THREADS=6
U=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
free(){ nvidia-smi --id=2 --query-gpu=memory.free --format=csv,noheader,nounits; }
r2(){ ps -eo args | awk '$1 ~ /bash/ && $0 ~ /rbd6_pool\.sh 2 /' | grep -q .; }
gate(){ while true; do if ! r2; then a=$(free); sleep 60; b=$(free); if ! r2 && [ $a -ge 20000 ] && [ $b -ge 20000 ]; then return; fi; else sleep 20; fi; done; }
for m in qwen gemma; do
  gate; echo "$(date '+%m-%d %T') start $m ($(free) MiB free)"
  CUDA_VISIBLE_DEVICES=$U .venv/bin/python -u scripts/logit_lens.py topk --model $m --output $O > $O/logs/$m-topk.log 2>&1 && echo "$(date '+%m-%d %T') done $m" || echo "$(date '+%m-%d %T') FAILED $m"
  CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/logit_lens.py topk-analyze --model $m --output $O > $O/logs/$m-topk-analyze.log 2>&1; echo "$(date '+%m-%d %T') analyzed $m"
done; echo "$(date '+%m-%d %T') TOPK_DONE"
