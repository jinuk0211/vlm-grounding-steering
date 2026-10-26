#!/usr/bin/env bash
set -Eeuo pipefail
root=/workspace/original-baselines-20261025/grains_gpu_retry_20261026
export PYTHONPATH="/workspace/original-baselines-20261025/dependencies:/workspace/qwen-author-check-20261024/protocols/vti_selected_generation_20261025/runtime:/workspace/qwen-author-check-20261024/dependencies:/workspace/vlm-baselines-20261024/dependencies"
export HF_HUB_CACHE=/workspace/vlm-validation-1683e86/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export ACCELERATE_MIXED_PRECISION=no
export NLTK_DATA=/workspace/original-baselines-20261025/nltk_data_compat_20261026
export MPLCONFIGDIR="/workspace/original-baselines-20261025/matplotlib_cache"
exec /workspace/vlm-validation-1683e86/.venv/bin/python -u "$root/operations/run_clone_benchmark_20261026.py" --method rudder --output "$root/runs/rudder"
