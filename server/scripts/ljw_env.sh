#!/usr/bin/env bash
# Source this file from the LJW server checkout.
export LJW_ROOT=/home/project-b/LJW
export VLM_REPO="$LJW_ROOT/vlm-grounding-steering"
export VLM_GEMMA_CONFIG="$VLM_REPO/configs/gemma4_e4b.local.json"
export XDG_CACHE_HOME="$LJW_ROOT/.cache"
export XDG_CONFIG_HOME="$LJW_ROOT/.config"
export XDG_DATA_HOME="$LJW_ROOT/.local/share"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_TOKEN_PATH="$HF_HOME/token"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$XDG_DATA_HOME/uv/python"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export NLTK_DATA="$XDG_CACHE_HOME/nltk_data"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export TMPDIR="$LJW_ROOT/.tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$VLM_REPO/.venv/bin:$LJW_ROOT/.local/bin:$PATH"
