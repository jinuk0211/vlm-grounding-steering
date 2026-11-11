#!/bin/bash
# ALOHa scorer environment on project-b, all under /home/project-b/LJW (shared .venv untouched; same layout as pb_bench_setup.sh):
#   bench/aloha    DavidMChan/aloha @7da6b90 (HAT file data/hat-trainval.json, official metric)
#   venvs/aloha    CPU torch + aloha + spaCy en_core_web_lg; versions as tested on 2026-11-05. nltk and pandas are imported by
#                  aloha.metrics but missing from its pyproject.
#   HF cache       sentence-transformers/all-mpnet-base-v2, facebook/detr-resnet-50 (with its timm backbone)
set -euo pipefail
L=${HAT_L:-/home/project-b/LJW}; B=$L/bench; V=$L/venvs; UV=$L/.local/bin/uv; P=$V/aloha/bin/python
export UV_CACHE_DIR=$L/.cache/uv PIP_CACHE_DIR=$L/.cache/pip HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub \
       UV_PYTHON_INSTALL_DIR=$L/.local/share/uv/python UV_HTTP_TIMEOUT=600 NLTK_DATA=$L/.cache/nltk_data \
       XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp HF_HUB_DISABLE_XET=1   # nothing outside LJW; the xet download stalled on this server
mkdir -p $TMPDIR
PYBIN=$(readlink -f $L/vlm-grounding-steering/.venv/bin/python)
[ -d $B/aloha/.git ] || git clone -q https://github.com/DavidMChan/aloha $B/aloha
git -C $B/aloha checkout -q 7da6b90ebe392228ea532a84d78eab830c2b3cb2; git -C $B/aloha log -1 --format='aloha %H'
[ -x $P ] || $UV venv -q --python $PYBIN $V/aloha
$UV pip install -q --python $P 'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu' --index-url https://download.pytorch.org/whl/cpu   # PyPI's torchvision is built against CUDA torch
$UV pip install -q --python $P $B/aloha 'spacy==3.8.18' 'transformers==5.22.1' 'sentence-transformers==6.2.0' 'timm==1.0.31' 'scikit-learn==1.9.1' 'nltk>=3.9,<4' 'pandas<4'
W=$B/pkgs/en_core_web_lg-3.8.0-py3-none-any.whl; mkdir -p $B/pkgs
[ -s $W ] || curl -sSL --retry 5 -o $W https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
$UV pip install -q --python $P --no-deps $W
$P - <<'PY'
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor,DetrForObjectDetection
import aloha.metrics
spacy.load('en_core_web_lg');SentenceTransformer('all-mpnet-base-v2')
AutoImageProcessor.from_pretrained('facebook/detr-resnet-50');DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
print('aloha env ok')
PY
