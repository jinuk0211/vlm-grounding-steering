#!/bin/bash
# Environment for the RBD pilot on a host that caps each connection near 150 KB/s and drops long transfers:
# big wheels via parallel range requests (sha256 from the index), the rest from PyPI with retries.
set -eu
source /venv/main/bin/activate
cd /workspace/rbd_pilot
PY=/venv/main/bin/python; R=/home/project-b/LJW/vlm-grounding-steering
python3 wheels.py /workspace/wheels jinja2==3.1.6 markupsafe==3.0.3 mpmath==1.3.0 networkx==3.7 sympy==1.14.0 \
  numpy==2.5.4 pillow==13.0.0 \
  nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 \
  nvidia-cuda-runtime-cu12==12.8.90 nvidia-cudnn-cu12==9.10.2.21 nvidia-cufft-cu12==11.3.3.83 \
  nvidia-cufile-cu12==1.13.1.3 nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 \
  nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.27.3 \
  nvidia-nvjitlink-cu12==12.8.93 nvidia-nvtx-cu12==12.8.90 triton==3.4.0 torch==2.8.0+cu128 torchvision==0.23.0+cu128
uv pip install --python $PY --no-index --find-links /workspace/wheels 'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
L=https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
W=/workspace/wheels/en_core_web_lg-3.8.0-py3-none-any.whl
[ -s $W ] || python3 -c "import pdl,wheels;pdl.fetch('$L','$W',wheels.size('$L'),32)"
for i in 1 2 3 4 5 6 7 8; do
  UV_HTTP_TIMEOUT=600 uv pip install --python $PY 'transformers==5.20.2' 'accelerate>=1.0,<2' 'numpy<3' 'Pillow<13' \
    'nltk>=3.9,<4' 'PatternLite==3.6' pycocotools tqdm 'spacy>=3.8,<3.9' && break
  sleep 15
done
uv pip install --python $PY --no-deps $W
uv pip install --python $PY --no-deps -e $R
$PY -c "import nltk;[nltk.download(x,quiet=True) for x in ['wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
$PY -c "import torch,transformers,spacy;spacy.load('en_core_web_lg');print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo ENV_OK
