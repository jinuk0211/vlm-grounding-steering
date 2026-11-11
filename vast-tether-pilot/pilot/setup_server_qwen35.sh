#!/bin/bash
# Second environment for Qwen3.5 (transformers 5.22.2), beside .venv (4.57.3 and Spectra), run from the folder that holds pilot/.
# Torch and its CUDA libraries are not installed again: a .pth file appends .venv's site-packages after .venv35's own, so
# .venv35's transformers wins and torch 2.7.1+cu128 comes from .venv. Qwen3.5 has no Spectra adapter: none is installed here.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
[ -d .venv35 ] || uv venv --python 3.11 .venv35
PY=$ROOT/.venv35/bin/python
uv pip install -q --python "$PY" "transformers==5.22.2" einops pillow numpy scipy scikit-learn nltk pycocotools pytest tqdm "PatternLite==3.6" psutil
uv pip install -q --python "$PY" --no-deps accelerate
BASE_SITE=$("$ROOT/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])")
OWN_SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])")
echo "$BASE_SITE" > "$OWN_SITE/zz_base_venv.pth"
"$PY" -c "
import torch, transformers
from transformers import Qwen3_5ForConditionalGeneration
print('torch', torch.__version__, torch.__file__.split('site-packages')[0], 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__, transformers.__file__.split('site-packages')[0])"
