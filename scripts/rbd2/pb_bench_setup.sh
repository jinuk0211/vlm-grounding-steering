#!/bin/bash
# Scorer environments for round 2 on project-b, all under /home/project-b/LJW (shared .venv untouched).
#   venvs/amber   spaCy 3.8 + en_core_web_lg 3.8.0 + nltk   (AMBER official inference.py)
#   venvs/throne  THRONE requirements pins (transformers 4.37.2 ...) + torch 2.8.0+cu128 from the uv cache
#   bench/THRONE  amazon-science/THRONE @45badbb7; bench/models/flan-t5-{base,large,xl} at the pinned revisions
#   bench/coco2017/annotations/instances_val2017.json
set -euo pipefail
L=/home/project-b/LJW; B=$L/bench; V=$L/venvs; UV=$L/.local/bin/uv
export UV_CACHE_DIR=$L/.cache/uv PIP_CACHE_DIR=$L/.cache/pip HF_HOME=$L/.cache/huggingface UV_PYTHON_INSTALL_DIR=$L/.local/share/uv/python UV_HTTP_TIMEOUT=600
PYBIN=$(readlink -f /home/project-b/LJW/vlm-grounding-steering/.venv/bin/python)
mkdir -p $B/models $B/coco2017/annotations $B/pkgs $V
echo "$(date +%T) amber venv"
[ -x $V/amber/bin/python ] || $UV venv -q --python $PYBIN $V/amber
$UV pip install -q --python $V/amber/bin/python 'spacy>=3.8,<3.9' 'nltk>=3.9,<4' 'numpy<3' tqdm
W=$B/pkgs/en_core_web_lg-3.8.0-py3-none-any.whl
[ -s $W ] || curl -sSL --retry 5 -o $W https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
$UV pip install -q --python $V/amber/bin/python --no-deps $W
$V/amber/bin/python -c "import nltk;[nltk.download(x,download_dir='$L/.cache/nltk_data',quiet=True) for x in ('averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]"
NLTK_DATA=$L/.cache/nltk_data $V/amber/bin/python -c "import spacy,nltk;spacy.load('en_core_web_lg');nltk.pos_tag(nltk.word_tokenize('a dog on a sofa'));print('amber env ok',spacy.__version__)"
echo "$(date +%T) THRONE repo"
[ -d $B/THRONE/.git ] || git clone -q https://github.com/amazon-science/THRONE $B/THRONE
git -C $B/THRONE checkout -q 45badbb71c2b3e8d6619f6b45cccdac24dd20b46; git -C $B/THRONE log -1 --format='THRONE %H'
echo "$(date +%T) throne venv"
[ -x $V/throne/bin/python ] || $UV venv -q --python $PYBIN $V/throne
$UV pip install -q --python $V/throne/bin/python 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
$UV pip install -q --python $V/throne/bin/python "transformers==4.37.2" "tokenizers<0.19" "huggingface-hub<0.25" "numpy==1.26.4" \
  "pycocotools==2.0.8" "accelerate==0.21.0" "sentencepiece==0.2.0" "safetensors<0.5"
$V/throne/bin/python -c "import torch,transformers,pycocotools,accelerate;print('throne env ok',torch.__version__,transformers.__version__)"
echo "$(date +%T) coco2017 annotations"
J=$B/coco2017/annotations/instances_val2017.json
if [ ! -s $J ]; then Z=$B/coco2017/annotations_trainval2017.zip
  [ -s $Z ] || curl -sSL --retry 5 -o $Z http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  $V/throne/bin/python -c "import zipfile;z=zipfile.ZipFile('$Z');open('$J','wb').write(z.read('annotations/instances_val2017.json'))"; rm -f $Z; fi
echo "$(date +%T) judges"
for spec in base:7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 large:0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a xl:7d6315df2c2fb742f0f5b556879d730926ca9001; do
  s=${spec%%:*}; rev=${spec#*:}
  $V/throne/bin/python -c "
from huggingface_hub import snapshot_download
p=snapshot_download('google/flan-t5-$s',revision='$rev',local_dir='$B/models/flan-t5-$s',local_dir_use_symlinks=False,
  allow_patterns=['*.safetensors','config.json','generation_config.json','spiece.model','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json'])
print('judge ok',p)"
done
du -sh $B/models/* $V/*
echo "$(date +%T) BENCH_SETUP_DONE"
