#!/bin/bash
# Object HalBench + THRONE scoring setup (CPU/network only; the GPUs keep decoding).
#   judges    google/flan-t5-{base,large,xl} at pinned revisions (THRONE one-click.sh evaluators)
#   COCO 2017 instances_val2017.json (THRONE coco), spaCy en_core_web_trf 3.7.2 (Object HalBench README)
#   venvs     --system-site-packages, so torch comes from /venv/main; only the pinned scorer packages shadow it
#             throne: transformers 4.37.2 (THRONE requirements.txt);  objhal: spacy 3.7 + nltk 3.8.1 (RLHF-V README)
set -e
B=/workspace/bench; P=/workspace/rbd_pilot; PY=/venv/main/bin/python
mkdir -p $B/models $B/coco2017/annotations $B/coco2017/val2017 $B/pkgs
PIP="--retries 20 --timeout 60 --progress-bar off"
echo "$(date +%T) venvs"
for v in throne objhal; do [ -x /workspace/venv-$v/bin/python ] || $PY -m venv --system-site-packages /workspace/venv-$v; done
/workspace/venv-throne/bin/python -m pip install $PIP "transformers==4.37.2" "tokenizers<0.19" "huggingface-hub<0.25" "numpy==1.26.4" \
  "pycocotools==2.0.8" "accelerate==0.21.0" "sentencepiece==0.2.0" "safetensors<0.5"
echo "$(date +%T) spacy trf model"
$PY $P/fetchx.py url https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.2/en_core_web_trf-3.7.2-py3-none-any.whl $B/pkgs/en_core_web_trf-3.7.2-py3-none-any.whl
/workspace/venv-objhal/bin/python -m pip install $PIP "spacy>=3.7.2,<3.8" "nltk==3.8.1" jsonlines openai "numpy<2" $B/pkgs/en_core_web_trf-3.7.2-py3-none-any.whl
echo "$(date +%T) coco2017 annotations"
$PY $P/fetchx.py url http://images.cocodataset.org/annotations/annotations_trainval2017.zip $B/coco2017/annotations_trainval2017.zip
$PY -c "import zipfile;z=zipfile.ZipFile('$B/coco2017/annotations_trainval2017.zip');open('$B/coco2017/annotations/instances_val2017.json','wb').write(z.read('annotations/instances_val2017.json'))"
echo "$(date +%T) judges"
$PY $P/fetchx.py hf google/flan-t5-base 7bcac572ce56db69c1ea7c8af255c5d7c9672fc2 $B/models/flan-t5-base
$PY $P/fetchx.py hf google/flan-t5-large 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a $B/models/flan-t5-large
$PY $P/fetchx.py hf google/flan-t5-xl 7d6315df2c2fb742f0f5b556879d730926ca9001 $B/models/flan-t5-xl
echo "$(date +%T) checks"
/workspace/venv-throne/bin/python -c "import transformers,pycocotools,accelerate;print('throne env',transformers.__version__)"
/workspace/venv-objhal/bin/python -c "import spacy,nltk;spacy.load('en_core_web_trf');print('objhal env',spacy.__version__,nltk.__version__)"
echo "$(date +%T) BENCH_SETUP_DONE"
