#!/bin/bash
# Waits for the Qwen3.5-4B and lens downloads, then tests, probes 300 pilot images and summarizes.
cd /workspace/tether-pilot
export HF_HOME=$PWD/hf NLTK_DATA=$PWD/nltk_data PYTHONPATH=$PWD/pilot:$PWD/pilot/tests:$PWD/tools/jacobian-lens
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PILOT_MODEL=qwen35
until grep -q FETCH_DONE out/jlens_fetch.log; do sleep 30; done
if [ "$(grep -c FETCH_OK out/jlens_fetch.log)" -lt 2 ]; then echo DOWNLOAD_FAILED; echo RUN_EXIT=1; exit 1; fi
M=$PWD/models/Qwen3.5-4B
L=$PWD/models/JLensVL-lenses/lens_qwen35_4b_final.pt
echo "=== $(date +%T) tests"
(cd pilot && CUDA_VISIBLE_DEVICES="" ../.venv35/bin/python -m pytest tests/test_jlens_probe.py -q -W ignore -p no:cacheprovider) || { echo TESTS_FAILED; echo RUN_EXIT=1; exit 1; }
echo "=== $(date +%T) probe"
.venv35/bin/python pilot/jlens_probe.py --model-dir "$M" --lens "$L" --n-images "${N_IMAGES:-100}" || { echo RUN_EXIT=1; exit 1; }
echo "=== $(date +%T) summary"
.venv35/bin/python pilot/jlens_summary.py; echo RUN_EXIT=$?
