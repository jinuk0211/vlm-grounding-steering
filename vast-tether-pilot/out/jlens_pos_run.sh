#!/bin/bash
# Lens at every token position (prompt, image, Qwen3.5-4B caption) for the 100 pilot images -> out/jlens_pos/
cd /workspace/tether-pilot
export HF_HOME=$PWD/hf NLTK_DATA=$PWD/nltk_data PYTHONPATH=$PWD/pilot:$PWD/pilot/tests:$PWD/tools/jacobian-lens
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PILOT_MODEL=qwen35
.venv35/bin/python pilot/jlens_positions.py --model-dir models/Qwen3.5-4B --lens models/JLensVL-lenses/lens_qwen35_4b_final.pt --n-images 100 --out-dir out/jlens_pos
echo RUN_EXIT=$?
