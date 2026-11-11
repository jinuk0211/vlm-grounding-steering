#!/bin/bash
# Top 100 at every position for the 100 images of out/jlens_pos (captions reused, no generation) -> out/jlens_pos100/
cd /workspace/tether-pilot
export HF_HOME=$PWD/hf NLTK_DATA=$PWD/nltk_data PYTHONPATH=$PWD/pilot:$PWD/pilot/tests:$PWD/tools/jacobian-lens
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PILOT_MODEL=qwen35
.venv35/bin/python pilot/jlens_positions_top100.py --model-dir models/Qwen3.5-4B --lens models/JLensVL-lenses/lens_qwen35_4b_final.pt --topn 100
echo RUN_EXIT=$?