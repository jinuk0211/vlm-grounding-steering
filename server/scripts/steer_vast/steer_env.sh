# sourced by the steer_vast scripts: caches and temp files under /home/project-b/LJW, GPU by UUID (index $G)
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; S=$R/scripts/steer_vast; mkdir -p $L/.cache/tmp; cd $R
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       NLTK_DATA=$L/.cache/nltk_data XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       VDD_GRAY=$R/outputs/rbd2-20261102/gray448.png PYTHONPATH=$S:$R/scripts:$R/src RBD2_SETS=$R/outputs/rbd2-20261102/sets \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
# the anchor term of the Qwen2.5 pick full_an16_r03 (layer of the second hook is set per model with --layers2)
NEW="--steer resid --steer2 anchornorm --alpha2 1 --rho2 0.3 --anchor-steps 16"
