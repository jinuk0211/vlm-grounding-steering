L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export HAT_RUN=hat385 HAT_ALOHA_FIX=1 TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
for m in qwen gemma; do
  nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $L/venvs/aloha/bin/python scripts/rbd2/hat_aloha.py score hf:Qwen/Qwen3-VL-8B-Instruct \
    $O/runs/hat385_$m 60 385 $O/hat385/scorefix-$m-60-385 > $O/logs/hat_fix-$m.log 2> $O/logs/hat_fix-$m.err &
done; wait; echo HAT_FIX_DONE >> $O/logs/hat_fix-qwen.log
