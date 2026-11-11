#!/bin/bash
# User 11-10 ~18:55 KST: run the "- belief" row of the NEW-steering Qwen2.5 ablation (vast box abl_new.sh, tag abl_nobelief_new) on project-b GPU 2.
# vdd_steer.py copied unchanged from the box (/workspace/steer, md5 bd0a3e7c...); repo deps, set, images, prompts, vobj2, decay curve, gray448,
# model revision cc594898 and transformers 5.20.2 identical to the box; torch 2.8.0 here vs 2.7.1 there, A5000 here vs 2080 Ti there.
# After the 500 rows: rows 0-5 of Full (full_an16_r03) on this GPU, to compare with the box's Full rows (cross-hardware drift).
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; S=$R/scripts/steer_vast; O=$R/outputs/steer-nobelief-20261110; SET=$R/outputs/rbd2-20261102/sets/chair_test500.jsonl
mkdir -p $O $L/.cache/tmp; cd $R
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       NLTK_DATA=$L/.cache/nltk_data XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       VDD_GRAY=$R/outputs/rbd2-20261102/gray448.png PYTHONPATH=$S:$R/scripts:$R/src \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
NEW="--steer resid --layers 16 --alpha 2 --steer2 anchornorm --layers2 16 --alpha2 1 --rho2 0.3 --anchor-steps 16"
log(){ echo "$(date '+%m-%d %T') $*" >> $O/run.log; }
log "start abl_nobelief_new GPU 2 ($CUDA_VISIBLE_DEVICES)"
nice -n 5 .venv/bin/python $S/vdd_steer.py --model qwen --data $SET --output $O --tag abl_nobelief_new --start 0 --end 500 $NEW --lam 0 \
   < /dev/null >> $O/abl_nobelief_new-0-500.log 2>&1 && log "done abl_nobelief_new" || { log "FAILED abl_nobelief_new"; exit 1; }
nice -n 5 .venv/bin/python $S/vdd_steer.py --model qwen --data $SET --output $O --tag full_an16_r03_a5000 --start 0 --end 5 $NEW \
   < /dev/null >> $O/full_an16_r03_a5000-0-5.log 2>&1 && log "done full_an16_r03_a5000 0-5" || log "FAILED full check"
log STEER_NOBELIEF_DONE
