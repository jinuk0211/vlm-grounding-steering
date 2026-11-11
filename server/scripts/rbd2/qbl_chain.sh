#!/bin/bash
# User (11-05 19:5x): Qwen3-VL GrAInS before HAT+ALOHa. Vector as for the earlier models (extract_qwen3_grains.py, FP32, GPU 4), then 2 workers each on GPUs 4 and 5.
# Rows: runs/qchair_bl2. Log: logs/qbl.log. QBL_GRAINS_FAILED if the extraction stops, QBL_DONE at the end. REVIS is not here: its rerun waits for the user's decision.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/qbl.log; Q=$O/queues/qbl.txt
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
echo "=== $(date +%T) qbl_chain.sh start" >> $LOG
if ! CUDA_VISIBLE_DEVICES=$(uuid 4) .venv/bin/python -u scripts/rbd2/extract_qwen3_grains.py >> $LOG 2>$O/logs/qbl_grains.err; then echo "=== $(date +%T) QBL_GRAINS_FAILED (logs/qbl_grains.err)" >> $LOG; exit 1; fi
echo "=== $(date +%T) GrAInS vector ready" >> $LOG
for b in $(seq 0 50 450); do echo "grains_clone eager $b $((b+50))"; done > $Q
pool(){ G=$1; W=$2
  ( export CUDA_VISIBLE_DEVICES=$(uuid $G); C=$O/claims_qbl; mkdir -p $C
    while true; do OPEN=0; ran=0
      while read -r M AT S E; do
        F=$O/runs/qchair_bl2/$M-$S-$E.jsonl; U=$M-$S-$E
        [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ] && continue
        [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
        OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
        [ $(nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits) -ge 18900 ] || continue
        mkdir $C/$U 2>/dev/null || continue
        echo "$(date +%T) $W start $U" >> $O/logs/qblpool.log
        if .venv/bin/python -u scripts/rbd2/qwen3_bl.py --method $M --attn $AT --set $O/sets/chair_test500.jsonl --output $O/runs/qchair_bl2 --start $S --end $E < /dev/null > $O/logs/qblunit-$W.log 2>&1
        then echo "$(date +%T) $W end $U" >> $O/logs/qblpool.log
        else echo "$(date +%T) $W FAILED $U" >> $O/logs/qblpool.log; echo $W >> $C/fail-$U; rm -r "${C:?}/${U:?}"; fi
        ran=1; break
      done < $Q
      [ $ran = 1 ] && continue
      [ $OPEN -eq 0 ] && break
      sleep 30
    done; echo "$(date +%T) $W POOL_DONE" >> $O/logs/qblpool.log ) & }
# the second worker on a GPU starts 150 s later, so the free-memory check sees the first model
pool 4 q4a; pool 5 q5a; sleep 150; pool 4 q4b; pool 5 q5b
wait
echo "=== $(date +%T) decoded" >> $LOG
CUDA_VISIBLE_DEVICES= .venv/bin/python -u scripts/rbd2/analysis/qbl_report.py >> $LOG 2>$O/logs/qbl_report.err
echo "=== $(date +%T) QBL_DONE" >> $LOG
