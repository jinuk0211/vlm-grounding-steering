#!/bin/bash
# Waits until the four Qwen3 ablation runs have 500 rows, then writes outputs/steer-anchor-20261110/qwen3/abl_table.txt (QWEN3_ABL_DONE in pool.log).
G=0; source $(dirname $0)/steer_env.sh; export CUDA_VISIBLE_DEVICES=
O=$R/outputs/steer-anchor-20261110/qwen3; Q=$R/outputs/rbd2-20261102/runs/qchair_test500_all
done500(){ [ -f $1 ] && [ $(wc -l < $1) -ge 500 ]; }
until done500 $O/anchor_qwen3-0-500.jsonl && done500 $O/abl_nosteer_qwen3-0-500.jsonl && done500 $O/abl_nobelief_qwen3-0-500.jsonl \
      && done500 $O/abl_nopen_qwen3-0-500.jsonl; do sleep 120; done
nice -n 10 .venv/bin/python $S/steer_table.py chair_test500 $R/data/hallucination-suite/chair-scorer-state.json \
  "Full (anchor)=$O/anchor_qwen3-0-500.jsonl" "- steering=$O/abl_nosteer_qwen3-0-500.jsonl" "- penalty=$O/abl_nopen_qwen3-0-500.jsonl" \
  "- belief=$O/abl_nobelief_qwen3-0-500.jsonl" "Base (stored)=$Q/base-0-500.jsonl" "old Full=$Q/s20a1bl1p05-0-500.jsonl" > $O/abl_table.txt 2>> $O/pool.log
echo "$(date '+%m-%d %T') QWEN3_ABL_DONE ($O/abl_table.txt)" >> $O/pool.log
