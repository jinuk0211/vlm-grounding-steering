#!/bin/bash
# User (11-03 14:1x): VTI on CHAIR 500 for Qwen2.5-VL-7B, with a good setting. VTI = author code with the
# whitespace-normalised textual direction (vti_fix.py), FP16 eager as baseline_bench vti_author.
# 1. alpha_visual x alpha_textual grid on chair_dev60 (COCO train2014, disjoint from the 500 test images).
# 2. Rule, fixed before any dev result: among settings with dev recall >= Base dev recall - 2, lowest CHAIR_s, ties by
#    CHAIR_i; if none qualifies, the author default 0.9/0.9.
# 3. That one setting once on the 500 CHAIR test images, scored with score_sets.py (unmodified CHAIR scorer, paired
#    bootstrap) against the table's Base (fiveprofiles bf16-three-complete: CHAIR_s 25.6, CHAIR_i 8.75, recall 58.9).
# GPU 2 only (A5000, one FP16 model at a time). Log: logs/vti_chair.log
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
Q=$R/outputs/qwen25-vl7b-fiveprofiles-20261029-1793247386
LOG=$O/logs/vti_chair.log; DD=$O/runs/chair_dev60_vtiws; TD=$O/runs/chair_test500_vtiws
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src NLTK_DATA=$L/.cache/nltk_data \
  CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader) \
  RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json
mkdir -p $DD $TD; ln -sf $O/runs/chair_dev60/base-0-60.jsonl $DD/base-0-60.jsonl
for c in "0.9 0" "0 0.45" "0 0.9" "0 1.35" "0.9 0.45" "0.9 0.9" "0.9 1.35"; do set -- $c
  echo "=== $(date +%T) dev av $1 at $2" >> $LOG
  .venv/bin/python -u scripts/rbd2/vti_ws_run.py --set $O/sets/chair_dev60.jsonl --output $DD --start 0 --end 60 \
    --av $1 --at $2 --tag vtiws_v$1_t$2 < /dev/null >> $O/logs/vti_chair_dev.log 2>&1 || echo "=== FAILED dev $c" >> $LOG
done
{ echo "=== $(date +%T) chair_dev60 scores"; RBD2_N=60 .venv/bin/python scripts/rbd2/score_sets.py chair_dev60 $DD 2>&1 | grep -v 'it/s'; } >> $LOG
PICK=$(.venv/bin/python - $DD/score.json <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['results'];b=s['base'];c={t:v for t,v in s.items() if t.startswith('vtiws_')}
ok=[t for t,v in c.items() if v['recall']>=b['recall']-2]
print(min(ok,key=lambda t:(c[t]['CHAIR_s'],c[t]['CHAIR_i'])) if ok else 'vtiws_v0.9_t0.9')
PY
)
[ -n "$PICK" ] || { echo "=== $(date +%T) PICK_FAILED" >> $LOG; exit 1; }
AV=$(echo $PICK | cut -d_ -f2 | cut -c2-); AT=$(echo $PICK | cut -d_ -f3 | cut -c2-)
echo "=== $(date +%T) picked $PICK (av $AV at $AT); test 500" >> $LOG
cp $Q/bf16-three-complete/chair/base.jsonl $TD/base-0-500.jsonl
.venv/bin/python -u scripts/rbd2/vti_ws_run.py --set $O/sets/chair_test500.jsonl --output $TD --start 0 --end 500 \
  --av $AV --at $AT --tag $PICK < /dev/null >> $O/logs/vti_chair_test.log 2>&1 || { echo "=== $(date +%T) TEST_FAILED" >> $LOG; exit 1; }
{ echo "=== $(date +%T) chair_test500: table Base (BF16) vs $PICK"; RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TD 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) VTI_CHAIR_DONE"; } >> $LOG
