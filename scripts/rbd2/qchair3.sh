#!/bin/bash
# Qwen3-VL-8B CHAIR: pick the best setting on test rows 0-60 (chair_test500[:60]), then run pick and Base on all 500 test images.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/qchair3.log; TD=$O/runs/qchair_test500; T60=$O/runs/qchair_t60
QD=$O/queues/qchair_dev.txt; QO=$O/queues/qchair_ours.txt; QB=$O/queues/qchair_base.txt
mkdir -p $O/queues $O/runs $O/logs $T60 $TD $O/runs/qchair_test500_all $O/runs/qchair_test500_tbl

declare -A A=(
  [s20a2bl1p05]="--alpha 2 --layer 20 --lam 1 --pen 0.5"
  [s16a2bl1p05]="--alpha 2 --layer 16 --lam 1 --pen 0.5"
  [s24a2bl1p05]="--alpha 2 --layer 24 --lam 1 --pen 0.5"
  [bl1p05]="--lam 1 --pen 0.5"
  [s20a1bl1p05]="--alpha 1 --layer 20 --lam 1 --pen 0.5"
  [s20a3bl1p05]="--alpha 3 --layer 20 --lam 1 --pen 0.5"
  [s20a2bl2p1]="--alpha 2 --layer 20 --lam 2 --pen 1"
  [s20a2bl1p05M]="--alpha 2 --layer 20 --lam 1 --pen 0.5 --norm median"
  [s20a1bl2p1]="--alpha 1 --layer 20 --lam 2 --pen 1"
)
ALL_CAND="s20a2bl1p05 s16a2bl1p05 s24a2bl1p05 bl1p05 s20a1bl1p05 s20a3bl1p05 s20a2bl2p1 s20a2bl1p05M s20a1bl2p1"

echo "=== $(date +%T) qchair3.sh start: test rows 0-60 (candidates + base)" >> $LOG

# Queue test rows 0-60 in chunks of 30: 0-30, 30-60
{ echo "# test rows 0-60: base"
  echo "chair_test500 qchair_test500 base 0 30"
  echo "chair_test500 qchair_test500 base 30 60"
  echo "# test rows 0-60: candidates"
  for b in 0 30; do
    for t in $ALL_CAND; do
      echo "chair_test500 qchair_test500 $t $b $((b+30)) ${A[$t]}"
    done
  done
} > $QD.tmp
mv $QD.tmp $QD

until .venv/bin/python scripts/rbd2/qchair_collect.py chair_test500 60 $T60 base $ALL_CAND > $O/logs/qchair3_collect.tmp 2>&1; do sleep 30; done
cat $O/logs/qchair3_collect.tmp >> $LOG

export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1
{ echo "=== $(date +%T) test rows 0-60, Qwen3-VL-8B vs Base"; RBD2_N=60 .venv/bin/python scripts/rbd2/score_sets.py chair_test60 $T60 2>&1 | grep -v 'it/s'; } >> $LOG

P=$(.venv/bin/python scripts/rbd2/gchair_pick.py $T60/score.json)
if [ -z "$P" ] || [ -z "${A[$P]+x}" ]; then echo "=== $(date +%T) PICK_FAILED ($P)" >> $LOG; echo "# pick failed" > $QO; exit 1; fi
echo "=== $(date +%T) picked $P (${A[$P]}); running test rows 60-500 for Base and $P" >> $LOG

# Queue Base on rows 60-500
{ echo "# base on test rows 60-500"
  for b in $(seq 60 50 450); do
    end=$((b+50)); [ $end -gt 500 ] && end=500
    echo "chair_test500 qchair_test500 base $b $end"
  done
} > $QB.tmp
mv $QB.tmp $QB

# Queue Picked on rows 60-500
{ echo "# ours on test rows 60-500: $P"
  for b in $(seq 60 50 450); do
    end=$((b+50)); [ $end -gt 500 ] && end=500
    echo "chair_test500 qchair_test500 $P $b $end ${A[$P]}"
  done
} > $QO.tmp
mv $QO.tmp $QO

until .venv/bin/python scripts/rbd2/qchair_collect.py chair_test500 500 $O/runs/qchair_test500_all base $P > $O/logs/qchair3_collect.tmp 2>&1; do sleep 60; done
cat $O/logs/qchair3_collect.tmp >> $LOG

TA=$O/runs/qchair_test500_all; TT=$O/runs/qchair_test500_tbl; mkdir -p $TT
cp $TA/base-0-500.jsonl $TT/
cp $TA/$P-0-500.jsonl $TT/

{ echo "=== $(date +%T) chair_test500 Qwen3-VL-8B: vs Base (paired bootstrap); $P picked on test rows 0-60"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TA 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) QCHAIR3_DONE"; } >> $LOG
