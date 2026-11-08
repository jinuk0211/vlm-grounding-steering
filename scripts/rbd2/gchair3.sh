#!/bin/bash
# User (11-04 13:0x): Gemma CHAIR - pick the setting on the first 60 test images (chair_test500[:60] = chair_test60), not
# on dev, then run the pick on all 500 test images. Replaces gchair2.sh (dev sweep, stopped).
# Candidates on test rows 0-60: the 8 settings of gchair2.sh besides s24a2bl1p05, whose 500 test rows (and the vdd Base's)
# already exist. Rule as gchair_pick.py: among candidates with recall >= vdd Base recall - 2 on these 60, lowest CHAIR_s,
# ties by CHAIR_i; if none qualifies, highest F1. The pick then runs on test rows 60-500.
# Workers: gchair_pool.sh c2, c3, c4a, c4b (GPUs 2, 3, 4, 4), already running. Log: logs/gchair3.log, GCHAIR3_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gchair3.log; TD=$O/runs/gchair_test500; T60=$O/runs/gchair_t60
QD=$O/queues/gchair_dev.txt; QO=$O/queues/gchair_ours.txt
declare -A A=( [s24a2bl1p05]="--alpha 2 --layer 24 --lam 1 --pen 0.5" [s20a2bl1p05]="--alpha 2 --layer 20 --lam 1 --pen 0.5"
  [s28a2bl1p05]="--alpha 2 --layer 28 --lam 1 --pen 0.5" [bl1p05]="--lam 1 --pen 0.5" [s24a1bl1p05]="--alpha 1 --layer 24 --lam 1 --pen 0.5"
  [s24a3bl1p05]="--alpha 3 --layer 24 --lam 1 --pen 0.5" [s24a2bl2p1]="--alpha 2 --layer 24 --lam 2 --pen 1"
  [s24a2bl1p05M]="--alpha 2 --layer 24 --lam 1 --pen 0.5 --norm median" [s24a1bl2p1]="--alpha 1 --layer 24 --lam 2 --pen 1" )
NEW="s20a2bl1p05 s28a2bl1p05 bl1p05 s24a1bl1p05 s24a3bl1p05 s24a2bl2p1 s24a2bl1p05M s24a1bl2p1"
echo "=== $(date +%T) gchair3.sh start: candidates on test rows 0-60" >> $LOG
until .venv/bin/python scripts/rbd2/gchair_collect.py chair_test500 60 $T60 base s24a2bl1p05 $NEW > $O/logs/gchair3_collect.tmp 2>&1; do sleep 60; done
cat $O/logs/gchair3_collect.tmp >> $LOG
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1
{ echo "=== $(date +%T) test rows 0-60, Gemma (vdd), vs vdd Base"; RBD2_N=60 .venv/bin/python scripts/rbd2/score_sets.py chair_test60 $T60 2>&1 | grep -v 'it/s'; } >> $LOG
P=$(.venv/bin/python scripts/rbd2/gchair_pick.py $T60/score.json)
if [ -z "$P" ] || [ -z "${A[$P]+x}" ]; then echo "=== $(date +%T) PICK_FAILED ($P)" >> $LOG; echo "# pick failed" > $QO; exit 1; fi
if [ $P = s24a2bl1p05 ]; then echo "=== $(date +%T) picked $P: its 500 test rows exist" >> $LOG; echo "# pick $P: rows exist" > $QO
else
  echo "=== $(date +%T) picked $P (${A[$P]}); test rows 60-500" >> $LOG
  { echo "# ours on test rows 60-500: $P (picked on test rows 0-60)"; echo "chair_test500 gchair_test500 $P 60 100 ${A[$P]}"
    for b in $(seq 100 50 450); do echo "chair_test500 gchair_test500 $P $b $((b+50)) ${A[$P]}"; done; } > $QO.tmp
  mv $QO.tmp $QO
fi
until .venv/bin/python scripts/rbd2/gchair_collect.py chair_test500 500 $O/runs/gchair_test500_all $P > $O/logs/gchair3_collect.tmp 2>&1; do sleep 120; done
cat $O/logs/gchair3_collect.tmp >> $LOG
TA=$O/runs/gchair_test500_all; TT=$O/runs/gchair_test500_tbl; cp $TA/$P-0-500.jsonl $TT/$P-0-500.jsonl
{ echo "=== $(date +%T) chair_test500 Gemma: vs vdd Base (paired bootstrap); $P picked on test rows 0-60"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TA 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) chair_test500 Gemma: table rows (base = table Base) + vddbase + ours"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TT 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GCHAIR3_DONE"; } >> $LOG
