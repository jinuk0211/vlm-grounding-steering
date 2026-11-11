#!/bin/bash
# User (11-04 12:4x): Gemma CHAIR - check several settings on the 60 dev images, then the 500 test images with the pick.
# The first test run (s24a2bl1p05, fixed without dev at the user's request on 11-03 20:4x) was already scored and seen:
# 30.8/6.0/61.9 vs vdd Base 34.6/7.0/64.7. This is a second test run, chosen on dev; both are reported.
# Dev: chair_dev60 (COCO train2014, disjoint from the test images) in units of 30; rows decoded on 11-03 are reused.
# Candidates: the four of 11-03 (s20/s24/s28 a2bl1p05, bl1p05), s24a1bl1p05 and s24a3bl1p05 (steering strength),
# s24a2bl2p1 (belief strength), s24a2bl1p05M (--norm median, which removed AMBER's Cover loss on Qwen) and s24a1bl2p1
# (the Gemma THRONE dev pick).
# Rule (gchair_pick.py, unchanged, fixed before these dev results): among candidates with dev recall >= vdd Base recall - 2,
# lowest CHAIR_s, ties by CHAIR_i; if none qualifies, highest F1. If the pick is s24a2bl1p05, its test rows already exist.
# Workers: gchair_pool.sh on GPU 2, GPU 3 and GPU 4 x2 (no other chain on these GPUs). Ours is compared with the existing
# vdd Base test rows (runs/gchair_test500/base-*) and shown next to the Gemma table rows.
# Log: logs/gchair2.log, GCHAIR2_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gchair2.log; DEV=$O/runs/gchair_dev60; C=$O/claims_gchair
QD=$O/queues/gchair_dev.txt; QO=$O/queues/gchair_ours.txt; QB=$O/queues/gchair_base.txt
GT=$R/outputs/gemma-grains-fp16-v2-20261029-1793246507/chair
declare -A A=( [base]="" [s24a2bl1p05]="--alpha 2 --layer 24 --lam 1 --pen 0.5" [s20a2bl1p05]="--alpha 2 --layer 20 --lam 1 --pen 0.5"
  [s28a2bl1p05]="--alpha 2 --layer 28 --lam 1 --pen 0.5" [bl1p05]="--lam 1 --pen 0.5" [s24a1bl1p05]="--alpha 1 --layer 24 --lam 1 --pen 0.5"
  [s24a3bl1p05]="--alpha 3 --layer 24 --lam 1 --pen 0.5" [s24a2bl2p1]="--alpha 2 --layer 24 --lam 2 --pen 1"
  [s24a2bl1p05M]="--alpha 2 --layer 24 --lam 1 --pen 0.5 --norm median" [s24a1bl2p1]="--alpha 1 --layer 24 --lam 2 --pen 1" )
DT="base s24a2bl1p05 s20a2bl1p05 s28a2bl1p05 bl1p05 s24a1bl1p05 s24a3bl1p05 s24a2bl2p1 s24a2bl1p05M s24a1bl2p1"
echo "=== $(date +%T) gchair2.sh start" >> $LOG
rm -f "${QO:?}" /home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/claims_gchair/fail-gchair_dev60-base-30-60 \
  /home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/claims_gchair/fail-gchair_dev60-bl1p05-0-30
{ echo "# dev (11-04): vdd Base + 9 candidates"
  for b in 0 30; do for t in $DT; do echo "chair_dev60 gchair_dev60 $t $b $((b+30)) ${A[$t]}"; done; done; } > $QD
echo "# no Base test units (done on 11-03)" > $QB
W=()
bash scripts/rbd2/gchair_pool.sh 2 c2 18879 & W+=($!)
bash scripts/rbd2/gchair_pool.sh 3 c3 18879 & W+=($!)
bash scripts/rbd2/gchair_pool.sh 4 c4a 18879 & W+=($!)
sleep 90
bash scripts/rbd2/gchair_pool.sh 4 c4b 18879 & W+=($!)
echo "=== $(date +%T) workers c2 c3 c4a c4b (GPUs 2, 3, 4, 4)" >> $LOG
devfail(){ for f in $C/fail-gchair_dev60-*; do [ -f $f ] && [ $(wc -l < $f) -ge 2 ] && return 0; done; return 1; }
devdone(){ for t in $DT; do for b in 0 30; do f=$DEV/$t-$b-$((b+30)).jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 30 ] || return 1; done; done; }
until devdone || devfail; do sleep 60; done
if devfail; then echo "=== $(date +%T) DEV_FAILED ($(ls $C | grep fail-gchair_dev60))" >> $LOG; touch $DEV/PICK_FAILED; wait ${W[@]}; exit 1; fi
DA=$O/runs/gchair2_dev60_all; mkdir -p $DA
for t in $DT; do cat $DEV/$t-0-30.jsonl $DEV/$t-30-60.jsonl > $DA/$t-0-60.jsonl; done
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1
{ echo "=== $(date +%T) chair_dev60, Gemma (vdd), vs vdd Base"; RBD2_N=60 .venv/bin/python scripts/rbd2/score_sets.py chair_dev60 $DA 2>&1 | grep -v 'it/s'; } >> $LOG
P=$(.venv/bin/python scripts/rbd2/gchair_pick.py $DA/score.json)
if [ -z "$P" ] || [ -z "${A[$P]+x}" ]; then echo "=== $(date +%T) PICK_FAILED ($P)" >> $LOG; touch $DEV/PICK_FAILED; wait ${W[@]}; exit 1; fi
if [ $P = s24a2bl1p05 ]; then
  echo "=== $(date +%T) picked $P = the setting already run on test; no new test decode" >> $LOG
  echo "# pick $P: test rows exist" > $QO
else
  echo "=== $(date +%T) picked $P (${A[$P]}); test 500" >> $LOG
  { echo "# ours on test: $P (dev pick 11-04)"; for b in $(seq 0 50 450); do echo "chair_test500 gchair_test500 $P $b $((b+50)) ${A[$P]}"; done; } > $QO.tmp
  mv $QO.tmp $QO
fi
wait ${W[@]}
echo "=== $(date +%T) decoded" >> $LOG
TA=$O/runs/gchair_test500_all; TT=$O/runs/gchair_test500_tbl
[ $P = s24a2bl1p05 ] || .venv/bin/python scripts/rbd2/gchair_join.py $P >> $LOG 2>&1
[ -f $TA/$P-0-500.jsonl ] && cp $TA/$P-0-500.jsonl $TT/$P-0-500.jsonl
{ echo "=== $(date +%T) chair_test500 Gemma: vs vdd Base (paired bootstrap); s24a2bl1p05 = first test run, $P = dev pick"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TA 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) chair_test500 Gemma: table rows (base = table Base) + vddbase + ours"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TT 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GCHAIR2_DONE"; } >> $LOG
