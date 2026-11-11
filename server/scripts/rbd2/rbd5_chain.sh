#!/bin/bash
# User (11-05 17:5x): keep RBD, raise CHAIR recall. Analysis (scripts/rbd2/analysis/rbd_recall_analysis*.py): RBD loses secondary, late
# objects of classes the model rarely hallucinates, while nearly all its net hallucination gain is in low-precision classes.
# Candidates, same RBD settings as each model's table row, only V_obj changes (rbd5_prep.py):
#   gate80    high-prior classes out of V_obj
#   cfgate80  the same on the colour-fixed V_obj of THRONE v2.1 (no suppression of the colour word "orange")
# CHAIR 500 test, third or later test attempt on these images (disclose). Compared with vdd Base and with the current RBD row, with and
# without class orange. Workers start only when the Qwen3 baselines on their GPU have ended; nothing running is touched.
# Order: Qwen3, Gemma, Qwen2.5. Log: logs/rbd5.log, RBD5_<model>_DONE per model, RBD5_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/rbd5.log; Q=$O/queues/rbd5.txt; V=$O/vdd5
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
echo "=== $(date +%T) rbd5_chain.sh start" >> $LOG
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
  .venv/bin/python scripts/rbd2/rbd5_prep.py >> $LOG 2>&1 || { echo "=== $(date +%T) PREP_FAILED" >> $LOG; exit 1; }
{ echo "# MODEL ATTN SET OUTDIR TAG START END args"
  for spec in "qwen3 sdpa rbd5_qwen3 --alpha 1 --layer 20 --lam 1 --pen 0.5" "gemma eager rbd5_gemma --alpha 1 --layer 24 --lam 1 --pen 0.5" \
              "qwen sdpa rbd5_qwen25 --alpha 2 --layer 16 --lam 1 --pen 0.5"; do set -- $spec; M=$1; AT=$2; OD=$3; shift 3
    for b in $(seq 0 50 450); do for t in gate80 cfgate80; do
      echo "$M $AT chair_test500 $OD $t $b $((b+50)) $* --vobj $V/$M-$t.json"; done; done; done; } > $Q
echo "=== $(date +%T) queue $(grep -vc '^#' $Q) units" >> $LOG
start(){ G=$1; N=$2; D=$3; shift 3
  ( while alive $@; do sleep 30; done; sleep $D; echo "=== $(date +%T) GPU $G worker $N joins" >> $LOG
    exec bash scripts/rbd2/rbd5_pool.sh $G $N $Q 18900 ) & }
start 2 r2 0 3877612 3877590
start 3 r3 0 3877614 3877591
start 4 r4a 0 3750721 3750746 3795174
start 4 r4b 150 3750721 3750746 3795174
start 5 r5a 0 3750722 3750744 3795175
start 5 r5b 150 3750722 3750744 3795175
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
for spec in "rbd5_qwen3 qchair_test500_all s20a1bl1p05" "rbd5_gemma gchair_test500_all s24a1bl1p05" "rbd5_qwen25 chair_test500_all s16a2bl1p05"; do set -- $spec; OD=$1; SRC=$O/runs/$2; RB=$3; A=$O/runs/${OD}_all; mkdir -p $A
  until .venv/bin/python scripts/rbd2/rbd5_score.py join $OD > $O/logs/rbd5_join.tmp 2>&1; do sleep 120; done
  cat $O/logs/rbd5_join.tmp >> $LOG
  cp $SRC/base-0-500.jsonl $A/base-0-500.jsonl; cp $SRC/$RB-0-500.jsonl $A/rbd-0-500.jsonl
  { echo "=== $(date +%T) $OD CHAIR 500 vs vdd Base (paired bootstrap); rbd = current table row"
    CUDA_VISIBLE_DEVICES= RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $A 2>&1 | grep -v 'it/s'
    echo "--- vs the RBD row, and without class orange"
    CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/rbd2/rbd5_score.py score $A 2>&1 | grep -v 'it/s'
    echo "=== $(date +%T) RBD5_${OD}_DONE"; } >> $LOG
done
wait
echo "=== $(date +%T) RBD5_DONE" >> $LOG
