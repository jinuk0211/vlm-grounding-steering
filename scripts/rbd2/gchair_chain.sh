#!/bin/bash
# User (11-03 18:4x, 19:1x): Gemma 4 E4B CHAIR with our method first, on the free GPUs: 2 and 3 now, GPU 5 (x2) once the
# Qwen full-THRONE run there (pb_full1000_ours.sh, pid 2887430) ends; not GPU 4 (user). Test = chair_test500: the 500
# images, prompt and 512 tokens of the Gemma CHAIR table (outputs/gemma-grains-fp16-v2-20261029-1793246507/chair).
# Our setting is picked on chair_dev60 (COCO train2014, disjoint from the test images) before any test row of ours is
# decoded. Candidates: the Qwen CHAIR setting (alpha 2, lam 1, pen 0.5) at Gemma layers 20/24/28 (gemma_chain.sh moves the
# THRONE setting the same way) and the belief term alone (bl1p05). Rule, fixed before any dev result: among candidates
# with dev recall >= vdd Base recall - 2, lowest CHAIR_s, ties by CHAIR_i; if none qualifies, highest F1
# (P = 100 - CHAIR_i, R = recall).
# Base and ours both through vdd_decode.py --model gemma --attn eager (ours is compared with its own vdd Base, as for Qwen),
# then scored twice: ours vs vdd Base, and ours next to the Gemma table rows (base, revis, rudder_clone, grains_clone, vti).
# Smoke test first on GPU 2 (1 dev image, 512 tokens, vdd base + s24a2bl1p05); its peak memory sets the workers' gate.
# Log: logs/gchair.log, GCHAIR_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gchair.log; DEV=$O/runs/gchair_dev60; TD=$O/runs/gchair_test500
QD=$O/queues/gchair_dev.txt; QB=$O/queues/gchair_base.txt; QO=$O/queues/gchair_ours.txt
GT=$R/outputs/gemma-grains-fp16-v2-20261029-1793246507/chair
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
echo "=== $(date +%T) gchair_chain.sh start" >> $LOG
rm -f "${QO:?}"; mkdir -p $DEV $TD
.venv/bin/python - >> $LOG 2>&1 <<'PY' || { echo "=== $(date +%T) SET_FAILED" >> $LOG; exit 1; }
import json
from pathlib import Path
R=Path('/home/project-b/LJW/vlm-grounding-steering');S=R/'outputs/rbd2-20261102/sets'
G=R/'outputs/gemma-grains-fp16-v2-20261029-1793246507/chair'
rd=lambda p:[json.loads(l) for l in open(p) if l.strip()]
t=rd(S/'chair_test500.jsonl');d=rd(S/'chair_dev60.jsonl');b=rd(G/'base.jsonl')
assert len(t)==500 and len(d)==60 and {r['id'] for r in t}=={r['id'] for r in b}
assert all(r['prompt']==t[0]['prompt'] and r['max_new_tokens']==512 for r in t+d+b)
assert not {Path(r['image']).name for r in d}&{Path(r['image']).name for r in t}
(S/'gchair_smoke1.jsonl').write_text(json.dumps(d[0])+'\n')
print('sets ok: chair_test500 = Gemma table CHAIR ids/prompt/512 tokens; chair_dev60 disjoint;',repr(t[0]['prompt']))
PY
# smoke test on GPU 2
SM=$O/runs/gchair_smoke; rm -rf "${SM:?}"; mkdir -p $SM
B0=$(nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits)
( while :; do nvidia-smi --id=2 --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $SM/mem.txt & MS=$!
( export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src VDD_GRAY=$O/gray448.png \
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
  .venv/bin/python -u scripts/rbd2/vdd_decode.py --model gemma --attn eager --data $O/sets/gchair_smoke1.jsonl --output $SM \
    --tag base --start 0 --end 1 &&
  .venv/bin/python -u scripts/rbd2/vdd_decode.py --model gemma --attn eager --data $O/sets/gchair_smoke1.jsonl --output $SM \
    --tag s24a2bl1p05 --start 0 --end 1 --alpha 2 --layer 24 --lam 1 --pen 0.5 ) > $O/logs/gchair_smoke.log 2>&1; RC=$?
kill $MS
PEAK=$(sort -n $SM/mem.txt | tail -1); MF=$(( PEAK - B0 + 1500 )); [ $MF -lt 16000 ] && MF=16000
{ echo "smoke rc $RC, GPU 2 memory $B0 -> peak $PEAK MiB; worker gate $MF MiB"
  for f in $SM/*-0-1.jsonl; do echo "--- $(basename $f)"; .venv/bin/python -c "
import json,sys
for r in map(json.loads,open(sys.argv[1])):print(r['generated_tokens'],round(r.get('elapsed_seconds',0)),repr(r['prediction'][:160]))" $f; done; } >> $LOG 2>&1
if [ $RC -ne 0 ] || [ $(cat $SM/*-0-1.jsonl | wc -l) -ne 2 ]; then echo "=== $(date +%T) SMOKE_FAILED (logs/gchair_smoke.log)" >> $LOG; exit 1; fi
echo "=== $(date +%T) smoke ok" >> $LOG
DT="base s24a2bl1p05 s20a2bl1p05 s28a2bl1p05 bl1p05"
{ echo "# dev: vdd Base and the four candidates"
  for b in 0 30; do e=$((b+30)); echo "chair_dev60 gchair_dev60 base $b $e"
    for s in 24 20 28; do echo "chair_dev60 gchair_dev60 s${s}a2bl1p05 $b $e --alpha 2 --layer $s --lam 1 --pen 0.5"; done
    echo "chair_dev60 gchair_dev60 bl1p05 $b $e --lam 1 --pen 0.5"; done; } > $QD
{ echo "# test: vdd Base"; for b in $(seq 0 50 450); do echo "chair_test500 gchair_test500 base $b $((b+50))"; done; } > $QB
W=()
start(){ G=$1; N=$2; D=$3; shift 3
  ( while alive $@; do sleep 60; done; sleep $D; echo "=== $(date +%T) GPU $G worker $N joins" >> $LOG
    exec bash scripts/rbd2/gchair_pool.sh $G $N $MF ) & W+=($!); }
if [ $MF -le 23500 ]; then start 2 c2 0 0; start 3 c3 0 0
else echo "=== $(date +%T) gate $MF MiB > 24 GB GPUs: GPU 5 only" >> $LOG; fi
start 5 c5a 0 2887430
start 5 c5b 150 2887430
# dev pick
devfail(){ for f in $O/claims_gchair/fail-gchair_dev60-*; do [ -f $f ] && [ $(wc -l < $f) -ge 2 ] && return 0; done; return 1; }
devdone(){ for t in $DT; do for b in 0 30; do f=$DEV/$t-$b-$((b+30)).jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 30 ] || return 1; done; done; }
until devdone || devfail; do sleep 60; done
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1
P=
if devfail; then echo "=== $(date +%T) DEV_FAILED: no pick" >> $LOG
else
  DA=$O/runs/gchair_dev60_all; mkdir -p $DA
  for t in $DT; do cat $DEV/$t-0-30.jsonl $DEV/$t-30-60.jsonl > $DA/$t-0-60.jsonl; done
  { echo "=== $(date +%T) chair_dev60, Gemma (vdd)"; RBD2_N=60 .venv/bin/python scripts/rbd2/score_sets.py chair_dev60 $DA 2>&1 | grep -v 'it/s'; } >> $LOG
  P=$(.venv/bin/python scripts/rbd2/gchair_pick.py $DA/score.json)
  case $P in s??a2bl1p05) A="--alpha 2 --layer ${P:1:2} --lam 1 --pen 0.5";; bl1p05) A="--lam 1 --pen 0.5";; *) P=;; esac
fi
if [ -n "$P" ]; then
  echo "=== $(date +%T) picked $P ($A); test 500" >> $LOG
  { echo "# ours on test: $P (dev pick)"; for b in $(seq 0 50 450); do echo "chair_test500 gchair_test500 $P $b $((b+50)) $A"; done; } > $QO.tmp
  mv $QO.tmp $QO
else echo "=== $(date +%T) PICK_FAILED: vdd Base only" >> $LOG; touch $DEV/PICK_FAILED; fi
wait ${W[@]}
echo "=== $(date +%T) decoded" >> $LOG
TA=$O/runs/gchair_test500_all; TT=$O/runs/gchair_test500_tbl; mkdir -p $TA $TT
.venv/bin/python scripts/rbd2/gchair_join.py base $P >> $LOG 2>&1
for m in base revis rudder_clone grains_clone vti; do cp $GT/$m.jsonl $TT/$m-0-500.jsonl; done
[ -f $TA/base-0-500.jsonl ] && cp $TA/base-0-500.jsonl $TT/vddbase-0-500.jsonl
[ -n "$P" ] && [ -f $TA/$P-0-500.jsonl ] && cp $TA/$P-0-500.jsonl $TT/$P-0-500.jsonl
{ echo "=== $(date +%T) chair_test500 Gemma: ours vs vdd Base (paired bootstrap)"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TA 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) chair_test500 Gemma: table rows (base = table Base) + vddbase + ours"
  RBD2_N=500 .venv/bin/python scripts/rbd2/score_sets.py chair_test500 $TT 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GCHAIR_DONE"; } >> $LOG
