#!/bin/bash
# 11-03 17:2x user: after our runs, Gemma 4 E4B on the first 1000 full-THRONE images (throne_full1000, the images of the
# Qwen table): Base, REVIS, VTI (own implementation, as in the Gemma CHAIR/POPE table; user), RUDDER, GrAInS
# (gemma_bench.py) and our method (vdd_decode.py --model gemma, eager like the Gemma baselines). THRONE with flan-t5-base
# and -large only (user: no xl), pb_throne_partial.py, paired bootstrap vs Base.
# Our setting is picked on 60 dev images (throne_full rows 4940-4999, disjoint from the test 1000) before any test row of
# ours is decoded; rule in gemma_pick.py (user chose the dev pick).
# Smoke test first (2 dev images, 40 tokens: vdd base + s24a1bl2p1, each gemma_bench method), on GPU 5 once the Qwen
# full-THRONE workers there (pb_full1000_ours.sh o5a/o5b) exit; it also measures one Gemma worker's GPU memory, which
# becomes the workers' free-memory gate. SMOKE_FAILED stops everything.
# GPUs 5 and 3 only (user 18:2x: not GPU 4). Workers: GPU 5 x1 after the smoke test and x2 after the dev pick (so the dev
# judges have room), GPU 3 when the other session's AMBER worker there (m3) exits; every unit waits for enough free
# memory (other users share GPU 5).
# Judges: dev on the first of GPUs 5, 3 with 12 GB free; each test tag once its 1000 rows exist, on GPU 5 or 3 with no
# judge job and 12 GB free. Log: logs/gemma.log, GEMMA_DONE.
# usage: gemma_chain.sh
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gemma.log; DEV=$O/runs/gemma_throne_dev60; TD=$O/runs/gemma_throne1000
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
echo "=== $(date +%T) gemma_chain.sh start" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'PY' || { echo "=== $(date +%T) SET_FAILED" >> $LOG; exit 1; }
import json
from pathlib import Path
S=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/sets')
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
full=read(S/'throne_full.jsonl');dev=full[4940:5000];test={r['id'] for r in read(S/'throne_full1000.jsonl')}
assert len(dev)==60 and not test&{r['id'] for r in dev}
(S/'throne_gdev60.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dev))
print('throne_gdev60: throne_full rows 4940-4999, disjoint from throne_full1000')
PY
{ echo "# dev: Base and the four candidates of gemma_pick.py"
  echo "vdd throne_gdev60 gemma_throne_dev60 base 0 60"
  echo "vdd throne_gdev60 gemma_throne_dev60 s24a1bl2p1 0 60 --alpha 1 --layer 24 --lam 2 --pen 1"
  echo "vdd throne_gdev60 gemma_throne_dev60 s20a1bl2p1 0 60 --alpha 1 --layer 20 --lam 2 --pen 1"
  echo "vdd throne_gdev60 gemma_throne_dev60 s28a1bl2p1 0 60 --alpha 1 --layer 28 --lam 2 --pen 1"
  echo "vdd throne_gdev60 gemma_throne_dev60 bl2p1 0 60 --lam 2 --pen 1"; } > $O/queues/gemma_dev.txt
{ echo "# baselines on the test 1000, method by method so judging can start early"
  for m in base rudder_clone grains_clone revis vti; do for b in $(seq 0 100 900); do
    echo "bench throne_full1000 gemma_throne1000_parts $m $b $((b+100))"; done; done; } > $O/queues/gemma_bl.txt
# smoke test on GPU 5
while alive 2887433 2895291; do sleep 60; done
echo "=== $(date +%T) smoke test on GPU 5" >> $LOG
SM=$O/runs/gemma_smoke; rm -rf $SM; mkdir -p $SM
.venv/bin/python -c "
import json;r=[json.loads(l) for l in open('$O/sets/throne_gdev60.jsonl')][:2]
open('$O/sets/gsmoke2.jsonl','w').write(''.join(json.dumps({**x,'max_new_tokens':40})+'\n' for x in r))"
U5=$(uuid 5); B0=$(nvidia-smi --id=5 --query-gpu=memory.used --format=csv,noheader,nounits)
( while :; do nvidia-smi --id=5 --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > $SM/mem.txt & MS=$!
( export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          NLTK_DATA=$L/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src VDD_GRAY=$O/gray448.png CUDA_VISIBLE_DEVICES=$U5
  .venv/bin/python -u scripts/rbd2/vdd_decode.py --model gemma --attn eager --data $O/sets/gsmoke2.jsonl --output $SM --tag base --start 0 --end 2 &&
  .venv/bin/python -u scripts/rbd2/vdd_decode.py --model gemma --attn eager --data $O/sets/gsmoke2.jsonl --output $SM --tag s24a1bl2p1                    --start 0 --end 2 --alpha 1 --layer 24 --lam 2 --pen 1 &&
  for m in base revis vti rudder_clone grains_clone; do
    .venv/bin/python -u scripts/rbd2/gemma_bench.py --method $m --set $O/sets/gsmoke2.jsonl --output $SM --start 0 --end 1 || exit 1; done
) > $O/logs/gemma_smoke.log 2>&1; RC=$?
kill $MS
PEAK=$(sort -n $SM/mem.txt | tail -1); MF=$(( PEAK - B0 + 2500 )); [ $MF -lt 16000 ] && MF=16000; [ $MF -gt 22500 ] && MF=22500   # GPU 3 has 24 GB; other jobs on GPU 5 can inflate the delta
{ echo "smoke rc $RC, GPU 5 memory ${B0} -> peak ${PEAK} MiB; worker free-memory gate ${MF} MiB"
  for f in $SM/*.jsonl; do echo "--- $(basename $f)"; .venv/bin/python -c "
import json,sys
for r in map(json.loads,open(sys.argv[1])):print(r['generated_tokens'],r.get('elapsed_seconds',0)//1,repr(r['prediction'][:160]))" $f; done; } >> $LOG 2>&1
if [ $RC -ne 0 ] || [ $(cat $SM/*.jsonl | wc -l) -ne 9 ]; then echo "=== $(date +%T) SMOKE_FAILED (logs/gemma_smoke.log)" >> $LOG; exit 1; fi
echo "=== $(date +%T) smoke ok" >> $LOG
W=()
start(){ G=$1; N=$2; D=$3; shift 3
  ( while alive $@; do sleep 60; done; sleep $D; echo "=== $(date +%T) GPU $G worker $N joins" >> $LOG
    exec bash scripts/rbd2/gemma_pool.sh $G $N $MF ) & W+=($!); }
start 5 g5a 0 0
start 3 g3 0 3761973
# the second GPU 5 worker joins after the dev pick, so the dev judges have room on GPU 5
( until [ -f $O/queues/gemma_ours.txt ] || [ -f $DEV/PICK_FAILED ]; do sleep 60; done
  echo "=== $(date +%T) GPU 5 worker g5b joins" >> $LOG; exec bash scripts/rbd2/gemma_pool.sh 5 g5b $MF ) & W+=($!)
# dev: judge, pick, test queue for ours (if a dev unit fails twice: no pick, baselines only)
devfail(){ for f in $O/claims_gemma/fail-gemma_throne_dev60-*; do [ -f $f ] && [ $(wc -l < $f) -ge 2 ] && return 0; done; return 1; }
until [ $(cat $DEV/{base,s24a1bl2p1,s20a1bl2p1,s28a1bl2p1,bl2p1}-0-60.jsonl 2>/dev/null | wc -l) -ge 300 ] || devfail; do sleep 60; done
P=
if devfail; then echo "=== $(date +%T) DEV_FAILED: no pick, baselines only" >> $LOG; touch $DEV/PICK_FAILED
else
  echo "=== $(date +%T) dev decoded" >> $LOG
  export RBD2_THRONE_SET=throne_gdev60 RBD2_N=60
  DT="base s24a1bl2p1 s20a1bl2p1 s28a1bl2p1 bl2p1"
  until JG=$(for x in 5 3; do [ $(fre $x) -ge 12000 ] && echo $x && break; done); [ -n "$JG" ]; do sleep 60; done
  echo "=== $(date +%T) dev judges on GPU $JG" >> $LOG
  RBD2_JUDGES=base $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $(uuid $JG) $DEV $DT > $O/logs/gjudge-dev-base.log 2>&1 & JB=$!
  sleep 30
  RBD2_JUDGES=large $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $(uuid $JG) $DEV $DT > $O/logs/gjudge-dev-large.log 2>&1
  wait $JB   # not a bare wait: the worker subshells are children too
  { echo "=== $(date +%T) dev pick (2 judges)"; $L/venvs/throne/bin/python scripts/rbd2/gemma_pick.py 2>&1 | grep -v 'it/s'; } >> $LOG
  P=$(grep -o '^picked .*' $LOG | tail -1 | cut -d' ' -f2)
  [ -n "$P" ] || { echo "=== $(date +%T) PICK_FAILED: baselines only" >> $LOG; touch $DEV/PICK_FAILED; }
fi
# test: join and judge each tag once complete
export RBD2_THRONE_SET=throne_full1000 RBD2_N=1000
judge(){ T=$1; U=$(uuid $2)
  RBD2_JUDGES=base $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $TD $T > $O/logs/gjudge-$T-base.log 2>&1 & JB=$!
  sleep 30
  RBD2_JUDGES=large $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $TD $T > $O/logs/gjudge-$T-large.log 2>&1
  wait $JB; }
judged(){ ls $TD/eval/$1/combined/*flan-t5-base.json $TD/eval/$1/combined/*flan-t5-large.json >/dev/null 2>&1; }
# a tag is judged once its 1000 rows exist, on GPU 5 or 3 with no judge job and >= 12 GB free; a failed judge
# job is retried twice
TODO=($P base rudder_clone grains_clone revis vti); declare -A JOB JT TRY
while :; do
  for g in "${!JOB[@]}"; do alive ${JOB[$g]} && continue; T=${JT[$g]}; unset JOB[$g] JT[$g]
    if judged $T; then echo "=== $(date +%T) judged $T (GPU $g)" >> $LOG
    else TRY[$T]=$((${TRY[$T]:-0}+1)); echo "=== $(date +%T) judge $T FAILED on GPU $g (try ${TRY[$T]})" >> $LOG
      [ ${TRY[$T]} -lt 3 ] && TODO+=($T); fi
  done
  left=()
  for T in "${TODO[@]}"; do
    g=$(for x in 5 3; do [ -z "${JOB[$x]}" ] && [ $(fre $x) -ge 12000 ] && echo $x && break; done)
    if [ -n "$g" ] && .venv/bin/python scripts/rbd2/gemma_join.py $T >> $LOG 2>&1; then judge $T $g & JOB[$g]=$!; JT[$g]=$T
    else left+=($T); fi
  done
  TODO=("${left[@]}")
  [ ${#TODO[@]} -eq 0 ] && [ ${#JOB[@]} -eq 0 ] && break
  if ! alive ${W[@]} && [ ${#JOB[@]} -eq 0 ] &&      [ -z "$(for T in ${TODO[@]}; do .venv/bin/python scripts/rbd2/gemma_join.py $T >/dev/null 2>&1 && echo y; done)" ]; then
    echo "=== $(date +%T) INCOMPLETE: ${TODO[*]}" >> $LOG; break; fi
  sleep 120
done
{ echo "=== $(date +%T) gemma_throne1000, 2 judges (flan-t5 base+large, threshold 6), paired bootstrap vs Base (gemma_bench)"
  $L/venvs/throne/bin/python scripts/rbd2/pb_throne_partial.py $TD $(for T in base $P rudder_clone grains_clone revis vti; do judged $T && echo $T; done) 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GEMMA_DONE"; } >> $LOG
