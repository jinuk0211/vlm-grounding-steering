#!/bin/bash
# Takes over from gemma_chain.sh (stopped 20:5x). At 20:42 the chain declared DEV_FAILED: the dev unit bl2p1 had failed
# twice, both CUDA out-of-memory from model loads colliding with the other session's Gemma CHAIR workers on GPUs 3 and 5,
# so it skipped the pick. gemma_pool.sh no longer counts OOM failures (fail counts reset 20:44). This script does the
# chain's remaining steps unchanged: wait for the 5 dev decodes -> judges (flan-t5 base+large) -> gemma_pick.py (same rule)
# -> queues/gemma_ours.txt, which the workers take before the baselines; then join and judge every test tag once its 1000
# rows exist, and the final 2-judge table. Workers: gemma_pool.sh g5a, g5b (GPU 5) and g3 (GPU 3), started separately.
# Log: logs/gemma.log, GEMMA_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gemma.log; DEV=$O/runs/gemma_throne_dev60; TD=$O/runs/gemma_throne1000
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
pools(){ pgrep -f '^bash scripts/rbd2/gemma_pool.sh' > /dev/null; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
echo "=== $(date +%T) gemma_finish.sh start (takes over from gemma_chain.sh; dev pick redone, rule unchanged)" >> $LOG
rm -f $DEV/PICK_FAILED
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
  wait $JB
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
  if ! pools && [ ${#JOB[@]} -eq 0 ] && \
     [ -z "$(for T in ${TODO[@]}; do .venv/bin/python scripts/rbd2/gemma_join.py $T >/dev/null 2>&1 && echo y; done)" ]; then
    echo "=== $(date +%T) INCOMPLETE: ${TODO[*]}" >> $LOG; break; fi
  sleep 120
done
{ echo "=== $(date +%T) gemma_throne1000, 2 judges (flan-t5 base+large, threshold 6), paired bootstrap vs Base (gemma_bench)"
  $L/venvs/throne/bin/python scripts/rbd2/pb_throne_partial.py $TD $(for T in base $P rudder_clone grains_clone revis vti; do judged $T && echo $T; done) 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GEMMA_DONE"; } >> $LOG
