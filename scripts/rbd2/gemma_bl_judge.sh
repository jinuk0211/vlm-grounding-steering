#!/bin/bash
# Judges the five Gemma THRONE baselines (flan-t5 base+large) once each has its 1000 rows, the job gemma_chain.sh did
# before it was stopped (20:49). The dev pick, ours on the test 1000 and the final table with ours are done by the other
# session's gemma_ours_fix.sh (it waits for these judge files); gemma_finish.sh, which duplicated that, is stopped.
# A tag is judged on GPU 5 or 3 with no judge job of this script and >= 12 GB free; a failed judge job is retried twice.
# Ends with a 2-judge table over every judged tag (ours too if gemma_ours_fix.sh has judged it). Log: logs/gemma.log,
# GEMMA_DONE (gemma_ours_fix.sh also stops waiting on it).
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gemma.log; TD=$O/runs/gemma_throne1000
alive(){ [ -n "$(ps -o pid= -p $(echo $@ | tr ' ' ,))" ]; }
pools(){ pgrep -f '^bash scripts/rbd2/gemma_(pool|split).sh' > /dev/null; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_THRONE_SET=throne_full1000 RBD2_N=1000
echo "=== $(date +%T) gemma_bl_judge.sh start (baseline judging; dev pick and ours: gemma_ours_fix.sh)" >> $LOG
judge(){ T=$1; U=$(uuid $2)
  RBD2_JUDGES=base $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $TD $T > $O/logs/gjudge-$T-base.log 2>&1 & JB=$!
  sleep 30
  RBD2_JUDGES=large $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $TD $T > $O/logs/gjudge-$T-large.log 2>&1
  wait $JB; }
judged(){ ls $TD/eval/$1/combined/*flan-t5-base.json $TD/eval/$1/combined/*flan-t5-large.json >/dev/null 2>&1; }
TODO=(base rudder_clone grains_clone revis vti); declare -A JOB JT TRY
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
P=$(head -1 $O/queues/gemma_ours.txt 2>/dev/null | awk '{print $4}')
{ echo "=== $(date +%T) gemma_throne1000 baselines, 2 judges (flan-t5 base+large, threshold 6), paired bootstrap vs Base (gemma_bench)"
  $L/venvs/throne/bin/python scripts/rbd2/pb_throne_partial.py $TD $(for T in base $P rudder_clone grains_clone revis vti; do judged $T && echo $T; done) 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GEMMA_DONE (baselines; the table with ours ends with GEMMA_OURS_DONE)"; } >> $LOG
