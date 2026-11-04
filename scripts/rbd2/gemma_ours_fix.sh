#!/bin/bash
# Our method back into the Gemma THRONE run. gemma_chain.sh declared DEV_FAILED at 20:42 because the dev unit bl2p1 hit
# OOM twice when the other session's CHAIR workers collided with it (not a real failure; gemma_pool.sh no longer counts
# OOM since 20:44), so it went on with the baselines only. This script does the steps gemma_chain.sh skipped, unchanged:
# wait for the five dev tags (60 rows each) -> dev judges flan-t5 base + large -> gemma_pick.py (same rule; writes
# queues/gemma_ours.txt, which the gemma_pool.sh workers take before the baseline queue) -> ours on the test 1000 ->
# gemma_join.py -> judges -> once every baseline tag is judged too, pb_throne_partial.py over all tags (2 judges,
# threshold 6, paired bootstrap vs Base). Judges run on a GPU with >= 12 GB free (4, 2, 5, 3 in that order).
# Log: logs/gemma_ours_fix.log (the final table is also appended to logs/gemma.log), GEMMA_OURS_DONE.
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/gemma_ours_fix.log; DEV=$O/runs/gemma_throne_dev60; TD=$O/runs/gemma_throne1000
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
jgpu(){ until G=$(for x in 4 2 5 3; do [ $(fre $x) -ge 12000 ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done; echo $G; }
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
echo "=== $(date +%T) gemma_ours_fix.sh start" >> $LOG
DT="base s24a1bl2p1 s20a1bl2p1 s28a1bl2p1 bl2p1"
until [ $(cat $DEV/{base,s24a1bl2p1,s20a1bl2p1,s28a1bl2p1,bl2p1}-0-60.jsonl 2>/dev/null | wc -l) -ge 300 ]; do sleep 60; done
JG=$(jgpu); echo "=== $(date +%T) dev decoded; dev judges on GPU $JG" >> $LOG
export RBD2_THRONE_SET=throne_gdev60 RBD2_N=60
RBD2_JUDGES=base $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $(uuid $JG) $DEV $DT > $O/logs/gjudge-dev-base.log 2>&1 & JB=$!
sleep 30
RBD2_JUDGES=large $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $(uuid $JG) $DEV $DT > $O/logs/gjudge-dev-large.log 2>&1
wait $JB
{ echo "=== $(date +%T) dev pick (2 judges)"; $L/venvs/throne/bin/python scripts/rbd2/gemma_pick.py 2>&1 | grep -v 'it/s'; } >> $LOG
P=$(grep -o '^picked .*' $LOG | tail -1 | cut -d' ' -f2)
[ -n "$P" ] && [ -f $O/queues/gemma_ours.txt ] || { echo "=== $(date +%T) PICK_FAILED again (see above)" >> $LOG; exit 1; }
rm -f /home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/runs/gemma_throne_dev60/PICK_FAILED
echo "=== $(date +%T) picked $P; queues/gemma_ours.txt written, PICK_FAILED removed" >> $LOG
export RBD2_THRONE_SET=throne_full1000 RBD2_N=1000
until .venv/bin/python scripts/rbd2/gemma_join.py $P >> $LOG 2>&1; do sleep 120; done
JG=$(jgpu); echo "=== $(date +%T) $P 1000 rows joined; judges on GPU $JG" >> $LOG
RBD2_JUDGES=base $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $(uuid $JG) $TD $P > $O/logs/gjudge-$P-base.log 2>&1 & JB=$!
sleep 30
RBD2_JUDGES=large $L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $(uuid $JG) $TD $P > $O/logs/gjudge-$P-large.log 2>&1
wait $JB
judged(){ ls $TD/eval/$1/combined/*flan-t5-base.json $TD/eval/$1/combined/*flan-t5-large.json >/dev/null 2>&1; }
judged $P || { echo "=== $(date +%T) judge $P FAILED (logs/gjudge-$P-*.log)" >> $LOG; exit 1; }
echo "=== $(date +%T) judged $P; waiting for the baseline tags" >> $LOG
until ( for T in base rudder_clone grains_clone revis vti; do judged $T || exit 1; done ) || grep -q 'GEMMA_DONE' $O/logs/gemma.log; do sleep 300; done
M=$(for T in base $P rudder_clone grains_clone revis vti; do judged $T && echo $T; done)
{ echo "=== $(date +%T) gemma_throne1000 with ours ($P), 2 judges (flan-t5 base+large, threshold 6), paired bootstrap vs Base (gemma_bench)"
  $L/venvs/throne/bin/python scripts/rbd2/pb_throne_partial.py $TD $M 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) GEMMA_OURS_DONE"; } > $O/logs/gemma_ours_final.txt
cat $O/logs/gemma_ours_final.txt >> $LOG; cat $O/logs/gemma_ours_final.txt >> $O/logs/gemma.log
