#!/bin/bash
# After vti_ws_run.py (PID $1) ends: THRONE judges (pb_throne_judge.py, unchanged) for vti_author_ws on GPU 2, then wait
# for the three judge files of base in runs/throne_full1000, copy them, and score base vs vti_author_ws (score_throne.py,
# paired bootstrap). Separate run dir so the main full-THRONE scoring never judges the same files. Log: logs/score_vtiws.log
# usage: vti_ws_score.sh DECODE_PID
P=$1
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=1000 RBD2_THRONE_SET=throne_full1000
LOG=$O/logs/score_vtiws.log; TD=$O/runs/throne_full1000_vtiws; SRC=$O/runs/throne_full1000
while [ -n "$(ps -o pid= -p $P)" ]; do sleep 60; done
[ $(wc -l < $TD/vti_author_ws-0-1000.jsonl) -ge 1000 ] || { echo "=== $(date +%T) DECODE_INCOMPLETE" >> $LOG; exit 1; }
U=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
echo "=== $(date +%T) judge vti_author_ws on GPU 2" >> $LOG
$L/venvs/throne/bin/python scripts/rbd2/pb_throne_judge.py $U $TD vti_author_ws 2>&1 | grep -v 'it/s' >> $LOG
[ $(ls $TD/eval/vti_author_ws/combined/*.json 2>/dev/null | wc -l) -ge 3 ] || { echo "=== $(date +%T) JUDGE_FAILED" >> $LOG; exit 1; }
B=$SRC/eval/base/combined
until [ $(ls $B/*.json 2>/dev/null | wc -l) -ge 3 ]; do sleep 120; done
mkdir -p $TD/eval/base; cp $SRC/eval/base/responses.json $TD/eval/base/; cp -r $B $TD/eval/base/
{ echo "=== $(date +%T) throne_full1000: base vs vti_author_ws"
  $L/venvs/throne/bin/python scripts/rbd2/score_throne.py $U $TD 2>&1 | grep -v 'it/s' | tail -6
  echo "=== $(date +%T) VTIWS_DONE"; } >> $LOG
