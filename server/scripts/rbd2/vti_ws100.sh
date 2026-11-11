#!/bin/bash
# User (11-03 12:3x): VTI on full THRONE only up to 100 images. Stop vti_ws_run.py (PID $1) and its 1000-image scorer
# (PID $2) once 100 rows exist, then score base vs vti_author_ws on the first 100 images of full THRONE with THRONE's
# judges (score_throne.py, unchanged) on GPU 2. Log: logs/score_vtiws100.log
# usage: vti_ws100.sh DECODE_PID SCORER_PID
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/score_vtiws100.log; SRC=$O/runs/throne_full1000; T1=$O/runs/throne_full1000_vtiws; TD=$O/runs/throne_full100_vtiws
kill $2 2>/dev/null
until [ $(wc -l < $T1/vti_author_ws-0-1000.jsonl) -ge 100 ] || [ -z "$(ps -o pid= -p $1)" ]; do sleep 10; done
kill $1 2>/dev/null; while [ -n "$(ps -o pid= -p $1)" ]; do sleep 2; done
echo "=== $(date +%T) decode stopped at $(wc -l < $T1/vti_author_ws-0-1000.jsonl) rows" >> $LOG
head -100 $O/sets/throne_full1000.jsonl > $O/sets/throne_full100.jsonl
mkdir -p $TD; head -100 $T1/vti_author_ws-0-1000.jsonl > $TD/vti_author_ws-0-100.jsonl; head -100 $SRC/base-0-1000.jsonl > $TD/base-0-100.jsonl
[ $(wc -l < $TD/vti_author_ws-0-100.jsonl) -eq 100 ] || { echo "=== VTI_ROWS_SHORT" >> $LOG; exit 1; }
export RBD2_SETS=$O/sets BENCH=$L/bench HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1 RBD2_N=100 RBD2_THRONE_SET=throne_full100
U=$(nvidia-smi --id=2 --query-gpu=uuid --format=csv,noheader)
{ echo "=== $(date +%T) throne_full100: base vs vti_author_ws (judges on GPU 2)"
  $L/venvs/throne/bin/python scripts/rbd2/score_throne.py $U $TD 2>&1 | grep -v 'it/s' | tail -6
  echo "=== $(date +%T) VTIWS100_DONE"; } >> $LOG
