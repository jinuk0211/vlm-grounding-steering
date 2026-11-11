#!/bin/bash
# Score round-2 runs on project-b with the matching environment.
# usage: pb_score.sh SET [GPU_INDEX]      SET = chair_test60 | chair_test60_sweep | amber_test60 | amber_test60L | throne_test60
#   chair_*  shared .venv (unmodified CHAIR scorer; test state = data/hallucination-suite/chair-scorer-state.json)
#   amber_*  /home/project-b/LJW/venvs/amber (official AMBER inference.py, spaCy en_core_web_lg)
#   throne   /home/project-b/LJW/venvs/throne (THRONE @45badbb7, flan-t5 judges on GPU_INDEX)
set -euo pipefail
S=$1; G=${2:-}
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export RBD2_SETS=$O/sets RBD2_CHAIR_TEST_STATE=$R/data/hallucination-suite/chair-scorer-state.json AMBER_REPO=$R/data/amber/repo \
       BENCH=$L/bench NLTK_DATA=$L/.cache/nltk_data HF_HOME=$L/.cache/huggingface PYTHONDONTWRITEBYTECODE=1
case $S in
  chair_test60) PYTHONPATH=scripts:src .venv/bin/python scripts/rbd2/score_sets.py chair_test60 $O/runs/chair_test60;;
  chair_test60_sweep)   # the sweep shares base and main with chair_test60 (same decoding path)
    for t in base main; do [ -f $O/runs/chair_test60_sweep/$t-0-60.jsonl ] || cp $O/runs/chair_test60/$t-0-60.jsonl $O/runs/chair_test60_sweep/; done
    PYTHONPATH=scripts:src .venv/bin/python scripts/rbd2/score_sets.py chair_test60 $O/runs/chair_test60_sweep;;
  amber_test60|amber_test60L) $L/venvs/amber/bin/python scripts/rbd2/score_sets.py $S $O/runs/$S;;
  throne_test60) U=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
    $L/venvs/throne/bin/python scripts/rbd2/score_throne.py $U $O/runs/throne_test60;;
  *) echo "unknown set $S"; exit 2;;
esac
