#!/bin/bash
# User 11-09: THRONE on the other 4,000 of its 5,000 COCO val2017 images, every row of tab:throne_full (7 per backbone) and the evaluation,
# on GPUs 0 and 5 only (both A6000). t4k_prep.py: set throne_r4000 (rows 1000-4999 of throne_full) and the queues; then per GPU two decode slots
# (repro units first: 2 stored A6000 rows per method; then the block-major decode queue; then any judge) and one light judge slot (flan-t5 base and
# large on finished blocks while the decoders run). When the repro units are done, t4k_prep.py repro compares them with the stored captions; a
# difference stops every worker (t4k/STOP) after its current unit. At the end: join -> t4k_score.py -> logs/t4k_table.txt, logs/t4k_score.json.
# Log logs/t4k.log (T5K_DONE / T5K_STOPPED).
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/t4k.log; PY=.venv/bin/python; TPY=$L/venvs/throne/bin/python
export TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data; mkdir -p $TMPDIR $O/t4k
CPU="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="
log(){ echo "=== $(date '+%m-%d %T') $*" >> $LOG; }
log "t4k_chain.sh start (pid $$)"
$CPU $PY scripts/rbd2/t4k_prep.py sets >> $LOG 2>&1 && $CPU $PY scripts/rbd2/t4k_prep.py queues >> $LOG 2>&1 || { log "T5K_STOPPED: prep failed"; exit 1; }
[ -f $O/t4k/jgate ] || printf 'base 5000\nlarge 7000\nxl 16000\n' > $O/t4k/jgate
[ -f $O/t4k/light ] || echo 'base large' > $O/t4k/light
dslot(){ G=$1; W=$2
  bash scripts/rbd2/t4k_pool.sh $G $W $O/queues/t4k_repro.txt
  bash scripts/rbd2/t4k_pool.sh $G $W $O/queues/t4k_decode.txt; echo $W >> $O/t4k/decode_left
  bash scripts/rbd2/t4k_jpool.sh $G $W all; }
PIDS=()
for spec in "0 d0a" "5 d5a" "0 d0b" "5 d5b"; do set -- $spec; dslot $1 $2 & PIDS+=($!); sleep 45; done
for spec in "0 j0" "5 j5"; do set -- $spec; bash scripts/rbd2/t4k_jpool.sh $1 $2 light & PIDS+=($!); done
log "slots: decode d0a d0b (GPU 0), d5a d5b (GPU 5); light judges j0, j5; pids ${PIDS[*]}"
# repro: wait until every repro unit is complete or failed twice
until $CPU $PY - <<'EOF'
import sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');ok=True
for l in open(O/'queues/t4k_repro.txt'):
    k,d,od,t,s,e,*_=l.split();f=O/'runs'/od/f'{t}-{s}-{e}.jsonl';u=f'{od}-{t}-{s}-{e}'
    fails=(O/'claims_t4k'/f'fail-{u}');n=len(fails.read_text().split()) if fails.exists() else 0
    if not ((f.exists() and len(f.read_text().splitlines())>=int(e)-int(s)) or n>=2):ok=False
sys.exit(0 if ok else 1)
EOF
do sleep 60; done
if $CPU $PY scripts/rbd2/t4k_prep.py repro >> $LOG 2>&1; then log "repro identical on every method"
else touch $O/t4k/STOP; log "REPRO differs (above): t4k/STOP written, workers leave after their current unit"; fi
wait "${PIDS[@]}"
[ -f $O/t4k/STOP ] && { log "T5K_STOPPED (t4k/STOP)"; exit 1; }
log "decode and judging ended: $(cat $O/claims_t4k/fail-* $O/claims_t4kj/fail-* 2>/dev/null | wc -l) failure marks"
$CPU $PY scripts/rbd2/t4k_prep.py join >> $LOG 2>&1 || log "join incomplete (above)"
BENCH=$L/bench $CPU $TPY scripts/rbd2/t4k_score.py --json $O/logs/t4k_score.json > $O/logs/t4k_table.txt 2>> $LOG && log "table: logs/t4k_table.txt" || log "scoring failed"
log T5K_DONE
