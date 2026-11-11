#!/bin/bash
# RBD AMBER vdd3 (user, via session ccf836b1 "논문 업데이트 amber": run after the Gemma CHAIR chain and the logit-lens run).
# Plan and fixed protocol: amber_v3_runbook.md of that session; steps in vdd3_prep.py.
# 0. wait for GCHAIR3_DONE (logs/gchair3.log) and ALL_DONE (outputs/logit-lens-20261104/logs/run.log)
# 1. checks on a free GPU (2,3 first): (a) vdd3_decode.py with the mainMed arguments reproduces the existing mainMed captions
#    of 3 AMBER dev images; (b) a g6 run lists non-empty stuff_exempt and hedge_tokens in its status json. Fail -> stop.
# 2. dev: g1-g6 on AMBER ids 101-300 (24 units of 50), workers on GPUs 3,4,4 (user 16:2x; memory-gated), then score vs vdd Base
# 3. pick by the fixed rule (vdd3_prep.py pick); none qualifies -> stop with the dev table
# 4. test once: the pick on the 804 other images, then two score tables (vs vdd Base, vs bench Base)
# Log: logs/vdd3.log, VDD3_DONE (or VDD3_STOPPED with the reason).
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
LOG=$O/logs/vdd3.log; LL=$R/outputs/logit-lens-20261104/logs/run.log
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "VDD3_STOPPED: $*"; exit 1; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
export RBD2_SETS=$O/sets AMBER_REPO=$R/data/amber/repo NLTK_DATA=$L/.cache/nltk_data PYTHONDONTWRITEBYTECODE=1
log "vdd3_chain.sh start; waiting for GCHAIR3_DONE and ALL_DONE"
until grep -q GCHAIR3_DONE $O/logs/gchair3.log 2>/dev/null; do sleep 120; done   # user 16:2x: GPUs 3,4 now, no wait for the logit-lens run
log "start conditions met (GCHAIR3_DONE; logit lens still running on other GPUs, not waited for)"
.venv/bin/python scripts/rbd2/vdd3_prep.py sets >> $LOG 2>&1 || stop "sets failed"
# --- checks
export HF_HOME=$R/../.cache/huggingface HF_HUB_CACHE=$R/../.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src VDD_GRAY=$O/gray448.png
until CG=$(for x in 3 4; do [ $(fre $x) -ge 19500 ] && echo $x && break; done); [ -n "$CG" ]; do sleep 60; done
log "checks on GPU $CG"
CK=$O/runs/amber_v3_check; rm -rf $CK; mkdir -p $CK
CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$CG --query-gpu=uuid --format=csv,noheader) \
  .venv/bin/python -u scripts/rbd2/vdd3_decode.py --model qwen --data $O/sets/amber_full1004.jsonl --tag chk --output $CK --start 100 --end 103 \
  --lam 1 --pen 0.5 --norm median --alpha 1 --layer 16 > $O/logs/v3-check-a.log 2>&1 || stop "identity check run failed (logs/v3-check-a.log)"
.venv/bin/python - >> $LOG 2>&1 <<'PY' || stop "identity check: captions differ from the mainMed rows (GPU $CG)"
import json
O='/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102'
a=[json.loads(l) for l in open(O+'/runs/amber_v3_check/chk-100-103.jsonl')];m=[json.loads(l) for l in open(O+'/runs/amber_full1004_ours_all/mainMed-0-1004.jsonl')][100:103]
same=[x['id']==y['id'] and x['prediction']==y['prediction'] for x,y in zip(a,m)];print('identity check vs mainMed rows 100-102:',same)
assert len(a)==3 and all(same)
PY
python3 - <<'PY'
import json
O='/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102'
r=[json.loads(l) for l in open(O+'/sets/amber_full1004.jsonl')][100]
open(O+'/sets/amber_v3smoke.jsonl','w').write(json.dumps({**r,'max_new_tokens':30})+'\n')
PY
CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$CG --query-gpu=uuid --format=csv,noheader) \
  .venv/bin/python -u scripts/rbd2/vdd3_decode.py --model qwen --data $O/sets/amber_v3smoke.jsonl --tag g6 --output $CK --start 0 --end 1 \
  --alpha 1 --layer 16 --norm median --gpen 2 --tau 1.0 --stuff exempt --hedge 2 > $O/logs/v3-check-b.log 2>&1 || stop "g6 smoke run failed (logs/v3-check-b.log)"
python3 - >> $LOG 2>&1 <<'PY' || stop "g6 status json: stuff_exempt or hedge_tokens empty"
import json
s=json.load(open('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102/runs/amber_v3_check/g6-0-1-status.json'))
print('g6 stuff_exempt',len(s['stuff_exempt']),s['stuff_exempt'][:8],'| hedge_tokens',len(s['hedge_tokens']),s['hedge_tokens'])
assert s['stuff_exempt'] and s['hedge_tokens']
PY
log "checks passed"
# --- workers
phase(){ Q=$1; PIDS=(); i=0
  for spec in "3 v3" "4 v4a" "4 v4b"; do set -- $spec; bash scripts/rbd2/vdd3_pool.sh $1 $2 $Q & PIDS+=($!); sleep 60; done
  wait "${PIDS[@]}"; }
log "dev decode (24 units)"; phase $O/queues/amber_v3_dev.txt
.venv/bin/python scripts/rbd2/vdd3_prep.py join_dev >> $LOG 2>&1 || stop "dev incomplete (see claims_vdd3/fail-*, logs/v3pool-*.log)"
{ echo "=== $(date +%T) dev table (AMBER ids 101-300, paired bootstrap vs vdd Base)"
  $L/venvs/amber/bin/python scripts/rbd2/score_sets.py amber_v3dev $O/runs/amber_v3_dev_all 2>&1 | grep -v 'it/s'; } >> $LOG
.venv/bin/python scripts/rbd2/vdd3_prep.py pick >> $LOG 2>&1 || stop "pick step failed"
PICK=$(head -1 $O/runs/amber_v3_pick.txt); [ "$PICK" = none ] && stop "no dev candidate meets the rule (Cover >= vdd Base - 1.0, CHAIR <= vdd Base); test not run"
if [ "$PICK" != mainMed ]; then log "test decode: $PICK on 804 images (17 units)"; phase $O/queues/amber_v3_test.txt; fi
.venv/bin/python scripts/rbd2/vdd3_prep.py join_test >> $LOG 2>&1 || stop "test incomplete (see claims_vdd3/fail-*, logs/v3pool-*.log)"
export RBD2_N=804
{ echo "=== $(date +%T) test table 1/2 (804 images, paired bootstrap vs vdd Base): base = vdd Base"
  $L/venvs/amber/bin/python scripts/rbd2/score_sets.py amber_v3test $O/runs/amber_v3_test_vs_vdd 2>&1 | grep -v 'it/s'
  echo "=== $(date +%T) test table 2/2 (804 images, paired bootstrap vs bench Base): base = bench Base"
  $L/venvs/amber/bin/python scripts/rbd2/score_sets.py amber_v3test $O/runs/amber_v3_test_vs_bench 2>&1 | grep -v 'it/s'; } >> $LOG
log VDD3_DONE
