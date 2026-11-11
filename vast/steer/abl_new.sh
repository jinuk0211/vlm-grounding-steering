#!/bin/bash
# Qwen2.5 ablation with the NEW steering (full_an16_r03 = resid L16 a2 + anchornorm L16 rho 0.3 K16), 2026-11-10.
# Full and -steering (nosteer) already exist on this box (500 rows each); this runs -penalty (pen 0) and -belief (lam 0).
cd /workspace/steer
R=/home/project-b/LJW/vlm-grounding-steering; P=$R/outputs/rbd2-20261102/runs/chair_abl500_all
export NLTK_DATA=/workspace/tether-pilot/nltk_data RBD2_SETS=/workspace/steer/sets RBD2_CHAIR_TEST_STATE=/workspace/steer/chair-test500-scorer-state.json
export PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src
NEW="--steer resid --layers 16 --alpha 2 --steer2 anchornorm --layers2 16 --alpha2 1 --rho2 0.3 --anchor-steps 16"
Q=queue_abl_new.txt
{ echo "abl_nopen_new $NEW --pen 0"; echo "abl_nobelief_new $NEW --lam 0"; } > $Q
while pgrep -f vdd_steer.py > /dev/null; do sleep 30; done
echo "$(date +%T) GPU free"
# smoke: 1 image each, both steering hooks must fire
rm -rf runs/smoke_abl_new
./run_queue.sh $Q sets/chair_test500.jsonl runs/smoke_abl_new 0 1 < /dev/null
python3 -c "
import json
for t,lam,pen in (('abl_nopen_new',1.0,0.0),('abl_nobelief_new',0.0,0.5)):
    r=[json.loads(l) for l in open(f'runs/smoke_abl_new/{t}-0-1.jsonl')]
    assert r and all(x['steer_calls']>0 and x['steer2_calls']>0 for x in r), (t, r)
    s=json.load(open(f'runs/smoke_abl_new/{t}-0-1-status.json'))['args']; assert float(s['lam'])==lam and float(s['pen'])==pen, (t, s)
    print('smoke ok', t, 'steer_calls', r[0]['steer_calls'], 'steer2_calls', r[0]['steer2_calls'], 'lam', s['lam'], 'pen', s['pen'])" || { echo "$(date +%T) SMOKE_FAILED"; exit 1; }
# main: 500 images each
./run_queue.sh $Q sets/chair_test500.jsonl runs/abl_new 0 500 < /dev/null
# table (point estimates only)
F=runs/final500
venv/bin/python abl_table.py \
  "Full (TETHER, new steering)=$F/full_an16_r03-0-500.jsonl" "- steering (alpha=0)=$F/samebox/nosteer-0-500.jsonl" \
  "- penalty (pi=0)=runs/abl_new/abl_nopen_new-0-500.jsonl" "- belief (lambda=0)=runs/abl_new/abl_nobelief_new-0-500.jsonl" \
  "Base (server)=$P/base-0-500.jsonl" \
  "old: Full (server)=$P/rbd_full-0-500.jsonl" "old: - steering (server)=$P/abl_nosteer-0-500.jsonl" \
  "old: - penalty (server)=$P/abl_nopen-0-500.jsonl" "old: - belief (server)=$P/abl_nobelief-0-500.jsonl" \
  > runs/abl_new/table.txt 2> logs/abl_new_score.err
echo "$(date +%T) ABL_NEW_DONE (runs/abl_new/table.txt)"
