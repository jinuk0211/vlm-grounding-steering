#!/bin/bash
# usage: qbel.sh START END  -- Qwen -belief at alpha 3 (lam 0 pen .5 resid L16 a3) on GPU 5, sdpa
G=5; source /home/project-b/LJW/vlm-grounding-steering/scripts/steer_vast/steer_env.sh
O=$R/outputs/steer-noanchor-20261110/qwen; B=$1; E=$2
nice -n 5 .venv/bin/python $S/vdd_steer.py --model qwen --data $R/outputs/rbd2-20261102/sets/chair_test500.jsonl --output $O --attn sdpa \
  --tag abl_nobelief_a3 --start $B --end $E --lam 0 --pen 0.5 --steer resid --layers 16 --alpha 3 < /dev/null >> $O/abl_nobelief_a3-$B-$E.log 2>&1
echo "$(date '+%m-%d %T') qbel $B-$E exit $?" >> $R/outputs/steer-noanchor-20261110/pool.log
