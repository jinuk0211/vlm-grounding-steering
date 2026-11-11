#!/bin/bash
# After abl_new.sh: re-decode nosteer rows 0-99 with the CURRENT vdd_steer.py (modified 11-10 01:12 UTC, after the stored
# rows 0-99 of 11-09 17:35 UTC) and compare with the stored rows used in the -steering row. Writes a separate dir only.
cd /workspace/steer
while pgrep -f abl_new.sh > /dev/null || pgrep -f vdd_steer.py > /dev/null; do sleep 30; done
echo "$(date +%T) start recheck"
echo "nosteer --steer none" > queue_recheck.txt
./run_queue.sh queue_recheck.txt sets/chair_test500.jsonl runs/recheck_nosteer 0 100 < /dev/null
python3 -c "
import json
a={r['id']:r['prediction'] for r in map(json.loads,open('runs/steer100/nosteer-0-100.jsonl'))}
b={r['id']:r['prediction'] for r in map(json.loads,open('runs/recheck_nosteer/nosteer-0-100.jsonl'))}
same=sum(a[i]==b[i] for i in b); print('identical', same, 'of', len(b))"
echo "$(date +%T) RECHECK_DONE"
