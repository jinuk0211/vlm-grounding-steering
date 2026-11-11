#!/bin/bash
# vast 2080 Ti (22 GB): one Gemma worker. Queue lines: TAG START END ARGS... (re-read each pass). usage: vast_pool.sh
O=/workspace/steer/runs/gemma_abl; Q=$O/queue.txt; mkdir -p $O; touch $Q
R=/home/project-b/LJW/vlm-grounding-steering
export HF_HOME=/workspace/tether-pilot/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/steer:$R/scripts:$R/src CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
       VDD_GRAY=/workspace/steer/gray448.png OMP_NUM_THREADS=4
PY=/workspace/steer/venv/bin/python; D=/workspace/steer/sets/chair_test500.jsonl
log(){ echo "$(date '+%m-%d %T') $*" >> $O/pool.log; }
log "worker start"
while [ ! -f $O/STOP ]; do
  RAN=0
  while read -r T B E X; do
    [ -z "$T" ] && continue; case $T in \#*) continue;; esac
    F=$O/$T-$B-$E.jsonl
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-B)) ]; then continue; fi
    [ $(cat $O/fail-$T-$B-$E 2>/dev/null | wc -l) -ge 2 ] && continue
    log "start $T $B-$E $X"
    if $PY /workspace/steer/vdd_steer_srv.py --model gemma --data $D --output $O --attn eager --tag $T --start $B --end $E $X < /dev/null >> $O/$T-$B-$E.log 2>&1
    then log "done $T $B-$E"; else log "FAILED $T $B-$E"; echo x >> $O/fail-$T-$B-$E; fi
    RAN=1; break
  done < $Q
  [ $RAN -eq 0 ] && sleep 60
done
log "STOP"
