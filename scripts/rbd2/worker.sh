#!/bin/bash
# usage: worker.sh GPU QUEUEFILE   -- lines "SET TAG vdd_decode-args..."; run in order, one at a time
G=$1; Q=$2
while read -r S T A; do
  [ -z "$S" ] && continue; case $S in \#*) continue;; esac
  if [ -f /workspace/vdd/$S/$T-0-60.jsonl ] && [ $(wc -l < /workspace/vdd/$S/$T-0-60.jsonl) -ge 60 ]; then continue; fi
  echo "$(date +%T) start $S $T $A" >> /workspace/logs/vdd/worker$G.log
  /workspace/rbd_pilot/vrun.sh $G /workspace/sets/$S.jsonl /workspace/vdd/$S $T 60 $A || echo "$(date +%T) FAILED $S $T" >> /workspace/logs/vdd/worker$G.log
  echo "$(date +%T) end $S $T" >> /workspace/logs/vdd/worker$G.log
done < $Q
echo "$(date +%T) WORKER_DONE" >> /workspace/logs/vdd/worker$G.log
