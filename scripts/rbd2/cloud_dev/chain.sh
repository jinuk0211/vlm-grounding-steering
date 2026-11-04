#!/bin/bash
# usage: chain.sh GPU NEXTQUEUE  -- wait for the running worker on GPU to finish, then start the next queue
G=$1; N=$2
while pgrep -f "[w]orker.sh $G " > /dev/null; do sleep 20; done
exec /workspace/rbd_pilot/worker.sh $G $N
