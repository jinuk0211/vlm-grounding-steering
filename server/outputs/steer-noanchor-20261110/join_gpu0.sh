#!/bin/bash
# start pool workers w0a/w0b on GPU 0 once no other user's process is on GPU 0 (user 11-11 15:3x: "gpu 0 다되면 이거 우선")
O=/home/project-b/LJW/vlm-grounding-steering/outputs/steer-noanchor-20261110
U=$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader)
while [ ! -f $O/STOP ]; do
  N=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$U" | wc -l)
  if [ "$N" -eq 0 ]; then sleep 120; N=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$U" | wc -l)
    if [ "$N" -eq 0 ]; then echo "$(date '+%m-%d %T') join_gpu0: GPU 0 free -> w0a w0b" >> $O/pool.log
      setsid nohup bash $O/pool.sh 0 w0a 21000 > /dev/null 2>&1 < /dev/null & sleep 60
      setsid nohup bash $O/pool.sh 0 w0b 21000 > /dev/null 2>&1 < /dev/null & exit 0; fi; fi
  sleep 120
done
