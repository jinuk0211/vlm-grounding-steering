#!/bin/bash
while ! grep -q "^DONE" /workspace/steer/gemma_dl.log; do sleep 30; done
setsid nohup bash /workspace/steer/vast_pool.sh > /dev/null 2>&1 < /dev/null &
