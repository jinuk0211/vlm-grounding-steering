#!/bin/bash
# User, 11-05: HAT + ALOHa runs on GPU 3 (hat_gpu3.sh); other GPUs join when they free up, no restart.
#   GPU 2 x1        when the process LENS_PID (the earlier j2 pool) has gone, no scripts/logit_lens.py runs, and CHAIR dev (cdev_chain.sh) has ended
#   GPU 4 x2, 5 x2  ONLY when told: touch $HAT_RUN/join_gpu45. Session "baseline 실험" says when (Qwen3 GrAInS and REVIS go first). No process or
#                   memory check: at 19:52 rbd6 was stopped without its end line while the GPUs stayed booked, and an inferred "rbd6 has ended"
#                   put HAT workers on them. A GPU that looks free is not therefore free.
#                   First slot on each at join_gpu45, next to one CHAIR dev worker; second slot when CHAIR dev has ended.
# A joined worker runs hat_pool.sh over the current queues in priority order dev, test, bg (the chain rewrites dev and test per phase and model; bg holds the
# later model's units that need no pick, so workers do not idle while the chain parses and scores) until hat_gpu3.sh has ended; with
# no open unit a pool exits at once, so joining late costs nothing. hat_pool.sh keeps its memory gates (KJH's small jobs on GPUs 2 and 4).
# The second worker on a GPU starts 150 s later, so its check sees the first model. The process pattern is anchored at the start of the
# command line: an unanchored grep also matches a remote shell whose command text mentions it.
# usage: hat_join.sh [check]     check prints the conditions and exits.     Log: logs/hat_join.log
L=${HAT_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
RUN=${HAT_RUN:-hat385}; LOG=$O/logs/hat_join.log
MAIN=${HAT_MAIN_PID:-1258987}; LENS=${LENS_PID:-904641}; CDEV=${CDEV_PID:-0}
LENS_RE=${LENS_RE:-^\S*python\S* (-\S+ )*\S*logit_lens\.py}
log(){ echo "=== $(date +%T) $*" >> $LOG; }
main_on(){ kill -0 $MAIN 2>/dev/null; }
lens_done(){ ! kill -0 $LENS 2>/dev/null && ! ps -eo args | grep -qE "$LENS_RE"; }
gpu45_ok(){ [ -f $O/$RUN/join_gpu45 ]; }
cdev_done(){ grep -q 'CDEV_DONE\|CDEV_STOPPED' $O/logs/cdev.log 2>/dev/null || ! kill -0 $CDEV 2>/dev/null; }
gpu2_ok(){ lens_done && cdev_done; }
slot2_ok(){ gpu45_ok && cdev_done; }
if [ "$1" = check ]; then
  for c in main_on lens_done gpu45_ok cdev_done gpu2_ok slot2_ok; do $c && echo "$c yes" || echo "$c no"; done; exit 0
fi
join(){ G=$1; W=$2; COND=$3; D=$4
  until $COND || ! main_on; do sleep 60; done
  main_on || { log "$W: main run ended before GPU $G freed"; return; }
  sleep $D; log "GPU $G worker $W joins"
  while main_on; do
    for Q in dev test bg; do [ -f $O/queues/${RUN}_$Q.txt ] && bash scripts/rbd2/hat_pool.sh $G $W $O/queues/${RUN}_$Q.txt; done
    sleep 60
  done
  log "$W: main run ended, leaving"; }
log "hat_join.sh start, main pid $MAIN"
join 2 j2 gpu2_ok 0 &
join 4 j4a gpu45_ok 0 &
join 4 j4b slot2_ok 150 &
join 5 j5a gpu45_ok 0 &
join 5 j5b slot2_ok 150 &
wait
log HAT_JOIN_DONE
