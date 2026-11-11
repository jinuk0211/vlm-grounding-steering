#!/bin/bash
# Second RBD selection on HAT for ALOHa (user 11-08; GPUs 0 and 1 this time only). Design and pick rule in hat_v7_prep.py.
#  1 the 9 candidates on HAT rows 0-99: hat_pool.sh workers on GPU 0 x2 and GPU 1 (memory-gated, all owned by this chain)
#  2 parse (Qwen3-VL-8B, official prompts), ALOHa table on rows 0-99, pick
#  3 the pick on rows 100-384, parse, tables on all 385 rows and on rows 100-384 (not used for this selection), next to the existing rows
# Log: logs/hat_v7.log, V7_DONE / V7_STOPPED. Restart after a kill: rm -r claims_hat385/qwen-v7_* (finished units are kept).
L=${HAT_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
PARSER=hf:Qwen/Qwen3-VL-8B-Instruct; GPUS=${GPUS:-0 0 1}; PARSE_GPUS=${PARSE_GPUS:-1 0}
export HAT_RUN=hat385 TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
LOG=$O/logs/hat_v7.log; PL=$O/logs/hat_v7-parse.log; SE=$O/logs/hat_v7-scorer.log; AP=$L/venvs/aloha/bin/python; PY=.venv/bin/python; mkdir -p $TMPDIR
CPU="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "V7_STOPPED: $*"; exit 1; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
parse(){ for try in 1 2 3 4 5; do
    until G=$(for x in $PARSE_GPUS; do [ $(fre $x) -ge 19500 ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$(uuid $G) $PY -u scripts/rbd2/hat_parse.py $PARSER --run $O/runs/hat385_qwen > $PL 2>&1 && return 0
    grep -q OutOfMemoryError $PL || break; log "parser out of memory, try $try"; sleep 120
  done; stop "parse failed ($PL)"; }
score(){ $CPU $AP scripts/rbd2/hat_aloha.py score $PARSER $O/runs/hat385_qwen $1 $2 $O/hat385/score-qwen-v7-$1-$2 2>> $SE | tee -a $LOG; [ ${PIPESTATUS[0]} -eq 0 ]; }
# 11-06 Gemma selection: the scorer met a caption missing from the parse cache (KeyError); parse once more and retry before giving up
pscore(){ parse; score $1 $2 > /dev/null || { log "scoring rows $1-$2 failed, parse again"; parse; score $1 $2 > /dev/null; }; }
# every worker is this chain's own; a worker leaves only when no unit is open, so the wait covers every unit (hat_pool.sh)
pool(){ PIDS=(); i=0; for G in $GPUS; do i=$((i+1)); bash scripts/rbd2/hat_pool.sh $G v7$1$i $2 & PIDS+=($!); sleep 60; done; wait "${PIDS[@]}"; }
log "hat_v7_chain.sh start: parser $PARSER, workers on GPUs $GPUS"
$CPU $PY scripts/rbd2/hat_v7_prep.py queue_sel >> $LOG 2>&1 || stop "queue failed"
log "selection decode, rows 0-99"; pool s $O/queues/hat385_v7sel.txt
log "selection table, rows 0-99"; pscore 0 100 || stop "selection scoring failed ($SE)"
$CPU $PY scripts/rbd2/hat_v7_prep.py pick $O/hat385/score-qwen-v7-0-100.json >> $LOG 2>&1 || stop "pick failed"
[ "$(cat $O/hat385/pick-qwen-v7.txt)" = none ] && { log "V7_DONE (no pick, rows 100-384 not decoded)"; exit 0; }
$CPU $PY scripts/rbd2/hat_v7_prep.py queue_test >> $LOG 2>&1 || stop "test queue failed"
log "test decode, rows 100-384"; pool t $O/queues/hat385_v7test.txt
log "table, all 385 rows"; pscore 0 385 || log "385-row table failed"
log "table, rows 100-384 (not used for this selection)"; score 100 385 > /dev/null || log "rows 100-384 table failed"
log V7_DONE
