#!/bin/bash
# HAT + ALOHa in place of AMBER: Base, RUDDER, GrAInS, VTI and RBD on the 385 images of HAT, Qwen2.5-VL-7B and Gemma 4 E4B (REVIS: see hat_prep.py).
# Same decoding as CHAIR (prompt, 512 tokens; HAT_PROMPT / HAT_MAX_TOK change it, with another HAT_RUN so nothing mixes).
#  1 setup + set + DETR labels; HAT reproduction (AP on HAT's own captions vs the paper's 48.62), logged, does not stop the run
#  2 selection on rows 0-59: both Bases + 7 RBD candidates per model; hat_prep.py pick (rule and guard there)
#  3 the picks + both Bases on rows 60-384, the other baselines on all rows
#  4 tables on all 385 rows and on rows 60-384 (not used for selection)
# The official parser model no longer exists (gpt-3.5-turbo shut down 2026-10-23); PARSER selects the replacement, see hat_parse.py.
# Log: logs/$HAT_RUN.log, HAT_DONE / HAT_STOPPED; scorer stderr logs/$HAT_RUN-scorer.log.   progress: hat_prep.py status
# Restart after a kill: rm -r claims_$HAT_RUN (finished units are kept; a unit with partial rows resumes).
# Everything stays under LJW (TMPDIR, XDG_CACHE_HOME, HF_HOME, NLTK_DATA below); CPU steps run at nice 10 with 8 threads.
# usage: [PARSER=hf:Qwen/Qwen3-VL-8B-Instruct] [MODELS="qwen gemma"] [WORKERS="2 3 4 4 5 5"] [PARSE_GPUS="5 4 3 2"] [HAT_RUN=hat385]
#        [HAT_AFTER="PID ..."] hat_chain.sh     HAT_AFTER: nothing here touches a GPU until these processes have exited (the CPU steps
#        go on); two memory-gated pools on one GPU race at unit boundaries, so another chain's GPUs are taken only when it has ended.
# HAT_REPRO_ONLY=1: stop after step 1 (REPRO_DONE).
# HAT_PILOT=1 (HAT_RUN=pilot_chair500), instead of steps 1-4: ALOHa on the CHAIR 500 captions that already exist (vdd Base vs the
# paper's RBD row; one GPU for the parser, no decoding), PILOT_DONE. Reads the score's direction early; own log and parse cache, so it
# can run next to the HAT chain (run hat_setup.sh once first, so that they do not both start the setup).
L=${HAT_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
[ -n "$HAT_PILOT" ] && HAT_RUN=pilot_chair500
export HAT_RUN=${HAT_RUN:-hat385}
PARSER=${PARSER:-hf:Qwen/Qwen3-VL-8B-Instruct}; MODELS=${MODELS:-qwen gemma}; WORKERS=${WORKERS:-2 3 4 4 5 5}; PARSE_GPUS=${PARSE_GPUS:-5 4 3 2}
H=$O/$HAT_RUN; LOG=$O/logs/$HAT_RUN.log; SE=$O/logs/$HAT_RUN-scorer.log; PL=$O/logs/$HAT_RUN-parse.log; AP=$L/venvs/aloha/bin/python; PY=.venv/bin/python; mkdir -p $H $O/logs
APN="nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $AP"
export TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache; mkdir -p $TMPDIR
log(){ echo "=== $(date +%T) $*" >> $LOG; }
stop(){ log "HAT_STOPPED: $*"; exit 1; }
fre(){ nvidia-smi --id=$1 --query-gpu=memory.free --format=csv,noheader,nounits; }
uuid(){ nvidia-smi --id=$1 --query-gpu=uuid --format=csv,noheader; }
wait_after(){ for p in $HAT_AFTER; do while kill -0 $p 2>/dev/null; do sleep 60; done; done; }
log "hat_chain.sh start: parser $PARSER, models $MODELS, workers $WORKERS"
case $PARSER in
  hf:*) S=$L/.cache/huggingface/hub/models--$(echo ${PARSER#hf:} | sed 's#/#--#g'); [ -d $S ] || stop "no $S in the HF cache";;
  openai:*) [ -n "$OPENAI_API_KEY" ] || stop "PARSER=$PARSER needs OPENAI_API_KEY";;
esac
for f in scripts/rbd2/vdd4_decode.py scripts/rbd2/baseline_bench.py scripts/rbd2/gemma_bench.py scripts/rbd2/vti_ws_run.py \
         outputs/rbd-20261030/collect/qwen-vobj2cf.json outputs/rbd-20261030/collect/gemma-vobj2cf.json; do [ -e $f ] || stop "missing $f"; done
[ -x $AP ] || bash scripts/rbd2/hat_setup.sh >> $LOG 2>&1 || stop "setup failed"
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0   # the official metric iterates over sets
detect(){ [ -s $H/detections.json ] || $APN scripts/rbd2/hat_aloha.py detect 2>> $SE >> $LOG || stop "detections failed ($SE)"; }
# Another job's model can take the memory between the check and the load: out-of-memory is retried, any other failure stops.
parse(){ for try in 1 2 3 4 5; do
    case $PARSER in
      hf:*) wait_after; until G=$(for x in $PARSE_GPUS; do [ $(fre $x) -ge 19500 ] && echo $x && break; done); [ -n "$G" ]; do sleep 60; done
            CUDA_VISIBLE_DEVICES=$(uuid $G) $PY -u scripts/rbd2/hat_parse.py $PARSER "$@";;
      *)    $APN -u scripts/rbd2/hat_parse.py $PARSER "$@";;
    esac > $PL 2>&1 && return 0
    grep -q OutOfMemoryError $PL || break; log "parser out of memory, try $try"; sleep 120
  done; stop "parse failed ($PL)"; }
score(){ $APN scripts/rbd2/hat_aloha.py score $PARSER $O/runs/${HAT_RUN}_$1 $2 $3 $H/score-$1-$2-$3 2>> $SE | tee -a $LOG; [ ${PIPESTATUS[0]} -eq 0 ]; }
phase(){ wait_after; PIDS=(); i=0; for G in $WORKERS; do i=$((i+1)); bash scripts/rbd2/hat_pool.sh $G w$i $1 & PIDS+=($!); sleep 60; done; wait "${PIDS[@]}"; }
if [ -n "$HAT_PILOT" ]; then
  $PY scripts/rbd2/hat_prep.py pilot $MODELS >> $LOG 2>&1 || stop "pilot set failed"
  detect; N=$(wc -l < $O/sets/$HAT_RUN.jsonl)
  for M in $MODELS; do
    parse --run $O/runs/${HAT_RUN}_$M; log "$M pilot table: existing CHAIR captions, COCO references"
    score $M 0 $N > /dev/null || log "$M pilot table failed"
  done
  log PILOT_DONE; exit 0
fi
[ -s $O/sets/$HAT_RUN.jsonl ] || $PY scripts/rbd2/hat_prep.py set >> $LOG 2>&1 || stop "set failed"
detect
log "HAT reproduction check"; parse --hat
$APN scripts/rbd2/hat_aloha.py validate $PARSER 2>> $SE >> $LOG || stop "reproduction check failed ($SE)"
[ -n "$HAT_REPRO_ONLY" ] && { log REPRO_DONE; exit 0; }
$PY scripts/rbd2/hat_prep.py queue_dev $MODELS >> $LOG 2>&1 || stop "queue failed"
log "selection decode, rows 0-59"; phase $O/queues/${HAT_RUN}_dev.txt
for M in $MODELS; do
  parse --run $O/runs/${HAT_RUN}_$M; log "$M selection table (rows 0-59)"
  score $M 0 60 > /dev/null && $PY scripts/rbd2/hat_prep.py pick $M $H/score-$M-0-60.json >> $LOG 2>&1 || log "$M pick failed: RBD not continued (table above, logs/hat-$M-*.log)"
done
# User 11-05: REVIS, RUDDER, GrAInS and VTI run on all rows whatever the selection says; a model with no pick loses only RBD.
$PY scripts/rbd2/hat_prep.py queue_test $MODELS >> $LOG 2>&1 || stop "test queue failed"
log "test decode"; phase $O/queues/${HAT_RUN}_test.txt
for M in $MODELS; do
  parse --run $O/runs/${HAT_RUN}_$M
  log "$M table, all 385 rows"; score $M 0 385 > /dev/null || log "$M all-rows table failed"
  log "$M table, rows 60-384 (not used for selection)"; score $M 60 385 > /dev/null || log "$M rows 60-384 table failed"
done
log HAT_DONE
