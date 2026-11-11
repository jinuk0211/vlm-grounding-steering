#!/bin/bash
# Two images through every method's real code path on the HAT set, before the long queues reach them: REVIS, RUDDER, GrAInS, VTI on both
# models, Gemma Base and Gemma RBD with the colour-fixed V_obj. GPU 3, one at a time, memory-gated, so it takes turns with the parser.
# Rows go to runs/hat385_smoke_MODEL and are not scored. Log: logs/hat_smoke.log, HAT_SMOKE_DONE.
L=${HAT_L:-/home/project-b/LJW}; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R; G=${SMOKE_GPU:-3}; LOG=$O/logs/hat_smoke.log; D=$O/sets/hat385.jsonl
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       NLTK_DATA=$L/.cache/nltk_data PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src VDD_GRAY=$O/gray448.png TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
run(){ M=$1; T=$2; MF=$3; shift 3; OUT=$O/runs/hat385_smoke_$M; mkdir -p $OUT
  for try in 1 2 3; do
    until [ $(free) -ge $MF ]; do sleep 20; done
    S=$(date +%s); .venv/bin/python -u "$@" --output $OUT --start 0 --end 2 < /dev/null > $O/logs/hat_smoke-$M-$T.log 2>&1; RC=$?
    grep -q OutOfMemoryError $O/logs/hat_smoke-$M-$T.log && [ $RC -ne 0 ] && { echo "$(date +%T) $M $T out of memory, retry"  >> $LOG; sleep 60; continue; }
    break
  done
  N=$(cat $OUT/$T-0-2.jsonl 2>/dev/null | wc -l)
  echo "$(date +%T) $M $T rc $RC rows $N in $(( $(date +%s)-S ))s  $([ $RC -ne 0 ] && tail -1 $O/logs/hat_smoke-$M-$T.log | cut -c1-150)" >> $LOG; }
echo "=== $(date +%T) smoke start, GPU $G" >> $LOG
B=scripts/rbd2/baseline_bench.py; E=scripts/rbd2/gemma_bench.py
run qwen revis 19500 $B --method revis --attn sdpa --set $D
run qwen rudder_clone 19500 $B --method rudder_clone --attn sdpa --set $D
run qwen grains_clone 19500 $B --method grains_clone --attn eager --set $D
run qwen vtiws_v0.9_t0.9 19500 scripts/rbd2/vti_ws_run.py --av 0.9 --at 0.9 --tag vtiws_v0.9_t0.9 --set $D
run gemma base 21257 $E --method base --set $D
run gemma revis 21257 $E --method revis --set $D
run gemma rudder_clone 21257 $E --method rudder_clone --set $D
run gemma grains_clone 21257 $E --method grains_clone --set $D
run gemma vti 21257 $E --method vti --set $D
run gemma cf_a3bl2p1 21257 scripts/rbd2/vdd4_decode.py --model gemma --data $D --tag cf_a3bl2p1 --attn eager --layer 24 --alpha 3 --lam 2 --pen 1 --vobj $R/outputs/rbd-20261030/collect/gemma-vobj2cf.json
.venv/bin/python - >> $LOG 2>&1 <<'PY'
import glob,json
for f in sorted(glob.glob('outputs/rbd2-20261102/runs/hat385_smoke_*/*-0-2.jsonl')):
    r=[json.loads(l) for l in open(f)]
    print(f.split('/')[-2][12:],f.split('/')[-1][:-10],len(r),'rows tok',[x['generated_tokens'] for x in r],'|',r[0]['prediction'][:80].replace('\n',' ') if r else '')
PY
echo "=== $(date +%T) HAT_SMOKE_DONE" >> $LOG
