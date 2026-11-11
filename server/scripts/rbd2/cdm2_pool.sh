#!/bin/bash
# M3ID round 2 (user 11-11 20:3x): CHAIR dev 100 on Qwen2.5 / Gemma / Qwen3 + the rest of Qwen3 CHAIR test 500, on GPU 5 after the Qwen ablation.
# Same decoder and settings as round 1 (cdm_decode.py, M3ID lam 0.02 thr 0.3, attn of each model's vdd Base rows).
# Queue $O/queue2.txt lines "MODEL METHOD START END SET OUTDIR" (SET = sets/SET.jsonl, OUTDIR relative to $O); claims by mkdir in $O/claims2.
# usage: cdm2_pool.sh GPU_INDEX WORKER_NAME MIN_FREE_MIB      stop: touch $O/STOP2 (leaves after the current unit)
G=$1; W=$2; MF=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/cdm-20261109; Q=$O/queue2.txt; C=$O/claims2; SETS=$R/outputs/rbd2-20261102/sets
mkdir -p $O/logs $C $L/.cache/tmp
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       PYTHONDONTWRITEBYTECODE=1 NLTK_DATA=$L/.cache/nltk_data XDG_CACHE_HOME=$L/.cache TMPDIR=$L/.cache/tmp \
       OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
attn(){ case $1 in gemma) echo eager;; *) echo sdpa;; esac; }
log(){ echo "$(date '+%m-%d %T') $*" >> $O/logs/pool2-$W.log; }
log "worker $W GPU $G gate $MF MiB start"
while true; do
  [ -f $O/STOP2 ] && { log STOP2; break; }
  OPEN=0; RAN=0
  while read -r MOD MET S E SET OD; do
    [ -z "$MOD" ] && continue; case $MOD in \#*) continue;; esac
    OUT=$O/$OD; F=$OUT/$MET-$S-$E.jsonl; U=$SET-$MOD-$MET-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    OPEN=$((OPEN+1)); [ -d $C/$U ] && continue
    exec 9>$C/gpu$G.lock; flock 9
    if [ $(free) -lt $MF ] || ! mkdir $C/$U 2>/dev/null; then flock -u 9; exec 9>&-; continue; fi
    echo $W > $C/$U/worker; mkdir -p $OUT; ST=$OUT/$MET-$S-$E-status.json; rm -f $ST
    log "start $U"
    nice -n 5 .venv/bin/python -u scripts/rbd2/cdm_decode.py --model $MOD --method $MET --attn $(attn $MOD) --data $SETS/$SET.jsonl \
       --tag $MET --output $OUT --start $S --end $E < /dev/null 9>&- >> $O/logs/r2-$MOD-$MET.log 2>&1 &
    P=$!; sleep 20
    for i in $(seq 1 60); do grep -q '"stage": "running"' $ST 2>/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 5; done
    flock -u 9; exec 9>&-
    if wait $P; then log "end $U"; else log "FAILED $U"; echo $W >> $C/fail-$U; fi
    rm -r "${C:?}/${U:?}"; RAN=1; break
  done < $Q
  [ $RAN -eq 1 ] && continue
  [ $OPEN -eq 0 ] && break
  sleep 30
done
log POOL_DONE
