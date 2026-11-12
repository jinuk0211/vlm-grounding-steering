#!/bin/bash
# THRONE rows 1000-4999 (t4k_chain.sh): decode worker. Queue lines "KIND SET OUTDIR TAG START END args..." -> runs/OUTDIR/TAG-START-END.jsonl
# (the decoders resume partial files), with the decoder and environment of the pool that made the 1,000-image rows:
#   qb  baseline_bench.py --method TAG       (pb_bl_pool.sh)      gb  gemma_bench.py --method TAG                     (gemma_pool.sh)
#   qv  vdd_decode.py --model qwen           (pb_pool.sh)         gv  vdd_decode.py --model gemma --attn eager        (gemma_pool.sh)
#   v4q vdd4_decode.py --model qwen          (vdd4_pool.sh)       v4g vdd4_decode.py --model gemma                    (vdd4_pool.sh)
# A unit is claimed with mkdir in claims_t4k only when the GPU has the model's MiB free twice, 10-30 s apart. CUDA out-of-memory is not counted
# (back off 1-3 min); other failures twice -> skipped. Leaves as soon as no unit is left to claim (units other workers hold are theirs), so the
# slot moves on to judging. The file t4k/STOP makes it leave after its current unit.     usage: t4k_pool.sh GPU_INDEX WORKER_NAME QUEUE
G=$1; W=$2; Q=$3
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; LG=$O/logs; C=$O/claims_t4k; mkdir -p $LG $C $L/.cache/tmp $O/t4k
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       NLTK_DATA=$L/.cache/nltk_data VDD_GRAY=$O/gray448.png TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=$G --query-gpu=uuid --format=csv,noheader)
cd $R
free(){ nvidia-smi --id=$G --query-gpu=memory.free --format=csv,noheader,nounits; }
P3=$R/scripts/rbd2:$R/scripts:$R/src
while [ ! -f $O/t4k/STOP ]; do
  RAN=; LEFT=0
  while read -r K D OD T S E A; do
    [ -z "$K" ] && continue; case $K in \#*) continue;; esac
    OUT=$O/runs/$OD; F=$OUT/$T-$S-$E.jsonl; U=$OD-$T-$S-$E
    if [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then continue; fi
    [ $(cat $C/fail-$U 2>/dev/null | wc -l) -ge 2 ] && continue
    [ -d $C/$U ] && continue
    LEFT=$((LEFT+1))
    case $K in gb|gv|v4g) MF=21257;; *) MF=19500;; esac
    [ $(free) -ge $MF ] || continue; sleep $((RANDOM%20+10)); [ $(free) -ge $MF ] || continue
    mkdir $C/$U 2>/dev/null || continue
    echo $W > $C/$U/worker; mkdir -p $OUT; echo "$(date '+%m-%d %T') start $U" >> $LG/t4kpool-$W.log; UL=$LG/t4kunit-$W.log
    case $K in
      qb)  PP=$P3; X=(scripts/rbd2/baseline_bench.py --method $T --set $O/sets/$D.jsonl);;
      gb)  PP=$P3; X=(scripts/rbd2/gemma_bench.py --method $T --set $O/sets/$D.jsonl);;
      qv)  PP=$R/scripts; X=(scripts/rbd2/vdd_decode.py --model qwen --data $O/sets/$D.jsonl --tag $T);;
      gv)  PP=$P3; X=(scripts/rbd2/vdd_decode.py --model gemma --attn eager --data $O/sets/$D.jsonl --tag $T);;
      v4q) PP=$P3; X=(scripts/rbd2/vdd4_decode.py --model qwen --data $O/sets/$D.jsonl --tag $T);;
      v4g) PP=$P3; X=(scripts/rbd2/vdd4_decode.py --model gemma --data $O/sets/$D.jsonl --tag $T);;
      *)   echo "$(date '+%m-%d %T') unknown kind $K" >> $LG/t4kpool-$W.log; rm -r $C/$U; continue;;
    esac
    PYTHONPATH=$PP .venv/bin/python -u "${X[@]}" --output $OUT --start $S --end $E $A < /dev/null > $UL 2>&1; RC=$?
    cat $UL >> $LG/t4k-$OD-$T.log
    if [ $RC -eq 0 ] && [ -f $F ] && [ $(wc -l < $F) -ge $((E-S)) ]; then echo "$(date '+%m-%d %T') end $U" >> $LG/t4kpool-$W.log
    else
      rm -r $C/$U
      if grep -q 'OutOfMemoryError\|CUDA out of memory' $UL; then echo "$(date '+%m-%d %T') OOM $U (not counted)" >> $LG/t4kpool-$W.log; sleep $((RANDOM%120+60))
      else echo "$(date '+%m-%d %T') FAILED $U rc $RC" >> $LG/t4kpool-$W.log; echo $W >> $C/fail-$U; fi
    fi
    RAN=1; break
  done < $Q
  [ -n "$RAN" ] && continue
  [ $LEFT -eq 0 ] && break
  sleep 60
done
echo "$(date '+%m-%d %T') POOL_DONE $(basename $Q)" >> $LG/t4kpool-$W.log
