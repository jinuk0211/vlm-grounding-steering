#!/bin/bash
# vdd7 checks on GPU 1: (1) its defaults give vdd4's captions (same GPU, and the stored HAT rows of cf_a2bl1p05), (2) --cf blur --ew prob runs
L=/home/project-b/LJW; R=$L/vlm-grounding-steering; O=$R/outputs/rbd2-20261102; cd $R
export HF_HOME=$L/.cache/huggingface HF_HUB_CACHE=$L/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
       PYTHONPATH=$R/scripts/rbd2:$R/scripts:$R/src VDD_GRAY=$O/gray448.png TMPDIR=$L/.cache/tmp XDG_CACHE_HOME=$L/.cache \
       CUDA_VISIBLE_DEVICES=$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)
D=$O/sets/hat385.jsonl; OUT=$O/runs/v7_smoke; mkdir -p $OUT
A="--attn sdpa --layer 16 --alpha 2 --lam 1 --pen 0.5 --vobj $R/outputs/rbd-20261030/collect/qwen-vobj2cf.json"
echo "=== $(date +%T) vdd4"; .venv/bin/python -u scripts/rbd2/vdd4_decode.py --model qwen --data $D --tag v4 --output $OUT --start 0 --end 3 $A 2>&1 | grep -vE 'it/s|Loading' | tail -3
echo "=== $(date +%T) vdd7 defaults"; .venv/bin/python -u scripts/rbd2/vdd7_decode.py --model qwen --data $D --tag v7 --output $OUT --start 0 --end 3 $A 2>&1 | grep -vE 'it/s|Loading' | tail -3
B="--attn sdpa --layer 16 --alpha 2 --lam 2 --pen 1 --vobj $R/outputs/rbd-20261030/collect/qwen-vobj2cf.json --cf blur --ew prob --log-belief"
echo "=== $(date +%T) vdd7 blur + prob"; .venv/bin/python -u scripts/rbd2/vdd7_decode.py --model qwen --data $D --tag bpw --output $OUT --start 0 --end 2 $B 2>&1 | grep -vE 'it/s|Loading' | tail -3
echo "=== $(date +%T) compare"
.venv/bin/python - <<PY
import json
r=lambda p:[json.loads(s) for s in open(p)]
a=r('$OUT/v4-0-3.jsonl');b=r('$OUT/v7-0-3.jsonl');s=r('$O/runs/hat385_qwen/cf_a2bl1p05-0-60.jsonl')[:3];c=r('$OUT/bpw-0-2.jsonl')
print('vdd7 defaults == vdd4 (same GPU):',[x['prediction']==y['prediction'] for x,y in zip(a,b)])
print('vdd7 defaults == stored HAT cf_a2bl1p05 rows:',[x['prediction']==y['prediction'] for x,y in zip(b,s)])
print('settings', b[0]['setting'])
for x,y in zip(c,b[:2]):
    print('--- bpw',x['generated_tokens'],'tok, identical to default:',x['prediction']==y['prediction'],x['setting']['cf'],x['setting']['ew'])
    print(x['prediction'][:600]);print('belief top', x['belief_top'][-1])
PY
ls -la $O/cf | head -4
echo "=== $(date +%T) SMOKE_END"
