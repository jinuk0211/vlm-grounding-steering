#!/usr/bin/env bash
# Gemma-only launcher. Every model command checks the selected physical GPU.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ljw_env.sh"
cd "$VLM_REPO"
command="${1:-help}"
shift || true
case "$command" in
  help|-h|--help)
    echo 'Usage: bash scripts/gemma.sh check|doctor'
    echo '       bash scripts/gemma.sh smoke [GPU_INDEX=2]'
    echo '       bash scripts/gemma.sh smoke-with-explorer  # temporarily pause LJW Explorer on GPU 2'
    echo '       bash scripts/gemma.sh with-explorer COMMAND [CLI arguments]  # GPU 2'
    echo '       bash scripts/gemma.sh run|discover|fit|extract GPU_INDEX [CLI arguments]'
    echo '       bash scripts/gemma.sh suite GPU_INDEX DEVELOPMENT_JSONL EVALUATION_JSONL OUTPUT_DIR [tune|test]'
    exit 0 ;;
  check) export CUDA_VISIBLE_DEVICES=""; exec python scripts/ljw_check.py "$@" ;;
  doctor) export CUDA_VISIBLE_DEVICES=""; exec python -m vlm_grounding doctor "$@" ;;
  smoke-with-explorer) exec python scripts/ljw_gpu2_validation.py smoke "$@" ;;
  with-explorer) exec python scripts/ljw_gpu2_validation.py "$@" ;;
  smoke|run|discover|fit|extract|suite) ;;
  *) echo "Unknown Gemma command: $command" >&2; exit 2 ;;
esac
gpu_index="${1:-2}"
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo 'GPU index must be numeric' >&2; exit 2; }
if (( $# )); then shift; fi
mkdir -p "$VLM_REPO/.cache/gpu-locks"
exec 9>"$VLM_REPO/.cache/gpu-locks/gpu-${gpu_index}.lock"
flock -n 9 || { echo "Another Gemma command already holds GPU $gpu_index" >&2; exit 1; }
export LJW_GPU_INDEX="$gpu_index"
uuid="$(python - <<'PY'
import os, subprocess
index = os.environ['LJW_GPU_INDEX']
value = subprocess.check_output(['nvidia-smi', '-i', index,
    '--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits'], text=True).strip().split(',')
uuid, total, used, util = [part.strip() for part in value]
processes = subprocess.check_output(['nvidia-smi',
    '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
if int(total) < 22000:
    raise SystemExit('Gemma BF16 smoke needs a GPU with at least 22 GB VRAM')
if int(used) > 10 or int(util) != 0 or any(
        row.split(',')[0].strip() == uuid for row in processes.splitlines()):
    raise SystemExit(f'GPU {index} is busy ({used} MiB, {util}%); retry when it is free.')
print(uuid)
PY
)"
export CUDA_VISIBLE_DEVICES="$uuid"
echo "PHYSICAL_GPU=$gpu_index CUDA_VISIBLE_DEVICES=$uuid"
python scripts/ljw_check.py
case "$command" in
  smoke) exec python scripts/ljw_smoke.py "$@" ;;
  suite) exec bash scripts/run_gpu.sh "$VLM_GEMMA_CONFIG" "$@" ;;
  *)
    for argument in "$@"; do
      [[ "$argument" != --config && "$argument" != --config=* ]] || {
        echo 'This launcher supplies the pinned local Gemma config automatically.' >&2; exit 2;
      }
    done
    exec python -m vlm_grounding "$command" --config "$VLM_GEMMA_CONFIG" "$@" ;;
esac
