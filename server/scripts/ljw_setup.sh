#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/ljw_env.sh"
[[ "$(readlink -f "$VLM_REPO")" == "$VLM_REPO" ]] || exit 1
cd "$VLM_REPO"
mkdir -p logs outputs/setup artifacts data "$TMPDIR" "$MPLCONFIGDIR" "$NLTK_DATA"
export CUDA_VISIBLE_DEVICES=""
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
trap 'printf "\nSETUP_FAILED line=%s status=%s\n" "$LINENO" "$?" >&2' ERR
echo 'SETUP_STAGE=python_environment'
python3 - <<'PY'
import shutil
free = shutil.disk_usage('/home/project-b/LJW').free
print('FREE_BYTES=' + str(free), flush=True)
if free < 20_000_000_000:
    raise SystemExit('Less than 20 GB free; check disk space before installing.')
PY
uv venv --python 3.12 .venv
echo 'SETUP_STAGE=pytorch_cuda128'
uv pip install --python .venv/bin/python torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
echo 'SETUP_STAGE=project_dependencies'
uv pip install --python .venv/bin/python -e '.[test,eval]'
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python > outputs/setup/requirements.freeze.txt
echo 'SETUP_STAGE=offline_configuration'
.venv/bin/python - <<'PY'
import json
from pathlib import Path
source = Path('configs/gemma4_e4b.json')
target = Path('configs/gemma4_e4b.local.json')
config = json.loads(source.read_text())
config['model']['local_files_only'] = True
if target.exists() and json.loads(target.read_text()) != config:
    raise SystemExit('Existing local config differs; inspect it before replacing.')
target.write_text(json.dumps(config, indent=2) + '\n')
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo 'SETUP_STAGE=cpu_validation'
.venv/bin/python -m vlm_grounding doctor > outputs/setup/doctor-cpu.json
.venv/bin/python -m pytest -q > outputs/setup/pytest.txt 2>&1
cat outputs/setup/pytest.txt
git rev-parse HEAD > outputs/setup/source-commit.txt
echo 'SETUP_COMPLETE'
