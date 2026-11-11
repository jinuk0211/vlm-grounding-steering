"""Read-only SHA256 verification of the existing cached Qwen shards; CPU only."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/qwen25-vl7b-next-plan-20261028'
SNAPSHOT = ROOT.parent / '.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
         'scope': 'Existing Qwen cache content verification only', 'gpu_models_launched': False, 'files': []}
try:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())
    names = sorted(set(index['weight_map'].values())); assert len(names) == 5
    for name in names:
        file = SNAPSHOT / name; resolved = file.resolve(strict=True)
        assert ROOT.parent in resolved.parents
        digest = hashlib.sha256()
        with file.open('rb') as stream:
            while data := stream.read(8 * 1024 * 1024):
                digest.update(data)
        value = digest.hexdigest()
        assert re.fullmatch('[a-f0-9]{64}', resolved.name), 'Expected the existing HF LFS content-addressed blob'
        assert value == resolved.name, 'Cached shard content differs from its HF LFS SHA256'
        state['files'].append({'name': name, 'bytes': file.stat().st_size, 'sha256': value,
                               'resolved_path': str(resolved), 'hf_lfs_hash_match': True})
        state['completed_files'] = len(state['files']); state['updated_at'] = time.time()
        save(OUTPUT / 'weight-cache-verification-status.json', state)
    state.update(status='verified_cached_weight_files', completed_at=time.time(),
                 total_weight_bytes=sum(r['bytes'] for r in state['files']), all_paths_within_LJW=True,
                 all_hf_lfs_sha256_verified=True)
    save(OUTPUT / 'weight-cache-verification.json', state)
except BaseException:
    state.update(status='failed', error=traceback.format_exc(), updated_at=time.time())
    raise
finally:
    save(OUTPUT / 'weight-cache-verification-status.json', state)
    print(json.dumps(state), flush=True)
