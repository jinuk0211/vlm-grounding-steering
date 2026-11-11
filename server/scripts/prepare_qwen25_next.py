"""Inspect the existing Qwen cache and prepare the next run without GPU use."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig, AutoProcessor

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
snapshot = Path(HF_HUB_CACHE) / 'models--Qwen--Qwen2.5-VL-7B-Instruct' / 'snapshots' / REVISION
index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
names = sorted(set(index['weight_map'].values()))
assert len(names) == 5
weights = []
total_payload = 0
for name in names:
    file = snapshot / name
    with file.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        assert 0 < header_size < 20_000_000
        header = json.loads(stream.read(header_size))
    payload_size = file.stat().st_size - 8 - header_size
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    assert set(tensors) == {k for k, v in index['weight_map'].items() if v == name}
    spans = sorted(tuple(v['data_offsets']) for v in tensors.values())
    assert spans[0][0] == 0 and spans[-1][1] == payload_size
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    total_payload += payload_size
    weights.append({'name': name, 'file_bytes': file.stat().st_size,
                    'tensor_payload_bytes': payload_size, 'tensor_count': len(tensors)})
assert total_payload == index['metadata']['total_size']
processor = AutoProcessor.from_pretrained(CHECKPOINT, revision=REVISION,
                                         local_files_only=True, trust_remote_code=False)
config = AutoConfig.from_pretrained(CHECKPOINT, revision=REVISION,
                                   local_files_only=True, trust_remote_code=False)
yes = processor.tokenizer.encode('Yes', add_special_tokens=False)
no = processor.tokenizer.encode('No', add_special_tokens=False)
assert len(yes) == len(no) == 1 and yes != no
gemma = ROOT / 'outputs/author-clone-rerun-20261027-165446'
protocol = json.loads((gemma / 'protocol.json').read_text())
candidates = []
for base in (ROOT / 'data/hallucination-suite', gemma / 'data'):
    for file in sorted(base.iterdir()):
        if file.is_file() and any(word in file.name.lower() for word in ('fit', 'dev', 'calib')):
            candidates.append({'path': str(file), 'bytes': file.stat().st_size})
original = json.loads((ROOT / 'configs/qwen2_5_vl_7b.json').read_text())
plan = {
    'status': 'cpu_preparation_complete_gpu_not_started', 'created_at': time.time(),
    'sequence': 'Finish and officially score Gemma E4B before starting Qwen GPU experiments',
    'model': {'family': 'qwen2_5_vl', 'checkpoint': CHECKPOINT, 'revision': REVISION,
              'local_files_only': True, 'image_size': 448, 'attn_implementation': 'eager',
              'dtype_profiles': ['bfloat16', 'float32']},
    'cache': str(snapshot), 'weights_downloaded': False, 'weight_files': weights,
    'weight_index_sha256': hashlib.sha256((snapshot / 'model.safetensors.index.json').read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256((snapshot / 'config.json').read_bytes()).hexdigest(),
    'weight_index_and_safetensors_headers_verified': True,
    'offline_processor_and_config_load_passed': True,
    'model_type': config.model_type,
    'text_layers': getattr(getattr(config, 'text_config', config), 'num_hidden_layers', None),
    'answer_token_ids': {'Yes': yes, 'No': no},
    'methods': ['base', 'revis', 'vti', 'rudder_clone', 'base_fp32', 'grains_clone'],
    'methods_source': 'Same selected implementation versions and pinned author commits; refit vectors for Qwen',
    'benchmarks': {'POPE': 'COCO adversarial all3000 questions',
                   'CHAIR': 'same fixed COCO val2014 500 images per profile'},
    'pope_max_new_tokens': 128, 'chair_max_new_tokens': 512,
    'seed_per_example': 42, 'batch_size': 1, 'do_sample': False,
    'gpu_indices': [2, 3], 'other_gpu_fallback': False,
    'author_sources': {k: v['commit'] for k, v in protocol['author_sources'].items()},
    'data_reuse_from': str(gemma / 'data'), 'development_file_candidates': candidates,
    'gemma_vectors_must_not_be_reused': True, 'gpu_models_launched': False,
    'existing_repository_configuration': original,
}
output = ROOT / 'outputs/qwen25-vl7b-next-plan-20261028'
output.mkdir(exist_ok=True)
(output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(plan))
