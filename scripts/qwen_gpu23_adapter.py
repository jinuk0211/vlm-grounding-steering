"""Prepared Qwen loader restricted to physical GPU2/GPU3; no load on import."""
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vlm_grounding.backend import ModelAdapter, load_adapter as native_load_adapter

GPU2 = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3 = 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
PAIR = GPU2 + ',' + GPU3
CHECKPOINT = 'Qwen/Qwen2.5-VL-7B-Instruct'
REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
ROOT = Path(__file__).resolve().parents[1]


def require_gemma_complete():
    previous = ROOT / 'outputs/author-clone-rerun-20261027-165446'
    state = json.loads((previous / 'status.json').read_text())
    assert state['stage'] == 'complete', 'Finish Gemma before any Qwen GPU run'
    report = json.loads((previous / 'report.json').read_text())
    assert report['status'] == 'verified_complete'
    assert report['audit']['all_21000_responses_verified'] and report['audit']['official_scorers_used']


def verify_run_protocol(output):
    """Check sealed inputs before spending GPU time on this fresh Qwen run."""
    assert output.resolve().parent == ROOT / 'outputs'
    protocol = json.loads((output / 'protocol.json').read_text())
    body = {k: v for k, v in protocol.items() if k != 'sha256'}
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == protocol['sha256']
    assert protocol['model']['checkpoint'] == CHECKPOINT and protocol['model']['revision'] == REVISION
    assert protocol['model']['family'] == 'qwen2_5_vl' and protocol['model']['dtype'] == 'bfloat16'
    assert protocol['model']['image_size'] == 448 and protocol['model']['attn_implementation'] == 'eager'
    assert protocol['model']['local_files_only']
    assert digest(output / 'config.json') == protocol['config_sha256']
    assert json.loads((output / 'config.json').read_text())['model'] == protocol['model']
    for file, expected in protocol['source_hashes'].items():
        assert digest(ROOT / file) == expected, file
    for file, expected in protocol['manifest_hashes'].items():
        assert digest(output / 'data' / file) == expected, file
    assert digest(output / 'tasks.json') == protocol['tasks_sha256']
    return protocol


def load_qwen(config, output=None, worker=None):
    require_gemma_complete()
    assert config['family'] == 'qwen2_5_vl'
    assert config['checkpoint'] == CHECKPOINT and config['revision'] == REVISION
    assert config.get('local_files_only', False)
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    if config['dtype'] == 'bfloat16':
        assert visible in (GPU2, GPU3) and torch.cuda.device_count() == 1
        adapter = native_load_adapter(config)
    else:
        assert config['dtype'] == 'float32' and visible == PAIR
        assert torch.cuda.device_count() == 2
        shared = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
        processor = AutoProcessor.from_pretrained(CHECKPOINT, **shared)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CHECKPOINT, torch_dtype=torch.float32,
            attn_implementation=config.get('attn_implementation', 'eager'),
            device_map='balanced', max_memory={0: '21GiB', 1: '21GiB'}, **shared)
        adapter = ModelAdapter(model, processor, config)
        assert {p.device.type for p in model.parameters()} == {'cuda'}
        assert {p.device.index for p in model.parameters()} == {0, 1}
        assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.float32}
    assert adapter.model_type == 'qwen2_5_vl' and len(adapter.layers) == 28
    cached = try_to_load_from_cache(CHECKPOINT, 'config.json', revision=REVISION)
    assert isinstance(cached, str) and Path(cached).parent.name == REVISION
    observed = getattr(adapter.model.config, '_commit_hash', None)
    assert observed in (None, REVISION)
    adapter.model.config._commit_hash = REVISION
    proof = {
        'checkpoint': CHECKPOINT, 'revision': REVISION, 'cached_config': cached,
        'observed_revision_metadata': observed, 'restored_revision_metadata': REVISION,
        'visible_devices': visible, 'dtype': config['dtype'], 'text_layers': len(adapter.layers),
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'parameter_devices': sorted({str(p.device) for p in adapter.model.parameters()}),
        'parameter_dtypes': sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()}),
        'hf_device_map': {k: str(v) for k, v in getattr(adapter.model, 'hf_device_map', {}).items()},
        'quantization': False, 'cpu_or_disk_offload': False, 'model_weights_changed': False,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }
    adapter.qwen_gpu23_device_proof = proof
    if output is not None:
        assert worker is not None
        (output / (worker + '-device-map.json')).write_text(json.dumps(proof, indent=2) + '\n')
    return adapter
