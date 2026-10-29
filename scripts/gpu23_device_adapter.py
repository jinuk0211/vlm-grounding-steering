"""Device placement only: native BF16 replica or full FP32 over GPUs 2 and 3."""
import contextlib
import json
import os

import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from vlm_grounding.backend import ModelAdapter, load_adapter as native_load_adapter
from author_clone_bridge import grains_context as original_grains_context

GPU2 = 'GPU-bf42d0ab-8a2e-cc64-26df-f1b8a99ffbde'
GPU3 = 'GPU-4c593ff4-fcc4-39c9-d7b4-981dabee1cb3'
PAIR = GPU2 + ',' + GPU3


def load_on_selected_devices(config, output=None, worker=None):
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    if config['dtype'] == 'bfloat16':
        assert visible in (GPU2, GPU3) and torch.cuda.device_count() == 1
        adapter = native_load_adapter(config)
    else:
        assert config['dtype'] == 'float32' and visible == PAIR
        assert torch.cuda.device_count() == 2
        shared = dict(revision=config['revision'], local_files_only=True, trust_remote_code=False)
        processor = AutoProcessor.from_pretrained(config['checkpoint'], **shared)
        model = Gemma4ForConditionalGeneration.from_pretrained(
            config['checkpoint'], torch_dtype=torch.float32,
            attn_implementation=config.get('attn_implementation', 'eager'),
            device_map='balanced', max_memory={0: '21GiB', 1: '21GiB'}, **shared)
        adapter = ModelAdapter(model, processor, config)
        assert {p.device.type for p in model.parameters()} == {'cuda'}
        assert {p.device.index for p in model.parameters()} == {0, 1}
        assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {torch.float32}
    proof = {
        'visible_devices': visible, 'dtype': config['dtype'],
        'device_count': torch.cuda.device_count(),
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'parameter_dtypes': sorted({str(p.dtype) for p in adapter.model.parameters() if p.is_floating_point()}),
        'parameter_devices': sorted({str(p.device) for p in adapter.model.parameters()}),
        'hf_device_map': {k: str(v) for k, v in getattr(adapter.model, 'hf_device_map', {}).items()},
        'checkpoint': config['checkpoint'], 'revision': config['revision'],
        'quantization': False, 'cpu_or_disk_offload': False,
        'native_generation_and_original_author_functions_unchanged': True,
    }
    adapter.gpu23_device_proof = proof
    if output is not None:
        (output / (worker + '-device-map.json')).write_text(json.dumps(proof, indent=2) + '\n')
    return adapter


@contextlib.contextmanager
def grains_on_layer_device(adapter, vector, authors, diagnostics=None):
    device = next(adapter.layers[27].parameters()).device
    with original_grains_context(adapter, vector.to(device), authors, diagnostics):
        yield
    if diagnostics is not None:
        diagnostics['steering_vector_device'] = str(device)

