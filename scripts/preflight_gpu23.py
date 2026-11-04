"""Check native full-FP32 sharding and the original GrAInS hook before resuming."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch
from gpu23_device_adapter import PAIR, load_on_selected_devices, grains_on_layer_device
from author_clone_bridge import load_authors, verify_clone_sources
from vlm_grounding.cli import _seed

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output
result = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(), 'models_used_only_on': PAIR}
try:
    assert os.environ['CUDA_VISIBLE_DEVICES'] == PAIR
    config = json.loads((output / 'config.json').read_text())
    config['model']['dtype'] = 'float32'
    _seed(config)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert verify_clone_sources() == protocol['author_sources']
    adapter = load_on_selected_devices(config['model'])
    authors = load_authors()
    record = json.loads((output / 'data/fit50.jsonl').read_text().splitlines()[0])
    batch = adapter.prepare(record['image'], 'Please describe this image in detail.')
    baseline = adapter.generate(batch, max_new_tokens=8)
    artifact = torch.load(output / 'grains-author.pt', map_location='cpu', weights_only=True)
    assert artifact['model'] == config['model']
    diagnostics = {}
    _seed(config)
    with grains_on_layer_device(adapter, artifact['vector'], authors, diagnostics):
        steered = adapter.generate(batch, max_new_tokens=32)
    assert diagnostics['calls'] > 0 and diagnostics['changed_elements'] > 0
    result.update(status='passed', baseline=baseline, steered=steered,
                  diagnostics=diagnostics, device_proof=adapter.gpu23_device_proof,
                  max_memory_allocated_gb=[torch.cuda.max_memory_allocated(i)/1e9 for i in range(2)],
                  scope='Compatibility preflight on one development image; not a benchmark score',
                  completed_at=time.time())
except BaseException:
    result.update(status='failed', error=traceback.format_exc(), completed_at=time.time())
    raise
finally:
    (output / 'gpu23-preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
