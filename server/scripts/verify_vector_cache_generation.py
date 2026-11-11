"""Check exact generated token IDs with CPU versus preloaded GPU vectors."""
import json
import time
from pathlib import Path

import torch

from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')


def main():
    parent = Path(json.loads((ROOT / 'outputs/hallucination-suite-current.json').read_text())['output'])
    config = json.loads((parent / 'config.json').read_text())
    assert torch.cuda.device_count() == 1
    adapter = load_adapter(config['model'])
    row = json.loads((ROOT / 'data/mmvp-full/fit.jsonl').read_text().splitlines()[0])
    original_generate = adapter.model.generate
    captured = []

    def capture(*args, **kwargs):
        output = original_generate(*args, **kwargs)
        captured.append(output.detach().cpu())
        return output

    adapter.model.generate = capture
    results = []
    for method in ['revis', 'grains', 'vti']:
        artifact = torch.load(parent / 'artifacts' / f'{method}.pt', map_location='cpu', weights_only=True)
        prepared = {**artifact}
        for key in ['vectors', 'vision_vectors']:
            if key in artifact:
                prepared[key] = {layer: value.to(adapter.device) for layer, value in artifact[key].items()}
        responses, durations = [], []
        for candidate in [artifact, prepared]:
            _seed(config)
            batch = adapter.prepare(row['image'], 'Please describe this image in detail.')
            torch.cuda.synchronize();started = time.perf_counter()
            with build_baseline(method, adapter, candidate, config['methods'][method]):
                response = adapter.generate(batch, max_new_tokens=48)
            torch.cuda.synchronize()
            responses.append(response);durations.append(time.perf_counter() - started)
            del batch
        assert responses[0] == responses[1] and torch.equal(captured[-2], captured[-1]), method
        results.append({'method': method, 'generated_token_ids_identical': True, 'max_new_tokens': 48,
                        'cpu_vector_seconds': durations[0], 'gpu_vector_seconds': durations[1]})
    result = {'status': 'passed', 'all_generated_token_ids_identical': True,
              'development_image_used': row['image'], 'results': results}
    (ROOT / 'outputs/gpu-vector-cache-generation-validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
