import json
from pathlib import Path
import torch

from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import (_capture, _fused_embeddings, _preference_attributions,
                                     extract_baseline)

repo = Path('/home/project-b/LJW/vlm-grounding-steering')
config = json.loads((repo / 'configs/gemma4_e4b.local.json').read_text())
record = json.loads((repo / 'data/mmvp-full/fit.jsonl').read_text().splitlines()[0])
adapter = load_adapter(config['model'])
for parameter in adapter.model.parameters():
    parameter.requires_grad_(False)
positive = adapter.prepare(record['image'], record['prompt'], record['chosen'])
negative = adapter.prepare(record['image'], record['prompt'], record['rejected'])
scores = _preference_attributions(adapter, positive, negative, 2)
print(json.dumps({'score_min': scores.min().item(), 'score_max': scores.max().item(),
                  'score_norm': scores.norm().item(), 'positive_count': (scores > 0).sum().item(),
                  'negative_count': (scores < 0).sum().item(), 'tokens': len(scores)}), flush=True)
batch = adapter.prepare(record['image'], record['prompt'])
embeddings = _fused_embeddings(adapter, batch)
layers = [0, 5, 23, 41]
original = _capture(adapter, batch, layers)
for kind in ['positive', 'negative', 'first10', 'last']:
    if kind == 'positive':
        indices = (scores > 0).nonzero().flatten()
        indices = indices[scores[indices].argsort(descending=True)[:10]]
    elif kind == 'negative':
        indices = (scores < 0).nonzero().flatten()
        indices = indices[scores[indices].argsort()[:10]]
    elif kind == 'first10':
        indices = torch.arange(10)
    else:
        indices = torch.tensor([embeddings.shape[1] - 1])
    def mask(hidden):
        changed = hidden.clone()
        changed[:, indices.to(hidden.device)] = 0
        return changed
    changed = _capture(adapter, batch, layers, input_transform=mask)
    print(json.dumps({'kind': kind, 'indices': indices.tolist(),
                      'deltas': {i: float((original[i] - changed[i]).norm()) for i in layers}}), flush=True)
artifact = extract_baseline('grains', adapter, [record], {'ig_steps': 2, 'top_k': 10})
print(json.dumps({'counts': artifact['attribution_token_counts'],
                  'vector_norms': {i: float(v.norm()) for i, v in artifact['vectors'].items()}}), flush=True)
