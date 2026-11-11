import json
from pathlib import Path
import torch
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import _fused_embeddings, _hook, _replace_input, sequence_log_probability

r = Path('/home/project-b/LJW/vlm-grounding-steering')
c = json.loads((r / 'configs/gemma4_e4b.local.json').read_text())
record = json.loads((r / 'data/mmvp-full/fit.jsonl').read_text().splitlines()[0])
a = load_adapter(c['model'])
for p in a.model.parameters():
    p.requires_grad_(False)
positive = a.prepare(record['image'], record['prompt'], record['chosen'])
negative = a.prepare(record['image'], record['prompt'], record['rejected'])
pe, ne = _fused_embeddings(a, positive), _fused_embeddings(a, negative)
length = int((positive['labels'][0] != -100).nonzero()[0])
prompt = pe[:, :length]
for alpha in [0., .0053, .03125, .0625, .5, 1.]:
    variable = (prompt * alpha).detach().requires_grad_(True)
    gradients = []
    objectives = []
    for batch, embeddings, sign in [(positive, pe, 1), (negative, ne, -1)]:
        def replace(_module, args, kwargs):
            return _replace_input(args, kwargs, torch.cat((variable, embeddings[:, length:]), dim=1))
        with _hook(a.layers[0], replace, pre=True):
            output = a.forward(batch)
            objective = sequence_log_probability(output.logits, batch['labels'])
            gradient = sign * torch.autograd.grad(objective, variable)[0].float()
            gradients.append(gradient)
            objectives.append(objective.item())
    g = sum(gradients)
    print(json.dumps({'alpha': alpha, 'objectives': objectives, 'finite': bool(torch.isfinite(g).all()),
                      'finite_fraction': torch.isfinite(g).float().mean().item(),
                      'gradient_max': g.abs().max().item()}), flush=True)
