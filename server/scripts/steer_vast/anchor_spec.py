"""How image-specific is the steering anchor? Teacher forcing of stored Base captions with the real and the gray image; at the
steering layer L (1-based decoder layer output, as vdd_steer.py) the anchor of image i is a_i = mean_{t=1..16} (h_real - h_gray)
at caption steps t (query capstart-1+t). Reports, over images: mean pairwise cosine of anchors, the share of anchor norm along the
common mean direction (|mean a| / mean |a|), cosine of each anchor with the mean anchor, and cos(d_t, a_i) late in the caption.
usage: anchor_spec.py --model M --rows BASE.jsonl --layer L --n N"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C
GRAY = Path('/home/project-b/LJW/vlm-grounding-steering/outputs/visual-decay-20261030/gray448.png')

def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count() == 1
    config = json.loads((C.MODELS[a.model] / 'config.json').read_text())['model']
    rows = [json.loads(l) for l in open(a.rows) if l.strip()][:a.n]
    adapter = load_adapter(config); m = adapter.model
    for p in m.parameters(): p.requires_grad_(False)
    cap = {}
    adapter.layers[a.layer - 1].register_forward_hook(lambda mod, args, o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o)[0].detach().float()))
    A, late = [], []
    for r in rows:
        pred = r['prediction']
        br = adapter.prepare(r['image'], r['prompt'], answer=pred); bg = adapter.prepare(str(GRAY), r['prompt'], answer=pred)
        lab = br.pop('labels')[0]; bg.pop('labels'); ids = br['input_ids'][0]
        cs = int((lab != -100).nonzero()[0]); T = len(ids) - cs
        if T < 40: continue
        q = torch.arange(cs - 1, len(ids) - 1, device=adapter.device)
        with torch.no_grad():
            m(**br, use_cache=False); hr = cap['h'][q]; m(**bg, use_cache=False); hg = cap['h'][q]
        d = hr - hg; ai = d[1:17].mean(0); A.append(ai.cpu())
        lt = d[64:192] if T > 70 else d[32:]
        late.append(float(torch.nn.functional.cosine_similarity(lt, ai[None], dim=-1).mean()))
    A = torch.stack(A); An = A / A.norm(dim=-1, keepdim=True); S = An @ An.T; n = len(A)
    off = S[~torch.eye(n, dtype=bool)]; mu = A.mean(0)
    print(f'{a.model} L{a.layer}: {n} images | mean pairwise cos(a_i,a_j) {float(off.mean()):.3f} | |mean a|/mean|a| {float(mu.norm()/A.norm(dim=-1).mean()):.3f}'
          f' | cos(a_i, mean a) {float((An @ (mu / mu.norm())).mean()):.3f} | late cos(d_t, a_i) {np.mean(late):.3f}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--rows', required=True)
    p.add_argument('--layer', type=int, required=True); p.add_argument('--n', type=int, default=60)
    main(p.parse_args())
