"""Gray-image reliance curve r(t) of one backbone, computed as visual_decay_collect.py's E3 'gray' condition: teacher forcing of the
stored Base caption with the real image and with the 448x448 gray image, per caption step t (query capstart-1+t) the KL
sum p_real (log p_real - log p_gray), clamped at 0. In the same two passes, for every decoder layer and step: ||h_real|| and
||h_real - h_gray|| (the size of the steering signal d_t relative to the hidden state).
Per caption it stores the KL per step and per-layer bin means; rcurve_summary.py turns shards into a decay summary.
usage: rcurve_collect.py --model M --rows BASE.jsonl --start S --end E --output DIR [--attn eager]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from vlm_grounding.backend import load_adapter
import read_gap_collect_v2 as C

BINS = [0, 16, 32, 64, 96, 128, 192, 256, 384, 513]           # visual_decay_analyze.py
GRAY = Path('/home/project-b/LJW/vlm-grounding-steering/outputs/visual-decay-20261030/gray448.png')

def bin_means(v):
    v = np.asarray(v, float)
    return [float(np.nanmean(v[a:b])) if a < len(v) else None for a, b in zip(BINS, BINS[1:])]

def main(a):
    assert os.environ['CUDA_VISIBLE_DEVICES'] and torch.cuda.device_count() == 1
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    config = json.loads((C.MODELS[a.model] / 'config.json').read_text())['model']
    if a.attn: config = dict(config, attn_implementation=a.attn)
    rows = [json.loads(l) for l in open(a.rows) if l.strip()][a.start:a.end]
    tag = f'{a.model}-{a.start}-{a.end}'; status = out / f'{tag}-status.json'
    adapter = load_adapter(config); m = adapter.model
    for p in m.parameters(): p.requires_grad_(False)
    layers = adapter.layers; NL = len(layers); H = [None] * NL; on = [False]
    def hook(i):
        def f(mod, args, o):
            if on[0]: H[i] = (o[0] if isinstance(o, tuple) else o)[0].detach()
        return f
    for i, l in enumerate(layers): l.register_forward_hook(hook(i))
    recs = []; t0 = time.time()
    status.write_text(json.dumps({'stage': 'running', 'model': a.model, 'config': config, 'n_layers': NL, 'gray': str(GRAY),
                                  'rows': str(a.rows), 'gpu': torch.cuda.get_device_name(0), 'argv': sys.argv}, indent=1))
    for n, r in enumerate(rows):
        pred = r['prediction']
        if not pred.strip(): continue
        br = adapter.prepare(r['image'], r['prompt'], answer=pred); bg = adapter.prepare(str(GRAY), r['prompt'], answer=pred)
        assert torch.equal(br['input_ids'], bg['input_ids']); bg.pop('labels')
        lab = br.pop('labels')[0]; ids = br['input_ids'][0]
        capstart = int((lab != -100).nonzero()[0]); L = len(ids); T = L - capstart
        q = torch.arange(capstart - 1, L - 1, device=adapter.device)
        with torch.no_grad():
            on[0] = True; o = m(**br, use_cache=False, return_dict=True); on[0] = False
            lpb = o.logits[0, q].float().log_softmax(-1); Hr = [h[q].float() for h in H]; del o
            on[0] = True; o = m(**bg, use_cache=False, return_dict=True); on[0] = False
            lp = o.logits[0, q].float().log_softmax(-1); del o
            kl = (lpb.exp() * (lpb - lp)).sum(-1).clamp_min(0).cpu().numpy()
            hn = np.array([bin_means(hr.norm(dim=-1).cpu().numpy()) for hr in Hr], dtype=object)
            dn = np.array([bin_means((hr - H[i][q].float()).norm(dim=-1).cpu().numpy()) for i, hr in enumerate(Hr)], dtype=object)
            del Hr, lpb, lp
        recs.append({'id': r['id'], 'T': T, 'kl': kl.astype(np.float32).tolist(), 'kl_bins': bin_means(kl),
                     'h_norm_bins': hn.tolist(), 'd_norm_bins': dn.tolist()})
        if n % 10 == 0:
            s = json.loads(status.read_text()); s.update(done=len(recs), sec_per_caption=(time.time() - t0) / len(recs)); status.write_text(json.dumps(s, indent=1))
    (out / f'{tag}.json').write_text(json.dumps(recs))
    s = json.loads(status.read_text()); s.update(stage='complete', done=len(recs), sec_per_caption=(time.time() - t0) / max(len(recs), 1)); status.write_text(json.dumps(s, indent=1))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', choices=list(C.MODELS), required=True); p.add_argument('--rows', required=True)
    p.add_argument('--start', type=int, default=0); p.add_argument('--end', type=int, default=500); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attn', default=None)
    main(p.parse_args())
