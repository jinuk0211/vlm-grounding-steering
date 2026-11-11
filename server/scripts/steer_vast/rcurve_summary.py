"""Merge rcurve_collect.py shards into a decay summary with the fields rbd_common.r_curve reads (E3_causal.by_bin.gray.kl, the
same binning and per-caption-then-mean aggregation as visual_decay_analyze.py), plus per-layer ||d||/||h|| by bin.
Prints r(t) next to the stored Qwen2.5 and Gemma curves.
usage: rcurve_summary.py DIR MODEL OUT.json"""
import glob, json, re, sys
from pathlib import Path
import numpy as np

LAB = ['0-15', '16-31', '32-63', '64-95', '96-127', '128-191', '192-255', '256-383', '384-512']
d, model, outp = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
recs = [r for f in sorted(glob.glob(str(d / f'{model}-*-*.json'))) if re.fullmatch(rf'{model}-\d+-\d+\.json', Path(f).name)
        for r in json.load(open(f))]
assert recs and len({r['id'] for r in recs}) == len(recs)
M = np.array([[np.nan if v is None else v for v in r['kl_bins']] for r in recs], float)
kl = []
for j, l in enumerate(LAB):
    c = M[:, j]; k = int((~np.isnan(c)).sum())
    if k: kl.append({'bin': l, 'mean': float(np.nanmean(c)), 'n': k})
Hn = np.array([[[np.nan if v is None else v for v in lay] for lay in r['h_norm_bins']] for r in recs], float)
Dn = np.array([[[np.nan if v is None else v for v in lay] for lay in r['d_norm_bins']] for r in recs], float)
ratio = np.nanmean(Dn, 0) / np.nanmean(Hn, 0)                     # [layers, bins]
s = {'model': model, 'captions': len(recs), 'caption_length': {'mean': float(np.mean([r['T'] for r in recs]))}, 'bins': LAB,
     'E3_causal': {'by_bin': {'gray': {'kl': kl}}}, 'd_over_h_by_layer_bin': ratio.tolist(),
     'note': 'rcurve_collect.py (gray condition of visual_decay_collect.py, teacher-forced stored Base captions)'}
outp.write_text(json.dumps(s, indent=1))
print(f'{model}: {len(recs)} captions, mean length {s["caption_length"]["mean"]:.0f}')
ref = {}
for m in ('qwen', 'gemma'):
    p = Path(f'/home/project-b/LJW/vlm-grounding-steering/outputs/visual-decay-20261030/{m}-visual-decay-summary.json')
    ref[m] = {x['bin']: x['mean'] for x in json.load(open(p))['E3_causal']['by_bin']['gray']['kl']}
new = {x['bin']: x['mean'] for x in kl}
print(f"{'bin':8s} {'KL '+model:>10s} {'r '+model:>8s} | {'r qwen2.5':>9s} {'r gemma':>8s}")
for l in LAB:
    f = lambda D: (D[l] / D['0-15']) if l in D else float('nan')
    print(f"{l:8s} {new.get(l, float('nan')):10.3f} {f(new):8.3f} | {f(ref['qwen']):9.3f} {f(ref['gemma']):8.3f}")
print('||d||/||h|| per layer, bins 0-15 / 64-95 / 256-383:')
for i in range(ratio.shape[0]):
    print(f'  L{i+1:2d} ' + ' '.join(f'{ratio[i, j]:.3f}' for j in (0, 3, 7)))
