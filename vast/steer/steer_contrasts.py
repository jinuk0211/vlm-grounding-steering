"""Steering's effect in the paper's Qwen CHAIR 500 ablation: the 4 pairs that differ only in steering on/off.
Same scorer as cscore.py (unmodified CHAIR state, paired bootstrap 2,000, seed 0), length-matched control = the
no-steering caption cut to the steered caption's per-image word count. Rows 0-499 and rows 150-499 (full's setting
was chosen on rows 0-149). usage: steer_contrasts.py ABLDIR"""
import hashlib, json, re, sys, tempfile
from pathlib import Path
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module
SET = 'chair_test500'; D = Path(sys.argv[1])
S.SET = SET; S.STATE.setdefault(SET, S.STATE['chair_test60'])
ALL = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(S.STATE[SET].read_text())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_sc'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
def score(refs, rows):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in refs], f)
    d = sc.compute_chair(f.name, 'image_id', 'caption'); m = d['overall_metrics']
    s = {x['image_id']: x for x in d['sentences']}; im = [r['metadata']['image_id'] for r in refs]
    a = {'hal': np.array([len(s[i]['mscoco_hallucinated_words']) > 0 for i in im], float),
         'nhw': np.array([len(s[i]['mscoco_hallucinated_words']) for i in im], float),
         'ngw': np.array([len(s[i]['mscoco_generated_words']) for i in im], float),
         'n_gt': np.array([len(set(s[i]['mscoco_gt_words'])) for i in im], float),
         'n_hit': np.array([len(set(s[i]['mscoco_gt_words']) & set(s[i]['mscoco_generated_words'])) for i in im], float)}
    return {'CHAIR_s': 100 * m['CHAIRs'], 'CHAIR_i': 100 * m['CHAIRi'], 'recall': 100 * m['Recall'],
            'words': float(np.mean([len(rows[r['id']]['prediction'].split()) for r in refs]))}, a
def boot(x, b, n):
    B = np.random.default_rng(0).integers(0, n, (2000, n))
    ds = 100 * (x['hal'][B].mean(1) - b['hal'][B].mean(1))
    di = 100 * (x['nhw'][B].sum(1) / x['ngw'][B].sum(1) - b['nhw'][B].sum(1) / b['ngw'][B].sum(1))
    dr = 100 * (x['n_hit'][B].sum(1) / x['n_gt'][B].sum(1) - b['n_hit'][B].sum(1) / b['n_gt'][B].sum(1))
    q = lambda v: (float(np.mean(v)), float(np.quantile(v, .025)), float(np.quantile(v, .975)))
    return {'CHAIR_s': q(ds), 'CHAIR_i': q(di), 'recall': q(dr)}
def cut(text, n):
    w = text.split()
    if len(w) <= n: return text
    head = ' '.join(w[:n]); m = list(re.finditer(r'[.!?](\s|$)', head))
    return head[:m[-1].end()].strip() if m else head
rd = lambda t: {r['id']: r for r in S.read(D / f'{t}-0-500.jsonl')}
PAIRS = [('rbd_full', 'abl_nosteer', 'belief+pen  +steer'), ('abl_nopen', 'abl_belief', 'belief      +steer'),
         ('abl_nobelief', 'abl_pen', 'pen         +steer'), ('abl_steer', 'base', 'none        +steer')]
f = lambda v: f"{v[0]:+5.1f} [{v[1]:+5.1f},{v[2]:+5.1f}]{'*' if v[1] > 0 or v[2] < 0 else ' '}"
out = {}
for lo, hi in ((0, 500), (150, 500)):
    refs = ALL[lo:hi]; ids = [r['id'] for r in refs]; n = len(ids)
    print(f'\n=== rows {lo}-{hi - 1} (n={n}); d = with steering minus without, 95% CI, * = excludes 0')
    print(f"{'pair':22s} {'CHAIR_s on/off':>15s} {'recall on/off':>14s} {'words':>9s} | {'dCHAIR_s':>20s} {'dCHAIR_i':>20s} {'drecall':>20s} | length-matched: {'dCHAIR_s':>20s} {'drecall':>20s}")
    for on, off, lab in PAIRS:
        A, B = rd(on), rd(off); assert all(i in A and i in B for i in ids)
        ma, aa = score(refs, A); mb, ab = score(refs, B); v = boot(aa, ab, n)
        Bc = {i: {**B[i], 'prediction': cut(B[i]['prediction'], len(A[i]['prediction'].split()))} for i in ids}
        mc, ac = score(refs, Bc); vc = boot(aa, ac, n)
        print(f"{lab:22s} {ma['CHAIR_s']:6.1f}/{mb['CHAIR_s']:<6.1f}  {ma['recall']:6.1f}/{mb['recall']:<6.1f} {ma['words']:4.0f}/{mb['words']:<4.0f} | "
              f"{f(v['CHAIR_s'])} {f(v['CHAIR_i'])} {f(v['recall'])} | (cut off: {mc['CHAIR_s']:.1f}/{mc['recall']:.1f}) {f(vc['CHAIR_s'])} {f(vc['recall'])}")
        out[f'{lo}-{hi}:{on}-vs-{off}'] = {'on': ma, 'off': mb, 'off_cut': mc, 'd': v, 'd_lenmatched': vc}
Path('/workspace/steer/runs/steer_contrasts.json').write_text(json.dumps(out, indent=1))
