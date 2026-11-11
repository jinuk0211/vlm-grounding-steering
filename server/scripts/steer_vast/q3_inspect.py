"""Qwen3: where do Full (anchor) and -steering differ? Per caption CHAIR hallucinated words with their word position, and the
caption format (markdown headers / bullets). usage: q3_inspect.py FULL.jsonl NOSTEER.jsonl BASE.jsonl"""
import json, re, sys, tempfile, hashlib
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module
S.SET = 'chair_test500'; S.STATE.setdefault(S.SET, S.STATE['chair_test60'])
rd = lambda p: {r['id']: r for r in S.read(p)}
F, N, B = rd(sys.argv[1]), rd(sys.argv[2]), rd(sys.argv[3])
refs = [r for r in S.read(S.SETS / 'chair_test500.jsonl') if r['id'] in F and r['id'] in N]
src = S.R / 'upstream/chair/chair.py'; c = json.loads(S.STATE[S.SET].read_text())
mod = _load_external_module(src, 'chair_q3'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
def per(rows):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in refs], f)
    d = sc.compute_chair(f.name, 'image_id', 'caption'); return {x['image_id']: x for x in d['sentences']}
pf, pn, pb = per(F), per(N), per(B)
S.rng = np.random.default_rng(0); r = S.chair(refs, {'base': N, 'full': F})
print(f"{len(refs)} images; Full CHAIR_s {r['full']['CHAIR_s']:.1f} vs -steering {r['base']['CHAIR_s']:.1f}, d CI {r['full']['ci']['CHAIR_s']}")
fmt = lambda t: (len(re.findall(r'^#+ ', t, re.M)), len(re.findall(r'^\s*[-*] ', t, re.M)), len(re.findall(r'\*\*', t)) // 2)
for name, rows in (('Full', F), ('-steer', N), ('Base', B)):
    x = np.array([fmt(rows[r['id']]['prediction']) for r in refs]); print(f'{name:7s} headers {x[:,0].mean():.1f} bullets {x[:,1].mean():.1f} bold {x[:,2].mean():.1f} words {np.mean([len(rows[r["id"]]["prediction"].split()) for r in refs]):.0f}')
nf = nn = 0
for rr in refs:
    i = rr['metadata']['image_id']; hf = [w[1] for w in pf[i]['mscoco_hallucinated_words']]; hn = [w[1] for w in pn[i]['mscoco_hallucinated_words']]
    if bool(hf) != bool(hn):
        if hf: nf += 1
        else: nn += 1
        if nf + nn <= 6:
            print(f"--- {rr['id']} Full hal {sorted(set(hf))} | -steer hal {sorted(set(hn))} | gt {sorted(sc.imid_to_objects[i])[:8]}")
            t = F[rr['id']]['prediction'] if hf else N[rr['id']]['prediction']; w = sorted(set(hf or hn))[0]
            k = t.lower().find(w); print('   ...' + t[max(0, k - 160):k + 60].replace('\n', ' / ') + '...')
print(f'captions hallucinating only under Full: {nf}, only under -steering: {nn}')
