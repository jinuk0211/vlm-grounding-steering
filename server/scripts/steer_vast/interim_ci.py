"""Paired bootstrap (2,000 over images, seed 0) of FULL vs each REF on the images all runs already have: CHAIR_s, CHAIR_i, recall.
usage: interim_ci.py SET STATE_JSON FULL=rows.jsonl REF=rows.jsonl ...   (SET file in $RBD2_SETS)"""
import hashlib, json, sys, tempfile
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module

SET, STATE = sys.argv[1], sys.argv[2]
refs = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(open(STATE).read())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_ci'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
runs = {}
for a in sys.argv[3:]:
    lab, p = a.split('=', 1); runs[lab] = {r['id']: r for r in S.read(p)}
sub = [r for r in refs if all(r['id'] in v for v in runs.values())]; ids = [r['metadata']['image_id'] for r in sub]
per = {}
for lab, rows in runs.items():
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in sub], f)
    s = {x['image_id']: x for x in sc.compute_chair(f.name, 'image_id', 'caption')['sentences']}
    per[lab] = {'hal': np.array([len(s[i]['mscoco_hallucinated_words']) > 0 for i in ids], float),
                'nh': np.array([len(s[i]['mscoco_hallucinated_words']) for i in ids], float),
                'ng': np.array([len(s[i]['mscoco_generated_words']) for i in ids], float),
                'gt': np.array([len(set(s[i]['mscoco_gt_words'])) for i in ids], float),
                'hit': np.array([len(set(s[i]['mscoco_gt_words']) & set(s[i]['mscoco_generated_words'])) for i in ids], float)}
rng = np.random.default_rng(0); B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
m = lambda x, k: (100 * x['hal'][k].mean(), 100 * x['nh'][k].sum() / x['ng'][k].sum(), 100 * x['hit'][k].sum() / x['gt'][k].sum())
full = list(runs)[0]; F = per[full]; allk = np.arange(len(ids))
print(f'{len(ids)} images common to all runs; one image = {100/len(ids):.1f} CHAIR_s points')
for lab in list(runs)[1:]:
    X = per[lab]; d = np.array([np.subtract(m(F, k), m(X, k)) for k in B]); q = np.quantile(d, [.025, .975], axis=0)
    f0, x0 = m(F, allk), m(X, allk)
    print(f'{full} - {lab:12s} CHAIR_s {f0[0]-x0[0]:+5.1f} [{q[0,0]:+.1f},{q[1,0]:+.1f}]  CHAIR_i {f0[1]-x0[1]:+5.2f} [{q[0,1]:+.2f},{q[1,1]:+.2f}]'
          f'  recall {f0[2]-x0[2]:+5.1f} [{q[0,2]:+.1f},{q[1,2]:+.1f}]')
