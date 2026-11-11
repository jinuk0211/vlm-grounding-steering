"""Ablation table (point estimates only, no bootstrap): CHAIR_s, CHAIR_i, recall, mean generated tokens on chair_test500.
usage: abl_table.py LABEL=PATH ...   (each PATH a 500-row jsonl)"""
import hashlib, json, sys, tempfile
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module
SET = 'chair_test500'; S.SET = SET; S.STATE.setdefault(SET, S.STATE['chair_test60'])
refs = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(S.STATE[SET].read_text())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_tab'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
rows_out = []
for arg in sys.argv[1:]:
    lab, p = arg.split('=', 1); rows = {r['id']: r for r in S.read(p)}
    assert all(r['id'] in rows for r in refs), (lab, 'incomplete')
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in refs], f)
    m = sc.compute_chair(f.name, 'image_id', 'caption')['overall_metrics']
    tok = float(np.mean([rows[r['id']]['generated_tokens'] for r in refs]))
    rows_out.append((lab, 100 * m['CHAIRs'], 100 * m['CHAIRi'], 100 * m['Recall'], tok))
print(f"{'row':28s} CHAIR_s CHAIR_i recall  tok")
for lab, s, i, r, t in rows_out: print(f'{lab:28s} {s:7.1f} {i:7.2f} {r:6.1f} {t:4.0f}')
print('\nLaTeX:')
for lab, s, i, r, t in rows_out: print(f'    {lab:26s} & {s:.1f} & {i:.1f} & {r:.1f} & {t:.0f} \\')
