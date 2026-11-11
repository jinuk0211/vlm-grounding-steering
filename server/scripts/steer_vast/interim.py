"""Interim CHAIR table on the images that every given run already has (in set order), point estimates.
usage: interim.py SET STATE_JSON LABEL=ROWS.jsonl ...   (SET file in $RBD2_SETS)"""
import hashlib, json, sys, tempfile
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module

SET, STATE = sys.argv[1], sys.argv[2]
refs = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(open(STATE).read())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_int'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
runs = {}
for a in sys.argv[3:]:
    lab, p = a.split('=', 1)
    try: runs[lab] = {r['id']: r for r in S.read(p)}
    except FileNotFoundError: pass
sub = [r for r in refs if all(r['id'] in v for v in runs.values())]
print(f"{SET}: {len(sub)} images common to all runs\n{'row':18s} CHAIR_s CHAIR_i recall    F1  tok")
for lab, rows in runs.items():
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in sub], f)
    m = sc.compute_chair(f.name, 'image_id', 'caption')['overall_metrics']; P = 1 - m['CHAIRi']; Rc = m['Recall']
    print(f"{lab:18s} {100*m['CHAIRs']:7.1f} {100*m['CHAIRi']:7.2f} {100*Rc:6.1f} {100*2*P*Rc/(P+Rc):5.1f} {np.mean([rows[r['id']]['generated_tokens'] for r in sub]):4.0f}")
