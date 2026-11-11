"""CHAIR table for steering runs, point estimates (unmodified CHAIR scorer from a cached constructor state, as score_sets.py).
usage: steer_table.py SET STATE_JSON LABEL=ROWS.jsonl ...   (SET file in $RBD2_SETS; a run missing an image of the set is reported, not scored)"""
import hashlib, json, sys, tempfile
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module

SET, STATE = sys.argv[1], sys.argv[2]
refs = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(open(STATE).read())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_tab'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
print(f"{SET}, {len(refs)} images\n{'row':16s} CHAIR_s CHAIR_i recall    F1  tok @512")
for arg in sys.argv[3:]:
    lab, p = arg.split('=', 1)
    try: rows = {r['id']: r for r in S.read(p)}
    except FileNotFoundError: print(f'{lab:16s} missing file'); continue
    miss = sum(r['id'] not in rows for r in refs)
    if miss: print(f'{lab:16s} {miss} images missing'); continue
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in refs], f)
    m = sc.compute_chair(f.name, 'image_id', 'caption')['overall_metrics']; P = 1 - m['CHAIRi']; Rc = m['Recall']
    tok = [rows[r['id']]['generated_tokens'] for r in refs]
    print(f"{lab:16s} {100*m['CHAIRs']:7.1f} {100*m['CHAIRi']:7.2f} {100*Rc:6.1f} {100*2*P*Rc/(P+Rc):5.1f} {np.mean(tok):4.0f} {sum(t >= 512 for t in tok):4d}")
