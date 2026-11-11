"""Why does steering move CHAIR_s on one model and not on another? Per-caption comparison of run A (with steering) and run B (without)
on the images both have: captions with a hallucination and flips between the runs, hallucinated / all object mentions, GT objects hit,
caption length, word index of the first hallucinated object.
usage: flip.py SET STATE_JSON A=rows.jsonl B=rows.jsonl      (SET file in $RBD2_SETS)"""
import hashlib, json, sys, tempfile
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module

SET, STATE = sys.argv[1], sys.argv[2]
refs = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(open(STATE).read())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_flip'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
runs = {}
for a in sys.argv[3:5]:
    lab, p = a.split('=', 1); runs[lab] = {r['id']: r for r in S.read(p)}
sub = [r for r in refs if all(r['id'] in v for v in runs.values())]; ids = [r['metadata']['image_id'] for r in sub]; n = len(sub)
sent = {}
for lab, rows in runs.items():
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in sub], f)
    sent[lab] = {x['image_id']: x for x in sc.compute_chair(f.name, 'image_id', 'caption')['sentences']}
print(f'{n} images')
hal = {}
for lab, rows in runs.items():
    s = sent[lab]; hal[lab] = np.array([len(s[i]['mscoco_hallucinated_words']) > 0 for i in ids])
    nh = sum(len(s[i]['mscoco_hallucinated_words']) for i in ids); ng = sum(len(s[i]['mscoco_generated_words']) for i in ids)
    hit = sum(len(set(s[i]['mscoco_gt_words']) & set(s[i]['mscoco_generated_words'])) for i in ids); gt = sum(len(set(s[i]['mscoco_gt_words'])) for i in ids)
    words = [len(rows[r['id']]['prediction'].split()) for r in sub]; pos = []
    for r in sub:
        x = s[r['metadata']['image_id']]; cap = rows[r['id']]['prediction'].lower()
        ps = [cap.find(w[0] if isinstance(w, (list, tuple)) else w) for w in x['mscoco_hallucinated_words']]; ps = [p for p in ps if p >= 0]
        if ps: pos.append(len(cap[:min(ps)].split()))
    h = np.histogram(pos, bins=[0, 50, 100, 150, 200, 10**6])[0]
    print(f"{lab:12s} captions with hallucination {int(hal[lab].sum()):3d} ({100*hal[lab].mean():.1f}%) | per caption: hallucinated mentions {nh/n:.2f}, "
          f"correct mentions {(ng-nh)/n:.2f}, all {ng/n:.2f} | CHAIR_i {100*nh/ng:.2f} recall {100*hit/gt:.1f} | words {np.mean(words):.0f}"
          f" | first hallucinated word at index: median {np.median(pos) if pos else float('nan'):.0f}, <50/50-100/100-150/150-200/200+ = {list(h)}")
a, b = list(runs)
print(f"flips: hallucination only in {a} {int((hal[a] & ~hal[b]).sum())}, only in {b} {int((~hal[a] & hal[b]).sum())}, "
      f"both {int((hal[a] & hal[b]).sum())}, neither {int((~hal[a] & ~hal[b]).sum())}")
