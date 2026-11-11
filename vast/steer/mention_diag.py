"""Per-caption decomposition of CHAIR runs: WHERE does a method change hallucination? (diagnosis, not selection)
Uses the CHAIR scorer's own caption_to_words (object mentions with their word index) and its gt objects; the aggregates are checked
against compute_chair. All runs are compared on the images they share.
  table 1  CHAIR_s / CHAIR_i / recall, tokens, CHAIR words, object mentions per caption (all, hallucinated, distinct correct)
  table 2  FIRST mentions per 100 captions by word-index bin (h = hallucinated, c = correct) and % of captions that reach the bin
  table 3  CHAIR_s / recall if every caption were cut at W words (same-length comparison)
  pairs    A:B -> captions hallucinating in both / only A / only B
usage: mention_diag.py SET STATE_JSON LABEL=rows.jsonl ... [--pairs A:B,C:D] [--step 50] [--bins 6]"""
import hashlib, json, sys, tempfile
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module

args = sys.argv[1:]; pairs = []; step = 50; nb = 6
for flag in ('--pairs', '--step', '--bins'):
    if flag in args:
        i = args.index(flag); v = args[i + 1]; del args[i:i + 2]
        if flag == '--pairs': pairs = [p.split(':') for p in v.split(',')]
        elif flag == '--step': step = int(v)
        else: nb = int(v)
SET, STATE = args[0], args[1]
refs = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(open(STATE).read())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_md'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
sc.imid_to_objects = {int(k): set(v) for k, v in c['state']['imid_to_objects'].items()}
runs = {}
for a in args[2:]:
    lab, p = a.split('=', 1); runs[lab] = {r['id']: r for r in S.read(p)}
sub = [r for r in refs if all(r['id'] in v for v in runs.values())]; N = len(sub)

def parse(rows):
    out = []
    for r in sub:
        gt = sc.imid_to_objects[r['metadata']['image_id']]
        words, nodes, idxs, raw = sc.caption_to_words(rows[r['id']]['prediction'])
        seen = set(); m = []
        for nd, ix in zip(nodes, idxs):
            m.append((ix, nd, nd not in gt, nd not in seen)); seen.add(nd)          # (word index, object, hallucinated, first mention)
        out.append({'m': m, 'len': len(raw), 'gt': gt, 'tok': rows[r['id']]['generated_tokens']})
    return out

def agg(P, W=None):
    hal = [any(h for ix, nd, h, f in p['m'] if W is None or ix < W) for p in P]
    nh = sum(h for p in P for ix, nd, h, f in p['m'] if W is None or ix < W)
    nm = sum(1 for p in P for ix, nd, h, f in p['m'] if W is None or ix < W)
    hit = sum(len({nd for ix, nd, h, f in p['m'] if not h and (W is None or ix < W)}) for p in P); gt = sum(len(p['gt']) for p in P)
    return 100 * np.mean(hal), 100 * nh / max(nm, 1), 100 * hit / gt, np.array(hal)

D = {}
print(f'{SET}: {N} images common to all runs')
print(f"{'row':22s} CHAIR_s CHAIR_i recall |  tok words | mentions/caption: all  hall  distinct-correct  distinct-hall")
for lab, rows in runs.items():
    P = parse(rows); D[lab] = P; s, i, rc, _ = agg(P)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump([{'image_id': r['metadata']['image_id'], 'caption': rows[r['id']]['prediction']} for r in sub], f)
    m = sc.compute_chair(f.name, 'image_id', 'caption')['overall_metrics']
    assert abs(100 * m['CHAIRs'] - s) < 1e-6 and abs(100 * m['CHAIRi'] - i) < 1e-6 and abs(100 * m['Recall'] - rc) < 1e-6, (lab, m, s, i, rc)
    nm = np.mean([len(p['m']) for p in P]); nh = np.mean([sum(h for _, _, h, _ in p['m']) for p in P])
    dc = np.mean([len({nd for _, nd, h, _ in p['m'] if not h}) for p in P]); dh = np.mean([len({nd for _, nd, h, _ in p['m'] if h}) for p in P])
    print(f"{lab:22s} {s:7.1f} {i:7.2f} {rc:6.1f} | {np.mean([p['tok'] for p in P]):4.0f} {np.mean([p['len'] for p in P]):5.0f} |"
          f"                 {nm:5.2f} {nh:5.2f} {dc:8.2f} {dh:14.2f}")

edges = [k * step for k in range(nb)] + [10 ** 9]
print(f"\nFIRST mentions per 100 captions by word-index bin (h hallucinated / c correct), and % of captions reaching the bin")
print(f"{'row':22s} " + ' '.join(f"{(str(edges[k]) + '-' + (str(edges[k + 1]) if k < nb - 1 else '')):>17s}" for k in range(nb)))
for lab, P in D.items():
    cells = []
    for k in range(nb):
        lo, hi = edges[k], edges[k + 1]
        h = sum(1 for p in P for ix, nd, hh, f in p['m'] if f and hh and lo <= ix < hi); cc = sum(1 for p in P for ix, nd, hh, f in p['m'] if f and not hh and lo <= ix < hi)
        cells.append(f"h{100 * h / N:5.1f} c{100 * cc / N:5.1f} {100 * np.mean([p['len'] > lo for p in P]):3.0f}%")
    print(f"{lab:22s} " + ' '.join(f'{x:>17s}' for x in cells))

print(f"\nCHAIR_s / recall if every caption is cut at W words")
print(f"{'row':22s} " + ' '.join(f"{('W=' + str(edges[k])):>11s}" for k in range(1, nb)) + f"{'full':>12s}")
for lab, P in D.items():
    print(f"{lab:22s} " + ' '.join(f"{a[0]:5.1f}/{a[2]:5.1f}" for a in [agg(P, edges[k]) for k in range(1, nb)] + [agg(P)]))

for a, b in pairs:
    ha, hb = agg(D[a])[3], agg(D[b])[3]
    print(f"\n{a} vs {b}: captions hallucinating in both {int((ha & hb).sum())}, only {a} {int((ha & ~hb).sum())}, only {b} {int((~ha & hb).sum())}, neither {int((~ha & ~hb).sum())}")
