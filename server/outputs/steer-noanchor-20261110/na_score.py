"""No-anchor steering search scorer. usage: na_score.py MODEL START END
Scores every run in outputs/steer-noanchor-20261110/MODEL with TAG-START-END.jsonl (partial runs: on the rows done, marked)
against 'ns' (same weak belief, no steering) on the common rows: CHAIR_s / CHAIR_i / recall / tokens, paired bootstrap 2,000
(seed 0) 95% CI, and the length-matched control (ns cut to the run's per-image word count). Stored 500-row references
(base, rbd_full, abl_nosteer, abl_steer) are scored on the same rows for context (not compared)."""
import glob, hashlib, json, re, sys, tempfile
from pathlib import Path
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module
MODEL, START, END = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
R = Path('/home/project-b/LJW/vlm-grounding-steering'); D = R / 'outputs/steer-noanchor-20261110' / MODEL
STORED = R / 'outputs/rbd2-20261102/runs' / {'qwen': 'chair_abl500_all', 'gemma': 'gabl_chair500_all'}[MODEL]
ALL = S.read(S.SETS / 'chair_test500.jsonl')[START:END]
src = S.R / 'upstream/chair/chair.py'; c = json.loads((R / 'data/hallucination-suite/chair-scorer-state.json').read_text())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_na'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
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
    tok = np.mean([rows[r['id']].get('generated_tokens', 0) for r in refs])
    return (100 * m['CHAIRs'], 100 * m['CHAIRi'], 100 * m['Recall'], tok), a
def boot(x, b):
    n = len(x['hal']); B = np.random.default_rng(0).integers(0, n, (2000, n))
    q = lambda v: (np.quantile(v, .025), np.quantile(v, .975))
    return (q(100 * (x['hal'][B].mean(1) - b['hal'][B].mean(1))),
            q(100 * (x['nhw'][B].sum(1) / x['ngw'][B].sum(1) - b['nhw'][B].sum(1) / b['ngw'][B].sum(1))),
            q(100 * (x['n_hit'][B].sum(1) / x['n_gt'][B].sum(1) - b['n_hit'][B].sum(1) / b['n_gt'][B].sum(1))))
def cut(text, n):
    w = text.split()
    if len(w) <= n: return text
    head = ' '.join(w[:n]); m = list(re.finditer(r'[.!?](\s|$)', head))
    return head[:m[-1].end()].strip() if m else head
runs = {}; want = {r['id'] for r in ALL}
for f in sorted(glob.glob(str(D / '*.jsonl'))):          # merge all chunk files TAG-B-E.jsonl of a tag, keep rows START..END-1
    mm = re.match(r'^(.*)-(\d+)-(\d+)\.jsonl$', Path(f).name)
    if not mm: continue
    rows = {r['id']: r for r in S.read(f) if r['id'] in want}
    if rows: runs.setdefault(mm.group(1), {}).update(rows)
if 'ns' not in runs: print('no ns yet'); sys.exit()
ids = [r['id'] for r in ALL if r['id'] in runs['ns']]
print(f'{MODEL} chair_test500 rows {START}-{END-1}; ns has {len(ids)} rows. d = run - ns [95% CI]; cut = ns cut to run length')
fmt = lambda v: f'{v[0]:6.1f} {v[1]:6.2f} {v[2]:5.1f} {v[3]:4.0f}'
for t in ['base', 'rbd_full', 'abl_nosteer', 'abl_steer']:
    rows = {r['id']: r for r in S.read(STORED / f'{t}-0-500.jsonl')}; refs = [r for r in ALL if r['id'] in ids]
    print(f'  stored {t:12s} {fmt(score(refs, rows)[0])}')
for t in sorted(runs, key=lambda t: (t != 'ns', t)):
    common = [r for r in ALL if r['id'] in runs[t] and r['id'] in runs['ns']]
    if not common: continue
    m, a = score(common, runs[t]); mn, an = score(common, runs['ns'])
    tag = f'{t}' + ('' if len(common) == END - START else f' [{len(common)}]')
    if t == 'ns': print(f'{tag:18s} {fmt(m)}'); continue
    v = boot(a, an); cutns = {r['id']: {**runs['ns'][r['id']], 'prediction': cut(runs['ns'][r['id']]['prediction'], len(runs[t][r['id']]['prediction'].split()))} for r in common}
    mc, ac = score(common, cutns); vc = boot(a, ac)
    print(f'{tag:18s} {fmt(m)} | ns {mn[0]:5.1f} {mn[2]:5.1f} | dS [{v[0][0]:+.0f},{v[0][1]:+.0f}] dI [{v[1][0]:+.1f},{v[1][1]:+.1f}] dR [{v[2][0]:+.1f},{v[2][1]:+.1f}] | cut S {mc[0]:.1f} R {mc[2]:.1f} dS [{vc[0][0]:+.0f},{vc[0][1]:+.0f}]')
