"""Confirmation scorer (PROTOCOL.md, rounds 2-3). Every complete TAG-START-END.jsonl in RUNDIR is scored with the unmodified
CHAIR state and compared with 'nosteer' (and 'full' when present): paired bootstrap 2,000 over images (seed 0), 95% and
Bonferroni 97.5% CIs for CHAIR_s, CHAIR_i and recall, plus the length-matched control (nosteer cut to the run's per-image
length, as in lenctl.py). With --pooled DIR0, full and nosteer are also pooled with DIR0's rows 0..START-1 of the same set.
usage: cscore.py SET RUNDIR START END [--pooled DIR0]
"""
import glob, hashlib, json, re, sys, tempfile
from pathlib import Path
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module

SET, RUNDIR, START, END = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
POOL = Path(sys.argv[sys.argv.index('--pooled') + 1]) if '--pooled' in sys.argv else None
S.SET = SET; S.STATE.setdefault(SET, S.STATE['chair_test60'])
ALL = S.read(S.SETS / f'{SET}.jsonl')
src = S.R / 'upstream/chair/chair.py'; c = json.loads(S.STATE[SET].read_text())
assert c['built_with_unmodified_constructor'] and c['scorer_sha256'] == hashlib.sha256(src.read_bytes()).hexdigest()
mod = _load_external_module(src, 'chair_cscore'); sc = mod.CHAIR.__new__(mod.CHAIR); sc.__dict__.update(c['state'])
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
    ci_ok = bool(abs(a['nhw'].sum() / a['ngw'].sum() - m['CHAIRi']) < 1e-9)
    met = {'CHAIR_s': 100 * m['CHAIRs'], 'CHAIR_i': 100 * m['CHAIRi'], 'recall': 100 * m['Recall'],
           'tokens': float(np.mean([rows[r['id']]['generated_tokens'] for r in refs])),
           'words': float(np.mean([len(rows[r['id']]['prediction'].split()) for r in refs])), 'chair_i_arrays_match': ci_ok}
    return met, a

def boot(x, b, n):
    B = np.random.default_rng(0).integers(0, n, (2000, n))
    ds = 100 * (x['hal'][B].mean(1) - b['hal'][B].mean(1))
    di = 100 * (x['nhw'][B].sum(1) / x['ngw'][B].sum(1) - b['nhw'][B].sum(1) / b['ngw'][B].sum(1))
    dr = 100 * (x['n_hit'][B].sum(1) / x['n_gt'][B].sum(1) - b['n_hit'][B].sum(1) / b['n_gt'][B].sum(1))
    q = lambda v: {'95': [float(np.quantile(v, .025)), float(np.quantile(v, .975))], '97.5': [float(np.quantile(v, .0125)), float(np.quantile(v, .9875))]}
    return {'CHAIR_s': q(ds), 'CHAIR_i': q(di), 'recall': q(dr)}

def cut(text, n):
    w = text.split()
    if len(w) <= n: return text
    head = ' '.join(w[:n]); m = list(re.finditer(r'[.!?](\s|$)', head))
    return head[:m[-1].end()].strip() if m else head

def load(f, ids):
    rows = {r['id']: r for r in S.read(f)}
    return rows if all(i in rows for i in ids) else None

refs = ALL[START:END]; ids = [r['id'] for r in refs]; n = len(ids)
runs = {}
for f in sorted(glob.glob(str(RUNDIR / f'*-{START}-{END}.jsonl'))):
    t = Path(f).name[:-len(f'-{START}-{END}.jsonl')]; rows = load(f, ids)
    if rows: runs[t] = rows
    else: print('incomplete, skipped:', t)
out = {'set': SET, 'rows': [START, END], 'n': n, 'runs': {}}
fmt = lambda c, k: f"[{c[k]['95'][0]:+.1f},{c[k]['95'][1]:+.1f}] (97.5%: [{c[k]['97.5'][0]:+.1f},{c[k]['97.5'][1]:+.1f}])"
print(f'{SET} rows {START}-{END - 1}, n={n}. Differences are run minus reference, 95% CI (Bonferroni 97.5% CI).')
if 'nosteer' not in runs: print('nosteer not complete yet'); sys.exit(0)
mn, an = score(refs, runs['nosteer']); arrs = {'nosteer': an}
print(f"nosteer        CHAIR_s {mn['CHAIR_s']:5.1f} CHAIR_i {mn['CHAIR_i']:5.2f} recall {mn['recall']:5.1f} tok {mn['tokens']:4.0f} words {mn['words']:4.0f}")
out['runs']['nosteer'] = {'metrics': mn}
for t in sorted(runs, key=lambda t: (t != 'full', t)):
    if t == 'nosteer': continue
    m, a = score(refs, runs[t]); arrs[t] = a; vs = boot(a, an, n)
    trunc = {i: {**runs['nosteer'][i], 'prediction': cut(runs['nosteer'][i]['prediction'], len(runs[t][i]['prediction'].split()))} for i in ids}
    mc, ac = score(refs, trunc); vc = boot(a, ac, n)
    rec = {'metrics': m, 'vs_nosteer': vs, 'cut_nosteer_metrics': mc, 'vs_cut_nosteer': vc}
    print(f"{t:14s} CHAIR_s {m['CHAIR_s']:5.1f} CHAIR_i {m['CHAIR_i']:5.2f} recall {m['recall']:5.1f} tok {m['tokens']:4.0f} words {m['words']:4.0f}")
    print(f"   vs nosteer      dCHAIR_s {fmt(vs, 'CHAIR_s')}  dCHAIR_i {fmt(vs, 'CHAIR_i')}  drecall {fmt(vs, 'recall')}")
    print(f"   vs cut nosteer  (CHAIR_s {mc['CHAIR_s']:.1f}, recall {mc['recall']:.1f})  dCHAIR_s {fmt(vc, 'CHAIR_s')}  drecall {fmt(vc, 'recall')}")
    if t != 'full' and 'full' in arrs:
        vf = boot(a, arrs['full'], n); rec['vs_full'] = vf
        print(f"   vs full         dCHAIR_s {fmt(vf, 'CHAIR_s')}  drecall {fmt(vf, 'recall')}")
    out['runs'][t] = rec
if POOL and 'full' in runs:
    prefs = ALL[:END]; pids = [r['id'] for r in prefs]; pooled = {}
    for t in ('nosteer', 'full'):
        r0 = load(POOL / f'{t}-0-{START}.jsonl', [r['id'] for r in ALL[:START]])
        if r0: pooled[t] = {**r0, **runs[t]}
    if len(pooled) == 2:
        (mf, af), (mo, ao) = score(prefs, pooled['full']), score(prefs, pooled['nosteer']); vp = boot(af, ao, len(pids))
        out['pooled_full_vs_nosteer'] = {'full': mf, 'nosteer': mo, 'diff': vp}
        print(f"pooled rows 0-{END - 1} (n={len(pids)}): full CHAIR_s {mf['CHAIR_s']:.1f} CHAIR_i {mf['CHAIR_i']:.2f} recall {mf['recall']:.1f} | "
              f"nosteer {mo['CHAIR_s']:.1f} / {mo['CHAIR_i']:.2f} / {mo['recall']:.1f} | dCHAIR_s {fmt(vp, 'CHAIR_s')} drecall {fmt(vp, 'recall')}")
if not all(r['metrics']['chair_i_arrays_match'] for r in out['runs'].values()): print('NOTE: per-image CHAIR_i arrays do not reproduce CHAIRi')
(RUNDIR / f'cscore-{START}-{END}.json').write_text(json.dumps(out, indent=1))
