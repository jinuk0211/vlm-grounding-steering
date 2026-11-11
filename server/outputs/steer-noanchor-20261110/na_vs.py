"""No-anchor steering search scorer. usage: na_vs.py MODEL START END RUN REF... (REF = our tag or stored:NAME)
Scores every run in outputs/steer-noanchor-20261110/MODEL with TAG-START-END.jsonl (partial runs: on the rows done, marked)
against 'ns' (same weak belief, no steering) on the common rows: CHAIR_s / CHAIR_i / recall / tokens, paired bootstrap 2,000
(seed 0) 95% CI, and the length-matched control (ns cut to the run's per-image word count). Stored 500-row references
(base, rbd_full, abl_nosteer, abl_steer) are scored on the same rows for context (not compared)."""
import glob, hashlib, json, re, sys, tempfile
from pathlib import Path
import numpy as np
import score_sets as S
from vlm_grounding.evaluation import _load_external_module
MODEL, START, END = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); RUN, REFS = sys.argv[4], sys.argv[5:]
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

def load(tag):
    want = {r['id'] for r in ALL}; rows = {}
    if tag.startswith('stored:'):
        return {r['id']: r for r in S.read(STORED / f"{tag[7:]}-0-500.jsonl") if r['id'] in want}
    for f in sorted(glob.glob(str(D / f'{tag}-*.jsonl'))):
        mm = re.match(r'^(.*)-(\d+)-(\d+)\.jsonl$', Path(f).name)
        if mm and mm.group(1) == tag: rows.update({r['id']: r for r in S.read(f) if r['id'] in want})
    return rows
run = load(RUN)
for ref in REFS:
    rr = load(ref); common = [r for r in ALL if r['id'] in run and r['id'] in rr]
    if not common: print(ref, 'no common rows'); continue
    m, a = score(common, run); mr, ar = score(common, rr); v = boot(a, ar)
    print(f"{RUN} vs {ref} on {len(common)} rows: run {m[0]:.1f}/{m[1]:.2f}/{m[2]:.1f}/{m[3]:.0f}  ref {mr[0]:.1f}/{mr[1]:.2f}/{mr[2]:.1f}/{mr[3]:.0f}  "
          f"dS [{v[0][0]:+.1f},{v[0][1]:+.1f}] dI [{v[1][0]:+.2f},{v[1][1]:+.2f}] dR [{v[2][0]:+.1f},{v[2][1]:+.1f}]")
