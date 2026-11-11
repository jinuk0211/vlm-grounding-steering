"""Paired bootstrap (2,000 over images, seed 0, score_sets.chair) of FULL vs each reference, on the images FULL and all refs have.
usage: ci_interim.py FULL=path REF=path ...   (RBD2_SETS, RBD2_CHAIR_TEST_STATE set; set chair_test500)"""
import sys
import numpy as np
import score_sets as S
S.SET = 'chair_test500'; S.STATE.setdefault(S.SET, S.STATE['chair_test60'])
runs = {}
for a in sys.argv[1:]:
    t, p = a.split('=', 1); runs[t] = {r['id']: r for r in S.read(p)}
full = list(runs)[0]
refs = [r for r in S.read(S.SETS / 'chair_test500.jsonl') if all(r['id'] in v for v in runs.values())]
print('images', len(refs))
for name in list(runs)[1:]:
    S.rng = np.random.default_rng(0)
    r = S.chair(refs, {'base': runs[name], full: runs[full]}); v = r[full]; b = r['base']; c = v['ci']
    print(f"{full} vs {name:12s} CHAIR_s {v['CHAIR_s']:5.1f} vs {b['CHAIR_s']:5.1f} d[{c['CHAIR_s'][0]:+.1f},{c['CHAIR_s'][1]:+.1f}]"
          f"  recall {v['recall']:5.1f} vs {b['recall']:5.1f} d[{c['recall'][0]:+.1f},{c['recall'][1]:+.1f}]")
