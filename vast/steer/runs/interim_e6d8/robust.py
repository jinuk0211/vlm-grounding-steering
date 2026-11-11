"""Is the Qwen2.5 anchor-steering result luck? Paired bootstrap (2,000 over images, seed 0, score_sets.chair) of Full vs references,
on all 500 rows and on each half (rows 0-249, 250-499).
usage: robust.py FULL=path REF1=path REF2=path ...   (Full first; each REF is compared with Full)"""
import sys
import numpy as np
import score_sets as S
S.SET = 'chair_test500'; S.STATE.setdefault(S.SET, S.STATE['chair_test60'])
refs_all = S.read(S.SETS / 'chair_test500.jsonl')
runs = {}
for a in sys.argv[1:]:
    t, p = a.split('=', 1); runs[t] = {r['id']: r for r in S.read(p)}
full = list(runs)[0]
for lo, hi in ((0, 500), (0, 250), (250, 500)):
    refs = refs_all[lo:hi]
    print(f'--- rows {lo}-{hi}')
    for ref in list(runs)[1:]:
        S.rng = np.random.default_rng(0)
        r = S.chair(refs, {'base': runs[ref], full: runs[full]})
        b, v = r['base'], r[full]; ci = v['ci']
        print(f"{full} vs {ref:22s} CHAIR_s {v['CHAIR_s']:5.1f} vs {b['CHAIR_s']:5.1f} d[{ci['CHAIR_s'][0]:+.1f},{ci['CHAIR_s'][1]:+.1f}]"
              f"  CHAIR_i {v['CHAIR_i']:5.2f} vs {b['CHAIR_i']:5.2f}  recall {v['recall']:5.1f} vs {b['recall']:5.1f} d[{ci['recall'][0]:+.1f},{ci['recall'][1]:+.1f}]")
