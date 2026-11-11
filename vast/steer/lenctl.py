"""Length control for the steering screen: does a variant's lower CHAIR_s come from saying less?
For each variant V and each image, nosteer's caption is cut to V's caption length (whitespace words, cut at the last
sentence end at or before that length when one exists, else at the word limit) and scored with the same CHAIR state.
V is then compared, paired bootstrap 2,000 over images (seed 0), against that length-matched nosteer.
usage: lenctl.py SET START END TAG=RUNFILE ... (nosteer must be one of the tags)
"""
import json, re, sys
import numpy as np
import score_sets as S

SET, START, END = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S.SET = SET; S.STATE.setdefault(SET, S.STATE['chair_test60'])
refs = S.read(S.SETS / f'{SET}.jsonl')[START:END]; ids = [r['id'] for r in refs]
runs = {}
for a in sys.argv[4:]:
    t, f = a.split('=', 1); rows = {r['id']: r for r in S.read(f)}; assert all(i in rows for i in ids), t; runs[t] = rows
ns = runs.pop('nosteer')

def cut(text, n):
    w = text.split()
    if len(w) <= n: return text
    head = ' '.join(w[:n]); m = list(re.finditer(r'[.!?](\s|$)', head))
    return head[:m[-1].end()].strip() if m else head

nw = lambda r: len(r['prediction'].split())
print(f'{SET} rows {START}-{END}; nosteer words/caption {np.mean([nw(ns[i]) for i in ids]):.0f}')
for t, rows in runs.items():
    trunc = {i: {**ns[i], 'prediction': cut(ns[i]['prediction'], nw(rows[i]))} for i in ids}
    S.rng = np.random.default_rng(0)
    r = S.chair(refs, {'base': trunc, t: rows, 'nosteer_full_length': ns})
    b, v = r['base'], r[t]; ci = v['ci']
    print(f"{t:10s} words {np.mean([nw(rows[i]) for i in ids]):4.0f} | variant CHAIR_s {v['CHAIR_s']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f}"
          f" | nosteer cut to same length CHAIR_s {b['CHAIR_s']:5.1f} CHAIR_i {b['CHAIR_i']:5.2f} recall {b['recall']:5.1f}"
          f" | variant - cut nosteer dCHAIR_s[{ci['CHAIR_s'][0]:+.1f},{ci['CHAIR_s'][1]:+.1f}] drecall[{ci['recall'][0]:+.1f},{ci['recall'][1]:+.1f}]")
