"""Score the steering screen and apply the selection rule fixed in PROTOCOL.md before any run finished.

Every run in RUNDIR named TAG-START-END.jsonl that covers all rows of the set is scored with the repo's unmodified CHAIR
scorer state (score_sets.chair) and compared, paired bootstrap 2,000 over images (seed 0), with
  'nosteer'  belief + penalty, no steering (the -steer ablation row)   -> primary comparison
  'full'     the paper's steering (vdd_decode.py, L16 alpha 2)
Selection rule (PROTOCOL.md): among the variants (not full / nosteer) whose recall >= nosteer recall - 1.0, the lowest
CHAIR_s; ties by CHAIR_i. A variant is carried to confirmation only if its CHAIR_s is below full's.
usage: RBD2_SETS=DIR steer_score.py SET RUNDIR START END     (SET e.g. chair_test100; rows from DIR/SET.jsonl)
"""
import glob, json, sys
from pathlib import Path

import numpy as np
import score_sets as S

SET, RUNDIR, START, END = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
S.SET = SET
S.STATE.setdefault(SET, S.STATE['chair_test60'])
refs = S.read(S.SETS / f'{SET}.jsonl')[START:END]; ids = [r['id'] for r in refs]
runs, info = {}, {}
for f in sorted(glob.glob(str(RUNDIR / f'*-{START}-{END}.jsonl'))):
    tag = Path(f).name[:-len(f'-{START}-{END}.jsonl')]; rows = {r['id']: r for r in S.read(f)}
    if not all(i in rows for i in ids): print('incomplete, skipped:', tag); continue
    runs[tag] = rows
    info[tag] = {'tokens': float(np.mean([rows[i]['generated_tokens'] for i in ids])), 'at512': int(sum(rows[i]['generated_tokens'] >= 512 for i in ids)),
                 'setting': rows[ids[0]].get('setting')}
assert 'nosteer' in runs and 'full' in runs, 'need complete nosteer and full runs'
out = {}
for ref in ('nosteer', 'full'):
    S.rng = np.random.default_rng(0)
    res = S.chair(refs, {'base': runs[ref], **{k: v for k, v in runs.items() if k != ref}})
    out[f'vs_{ref}'] = {('ref:' + ref if k == 'base' else k): v for k, v in res.items()}
V = out['vs_nosteer']; ns = V['ref:nosteer']; full = V['full']
cands = [t for t in V if t not in ('ref:nosteer', 'full') and V[t]['recall'] >= ns['recall'] - 1.0]
pick = min(cands, key=lambda t: (V[t]['CHAIR_s'], V[t]['CHAIR_i'])) if cands else None
out['selection'] = {'rule': 'recall >= nosteer - 1.0, then lowest CHAIR_s, ties by CHAIR_i; confirm only if CHAIR_s < full',
                    'eligible': cands, 'picked': pick, 'carry_to_confirmation': bool(pick and V[pick]['CHAIR_s'] < full['CHAIR_s'])}
out['info'] = info
(RUNDIR / f'steer_score-{START}-{END}.json').write_text(json.dumps(out, indent=1))
print(f"{SET} rows {START}-{END}, n={len(ids)}; d* = run minus nosteer [95% CI]")
for t, v in V.items():
    ci = v.get('ci', {}); fv = out['vs_full'].get(t, {}).get('ci', {}).get('CHAIR_s')
    print(f"{t:14s} CHAIR_s {v['CHAIR_s']:5.1f} CHAIR_i {v['CHAIR_i']:5.2f} recall {v['recall']:5.1f} tok {info.get(t.replace('ref:', ''), {}).get('tokens', 0):5.0f}"
          + (f"  dCHAIR_s[{ci['CHAIR_s'][0]:+.1f},{ci['CHAIR_s'][1]:+.1f}] drecall[{ci['recall'][0]:+.1f},{ci['recall'][1]:+.1f}]" if ci else '')
          + (f"  vs full dCHAIR_s[{fv[0]:+.1f},{fv[1]:+.1f}]" if fv else ''))
print('selection:', json.dumps(out['selection']))
