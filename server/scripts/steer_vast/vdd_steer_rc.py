"""vdd_steer.py with the reliance schedule r(t) read from another decay summary (env RCURVE_JSON), everything else unchanged.
User 11-10 21:05: Qwen3's stored schedule outputs/visual-decay-20261030/qwen3-visual-decay-summary.json is a byte copy of the Qwen2.5
one (md5 488bf465), never measured; rcurve_collect.py measured Qwen3's own (steer-qwen3diag-20261110/rcurve). The patched r_curve
uses rbd_common's own interpolation code on the given file; kinds other than 'measured' are unchanged.
usage: RCURVE_JSON=summary.json vdd_steer_rc.py <vdd_steer.py arguments>"""
import json, os, runpy, sys
from pathlib import Path
import numpy as np
import rbd_common

P = Path(os.environ['RCURVE_JSON']); _orig = rbd_common.r_curve
def r_curve(model, kind='measured'):
    if kind != 'measured': return _orig(model, kind)
    s = json.loads(P.read_text())
    kl = {r['bin']: r['mean'] for r in s['E3_causal']['by_bin']['gray']['kl']}
    keys = [k for k in (f'{a}-{b}' for a, b in rbd_common.BINS) if k in kl]
    xs = [(int(k.split('-')[0]) + int(k.split('-')[1])) / 2 for k in keys]; ys = np.array([kl[k] for k in keys]); ys = ys / ys[0]
    return lambda t: float(np.interp(t, xs, ys))
rbd_common.r_curve = r_curve
f = r_curve('x'); print('RCURVE', P, 'r(t) at 0, 8, 24, 100, 300, 450:', [round(f(t), 3) for t in (0, 8, 24, 100, 300, 450)], flush=True)
sys.argv = [str(Path(__file__).with_name('vdd_steer.py'))] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
