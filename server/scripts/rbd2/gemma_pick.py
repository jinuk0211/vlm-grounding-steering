"""Pick our Gemma setting on the 60 dev images and write the test queue for it (gemma_chain.sh).
Rule (fixed before any dev score): highest 2-judge THRONE F0.5 (pb_throne_partial.py, flan-t5 base+large, threshold 6)
among the candidates below, ties to the earlier one; s24a1bl2p1 first = Qwen's THRONE pick (layer 16/28) at the same
relative depth of Gemma's 42 layers. Writes queues/gemma_ours.txt: 20 units of 50 test images.
env as pb_throne_partial.py with RBD2_THRONE_SET=throne_gdev60 RBD2_N=60."""
import re,subprocess,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102');DEV=O/'runs/gemma_throne_dev60'
CAND={'s24a1bl2p1':'--alpha 1 --layer 24 --lam 2 --pen 1','s20a1bl2p1':'--alpha 1 --layer 20 --lam 2 --pen 1',
      's28a1bl2p1':'--alpha 1 --layer 28 --lam 2 --pen 1','bl2p1':'--lam 2 --pen 1'}
out=subprocess.run([sys.executable,str(Path(__file__).parent/'pb_throne_partial.py'),str(DEV),'base',*CAND],
                   capture_output=True,text=True,check=True).stdout
print(out,end='')
f={m[1]:float(m[2]) for m in re.finditer(r'^(\S+)\s+P .*F0\.5\s+([\d.]+)',out,re.M)}
assert set(CAND)<=set(f),f
pick=max(CAND,key=lambda t:(f[t],-list(CAND).index(t)))
(O/'queues/gemma_ours.txt').write_text(f'# dev pick {pick} (F0.5 {f[pick]:.1f}, base {f["base"]:.1f})\n'+''.join(
    f'vdd throne_full1000 gemma_throne1000_parts {pick} {b} {b+50} {CAND[pick]}\n' for b in range(0,1000,50)))
print('picked',pick)
