"""Qwen2.5-VL CHAIR ablation of the paper's RBD row (user 11-04: after the running jobs).
Full RBD = s16a2bl1p05: steering alpha 2 at layer 16 + belief lam 1 + penalty 0.5, decay schedules, measured r(t), raw
evidence, sdpa, old V_obj (runs/chair_test500_all; CHAIR_s 25.6 / CHAIR_i 6.86 / recall 68.9 / 178 tok).
Variants (one change each, everything else as the full row):
  abl_nosteer   alpha 0              (belief + penalty)
  abl_nobelief  lam 0                (steering + penalty)
  abl_nopen     pen 0                (steering + belief)        rows 0-59 reused from runs/chair_test60_sweep/s16a2bl1
  abl_steer     steering only        (alpha 2)
  abl_belief    belief only          (lam 1)                    rows 0-149 reused from runs/chair_test150_all/bl1
  abl_pen       penalty only         (pen 0.5)
  abl_const     schedules constant   (--bsched const --ssched const: w(t) = s(t) = 1 instead of 1 - r(t))
Rows are reused only where the stored setting is identical. vdd Base and the full row come from runs/chair_test500_all.
  queue      queues/abl_chair.txt (units of 50; reused prefixes skipped)
  join       runs/chair_abl500_all/<tag>-0-500.jsonl (base = vdd Base, rbd_full, variants) and
             runs/chair_abl500_vsfull/ (base = the full RBD row, so the bootstrap gives each variant's change vs full)"""
import json,sys
from pathlib import Path
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
V={'abl_nosteer':'--lam 1 --pen 0.5','abl_nobelief':'--alpha 2 --layer 16 --pen 0.5','abl_nopen':'--alpha 2 --layer 16 --lam 1',
   'abl_steer':'--alpha 2 --layer 16','abl_belief':'--lam 1','abl_pen':'--pen 0.5',
   'abl_const':'--alpha 2 --layer 16 --lam 1 --pen 0.5 --bsched const --ssched const'}
REUSE={'abl_nopen':(O/'runs/chair_test60_sweep/s16a2bl1-0-60.jsonl',60),'abl_belief':(O/'runs/chair_test150_all/bl1-0-150.jsonl',150)}
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def st(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};s.pop('vobj',None);return json.dumps(s,sort_keys=True)
ref=[r['id'] for r in read(O/'sets/chair_test500.jsonl')];assert len(ref)==500
def spans(t):
    a=REUSE[t][1] if t in REUSE else 0;b=[a]+list(range((a//50+1)*50,500,50)) if a%50 else list(range(a,500,50));return [(s,min(s+50,500) if s%50==0 else (s//50+1)*50) for s in b]
cmd=sys.argv[1]
if cmd=='queue':
    q=['# Qwen CHAIR ablation of RBD s16a2bl1p05 (abl_prep.py)']
    for t,a in V.items():
        for s,e in spans(t):q.append(f'chair_test500 chair_abl500 {t} {s} {e} --attn sdpa {a}')
    (O/'queues/abl_chair.txt').write_text('\n'.join(q)+'\n');print(len(q)-1,'units',{t:sum(e-s for s,e in spans(t)) for t in V})
elif cmd=='join':
    full=read(O/'runs/chair_test500_all/s16a2bl1p05-0-500.jsonl');base=read(O/'runs/chair_test500_all/base-0-500.jsonl')
    A=O/'runs/chair_abl500_all';Fd=O/'runs/chair_abl500_vsfull';A.mkdir(exist_ok=True);Fd.mkdir(exist_ok=True)
    def w(d,t,rows):(d/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    w(A,'base',base);w(A,'rbd_full',full);w(Fd,'base',full);w(Fd,'vddbase',base)
    for t in V:
        rows=[]
        if t in REUSE:rows=read(REUSE[t][0])
        for s,e in spans(t):
            f=O/f'runs/chair_abl500/{t}-{s}-{e}.jsonl';r=read(f) if f.exists() else [];assert len(r)==e-s,f'{f} incomplete';rows+=r
        assert [x['id'] for x in rows]==ref,t
        assert len({st(x) for x in rows})==1,f'{t}: settings differ between reused and new rows'
        w(A,t,rows);w(Fd,t,rows)
    print('joined',['base','rbd_full',*V])
