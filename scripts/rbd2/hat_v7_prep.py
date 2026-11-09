"""Second RBD selection on HAT, for ALOHa (hat_v7_chain.sh; user 11-08: raise HAT's ALOHa, select on 100 of HAT's 385 images, GPUs 0 and 1 this
time only). Qwen2.5, sdpa, colour-fixed V_obj, vdd7_decode.py (its defaults = vdd4_decode.py). Candidates from the 11-08 notes: belief/penalty
strength, the blurred image as counterfactual (--cf blur), evidence weighted by the model's own probability (--ew prob), layers 14 and 18 next to 16.
Rows 0-99 select (they hold rows 0-59 of the 11-05 selection), rows 100-384 are left for the table; bench Base, vdd Base and the baselines exist on all rows.
  queue_sel     queues/hat385_v7sel.txt: the candidates on rows 0-99
  pick JSON     rule as on 11-05 (hat_prep.py): ALOHa above bench Base and vdd Base, RefCov >= vdd Base - 2; highest ALOHa, within 0.5 of it the
                highest RefCov; none -> no further decoding. Writes hat385/pick-qwen-v7.txt
  queue_test    queues/hat385_v7test.txt: the pick on rows 100-384"""
import json,os,sys
from pathlib import Path
L=Path(os.environ.get('HAT_L','/home/project-b/LJW'));R=L/'vlm-grounding-steering';O=R/'outputs/rbd2-20261102';SEL=100;N=385;H=O/'hat385'
V=f'--vobj {R}/outputs/rbd-20261030/collect/qwen-vobj2cf.json'
def s(alpha,lam,pen,layer=16,x=''):return f'--attn sdpa --layer {layer} --alpha {alpha} --lam {lam} --pen {pen} {V} {x}'.rstrip()
CAND={'v7_a2bl1p05':s(2,1,0.5),'v7_a2bl2p1':s(2,2,1),'v7_blur_a2bl1p05':s(2,1,0.5,x='--cf blur'),'v7_blur_a2bl2p1':s(2,2,1,x='--cf blur'),
      'v7_pw_a2bl1p05':s(2,1,0.5,x='--ew prob'),'v7_pw_a2bl2p1':s(2,2,1,x='--ew prob'),'v7_blurpw_a2bl2p1':s(2,2,1,x='--cf blur --ew prob'),
      'v7_s14a2bl1p05':s(2,1,0.5,14),'v7_s18a2bl1p05':s(2,1,0.5,18)}
def queue(name,lines):p=O/'queues'/f'hat385_{name}.txt';p.write_text('\n'.join(lines)+'\n',newline='\n');print(p.name,len(lines),'units')

def pick(js):
    r=json.loads(Path(js).read_text())['results'];b,v=r['base'],r['vddbase'];c={t:r[t] for t in CAND if t in r}
    if len(c)<len(CAND):print('candidates without complete rows, left out:',sorted(set(CAND)-set(c)))
    elig={t:x for t,x in c.items() if x['ALOHa']>b['ALOHa'] and x['ALOHa']>v['ALOHa'] and x['RefCov']>=v['RefCov']-2}
    if not elig:
        (H/'pick-qwen-v7.txt').write_text('none\n')
        print(f"picked none: best ALOHa {max((x['ALOHa'] for x in c.values()),default=float('nan')):.1f}, bench Base {b['ALOHa']:.1f}, vdd Base {v['ALOHa']:.1f} (RefCov floor {v['RefCov']-2:.1f}); not continued");return
    top=max(x['ALOHa'] for x in elig.values());near=[t for t,x in elig.items() if x['ALOHa']>=top-.5];p=max(near,key=lambda t:elig[t]['RefCov'])
    (H/'pick-qwen-v7.txt').write_text(p+'\n')
    print(f"picked {p} ALOHa {elig[p]['ALOHa']:.1f} RefCov {elig[p]['RefCov']:.1f} (bench Base {b['ALOHa']:.1f}, vdd Base {v['ALOHa']:.1f}/{v['RefCov']:.1f}); eligible {list(elig)}; within 0.5: {near}")

if __name__=='__main__':
    c=sys.argv[1]
    if c=='queue_sel':queue('v7sel',[f'qwen vdd7 {t} {a} {a+50} {x}' for a in (0,50) for t,x in CAND.items()])
    elif c=='pick':pick(sys.argv[2])
    else:
        p=(H/'pick-qwen-v7.txt').read_text().strip();assert p in CAND,p
        queue('v7test',[f'qwen vdd7 {p} {a} {min(a+50,N)} {CAND[p]}' for a in range(SEL,N,50)])
