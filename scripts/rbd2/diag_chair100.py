"""Per-image breakdown of the CHAIR test-100 runs (same cached unmodified scorer as score_chair100.py):
objects mentioned per caption, image flips vs Base, GT objects lost/gained, hallucinations removed/added."""
import collections,json,tempfile
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module

R=Path('/home/project-b/LJW/vlm-grounding-steering');SRC=R/'upstream/chair/chair.py';O=R/'outputs/rbd-test-pilot-20261101/chair'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
ids=[r['metadata']['image_id'] for r in read('/workspace/chair_test100.jsonl')]
c=json.loads(Path('/workspace/rbd_pilot/chair-test100-scorer-state.json').read_text())
mod=_load_external_module(SRC,'chair_diag');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
hw=lambda s:{h[1] if isinstance(h,(list,tuple)) else h for h in s['mscoco_hallucinated_words']}
S={};T={}
for n in ['base','rbd01','const01','unif01']:
    rows={r['metadata']['image_id']:r for r in read(O/f'qwen-{n}-0-100.jsonl')}
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:json.dump([{'image_id':i,'caption':rows[i]['prediction']} for i in ids],f)
    S[n]={x['image_id']:x for x in sc.compute_chair(f.name,'image_id','caption')['sentences']};T[n]=rows
for n in S:
    s=S[n];gen=[len(set(s[i]['mscoco_generated_words'])) for i in ids];hal=[len(hw(s[i])) for i in ids]
    hit=[len(set(s[i]['mscoco_gt_words'])&set(s[i]['mscoco_generated_words'])) for i in ids]
    print(f"{n:8s} distinct COCO objs/caption {np.mean(gen):.2f}  correct {np.mean(hit):.2f}  halluc {np.mean(hal):.2f}  images w/ halluc {sum(h>0 for h in hal)}  tokens {np.mean([T[n][i]['generated_tokens'] for i in ids]):.0f}  @512 {sum(T[n][i]['generated_tokens']>=512 for i in ids)}")
b=S['base']
for n in ['rbd01','const01','unif01']:
    s=S[n];fixed=sum(bool(hw(b[i])) and not hw(s[i]) for i in ids);broke=sum(not hw(b[i]) and bool(hw(s[i])) for i in ids)
    lost=collections.Counter();gain=collections.Counter();hrem=collections.Counter();hadd=collections.Counter()
    for i in ids:
        gt=set(b[i]['mscoco_gt_words']);bh=set(b[i]['mscoco_generated_words'])&gt;sh=set(s[i]['mscoco_generated_words'])&gt
        lost.update(bh-sh);gain.update(sh-bh);hrem.update(hw(b[i])-hw(s[i]));hadd.update(hw(s[i])-hw(b[i]))
    print(f"\n{n} vs base: halluc images fixed {fixed} / newly broken {broke}")
    print(f"  GT objects lost {sum(lost.values())} gained {sum(gain.values())}; top lost {lost.most_common(8)}")
    print(f"  halluc objects removed {sum(hrem.values())} added {sum(hadd.values())}; top removed {hrem.most_common(8)}; top added {hadd.most_common(6)}")
r,u=S['rbd01'],S['unif01'];both=sum(bool(hw(r[i]))==bool(hw(u[i])) for i in ids)
print(f"\nrbd01 vs unif01: same halluc/clean status on {both}/100 images; identical captions {sum(T['rbd01'][i]['prediction']==T['unif01'][i]['prediction'] for i in ids)}")
for n in ['rbd01','unif01']:
    print(n,'identical to base:',sum(T[n][i]['prediction']==T['base'][i]['prediction'] for i in ids))
