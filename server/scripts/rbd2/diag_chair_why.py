"""CHAIR part of the AMBER 'why' diagnostic (11-05, CPU only): (1) flagged mentions by mention order for the SDPA Base
(runs/chair_test500_all/base) with the CHAIR scorer of the paper table; (5) the held-out split of the CHAIR 500: RBD
(s16a2bl1p05) vs vdd Base on rows 150-499 only (the setting was chosen on rows 0-149), plus rows 0-149 and all 500 for reference.
Paired image bootstrap 2000, seed 0, as score_sets.py."""
import collections,hashlib,json,os,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/project-b/LJW/vlm-grounding-steering/scripts/rbd2')
import score_sets as S
S.SET='chair_test500'
O=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
refs=S.read(S.SETS/'chair_test500.jsonl');assert len(refs)==500
def rows(p):return {r['id']:r for r in S.read(p)}
base=rows(O/'runs/chair_test500_all/base-0-500.jsonl');rbd=rows(O/'runs/chair_test500_all/s16a2bl1p05-0-500.jsonl')
# scorer as in score_sets.chair
from vlm_grounding.evaluation import _load_external_module
src=S.R/'upstream/chair/chair.py';c=json.loads(S.STATE[S.SET].read_text());assert c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_diag');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state']);sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
print('##### 1c. CHAIR 500, SDPA Base (vdd decoder at 0): flagged mentions by mention order')
with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:
    json.dump([{'image_id':r['metadata']['image_id'],'caption':base[r['id']]['prediction']} for r in refs],f)
d=sc.compute_chair(f.name,'image_id','caption');oc=collections.Counter();of=collections.Counter();n_caps_hal=0
for s in d['sentences']:
    gt=set(s['mscoco_gt_words']);fl=[w not in gt for w in s['mscoco_generated_words']];n_caps_hal+=int(any(fl))
    for k,x in enumerate(fl):oc[min(k,5)]+=1;of[min(k,5)]+=int(x)
tf=sum(of.values());print(f'   captions {len(d["sentences"])}, with a hallucination {n_caps_hal} (CHAIR_s {100*n_caps_hal/len(d["sentences"]):.1f}), mentions {sum(oc.values())}, flagged {tf}')
print('   mention order   mentions  flagged  flag rate  share of all flagged')
for b in range(6):print(f'   {"%d%s"%(b+1,"+" if b==5 else ""):4s}           {oc[b]:7d} {of[b]:8d}  {100*of[b]/max(oc[b],1):7.1f}%  {100*of[b]/max(tf,1):8.1f}%')
print('\n##### 5. CHAIR held-out: RBD s16a2bl1p05 vs vdd Base (paired bootstrap vs vdd Base)')
for lab,sl in (('rows 150-499 (350 images, NOT used to choose the setting)',slice(150,500)),('rows 0-149 (150 images, used to choose it)',slice(0,150)),('all 500',slice(0,500))):
    S.rng=np.random.default_rng(0);r=refs[sl];res=S.chair(r,{'base':base,'rbd':rbd})
    b,x=res['base'],res['rbd'];ci=x['ci']
    print(f'  {lab}: vdd Base CHAIR_s {b["CHAIR_s"]:.1f} CHAIR_i {b["CHAIR_i"]:.1f} recall {b["recall"]:.1f} | RBD CHAIR_s {x["CHAIR_s"]:.1f} CHAIR_i {x["CHAIR_i"]:.1f} recall {x["recall"]:.1f}'
          f' | dCHAIR_s [{ci["CHAIR_s"][0]:+.1f},{ci["CHAIR_s"][1]:+.1f}] drecall [{ci["recall"][0]:+.1f},{ci["recall"][1]:+.1f}]')
