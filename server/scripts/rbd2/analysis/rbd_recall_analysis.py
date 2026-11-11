"""Where does RBD lose CHAIR recall? CPU only, existing rows (CHAIR 500, RBD row vs its vdd Base), unmodified CHAIR word matching.
usage: rbd_recall_analysis.py"""
import collections,hashlib,json,re
from pathlib import Path
import numpy as np
from vlm_grounding.evaluation import _load_external_module
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';X=O/'runs'
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_rec');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
GT={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];IM={r['id']:r['metadata']['image_id'] for r in refs};ids=[r['id'] for r in refs]
rd=lambda p:{r['id']:r for r in map(json.loads,open(p))}
def men(cap):
    """node word -> first mention word index in the tokenised caption; total words"""
    w,n,i,raw=sc.caption_to_words(cap);f={}
    for a,b in zip(n,i):f.setdefault(a,b)
    return f,len(raw)
STOP=set('a an the of and in on at to with is are was were it its this that there which as by for from or be has have can appears seems also while their his her they he she into near some other two three one'.split())
def words(cap):return [w for w in re.findall(r"[a-z]+",cap.lower()) if w not in STOP and len(w)>2]
def account(name,B,D,detail=True):
    k=collections.Counter();pos=collections.defaultdict(collections.Counter);rank=collections.defaultdict(collections.Counter);cl=collections.defaultdict(collections.Counter);why=collections.Counter()
    for i in ids:
        im=IM[i];g=GT[im];fb,lb=men(B[i]['prediction']);fd,ld=men(D[i]['prediction'])
        k['gt']+=len(g)
        order=sorted((v,o) for o,v in fb.items() if o in g)
        rk={o:r for r,(v,o) in enumerate(order)}
        for o in g:
            hb,hd=o in fb,o in fd;cl[o]['gt']+=1
            if hb:k['hit_b']+=1;cl[o]['hit_b']+=1
            if hd:k['hit_d']+=1;cl[o]['hit_d']+=1
            bk=min([x for x in (20,50,100,200,10**9) if fb.get(o,0)<x]) if hb else None
            if hb and hd:k['kept']+=1;pos[bk]['kept']+=1;rank[min(rk[o],3)]['kept']+=1
            if hb and not hd:
                k['lost']+=1;cl[o]['lost']+=1;pos[bk]['lost']+=1;rank[min(rk[o],3)]['lost']+=1
                why['rbd_ended_before_base_position' if ld<=fb[o] else 'rbd_long_enough']+=1
                why['lost_in_rbd@512' if D[i]['generated_tokens']>=512 else 'lost_rbd<512']+=1
            if hd and not hb:k['gained']+=1;cl[o]['gained']+=1
        hb={o for o in fb if o not in g};hd={o for o in fd if o not in g}
        for o in fb:
            if o in g:cl[o]['men_true']+=1
            else:cl[o]['men_hal']+=1
        k['hal_b']+=len(hb);k['hal_d']+=len(hd);k['hal_removed']+=len(hb-hd);k['hal_new']+=len(hd-hb)
        for o in hb:
            bk=min([x for x in (20,50,100,200,10**9) if fb[o]<x])
            pos[bk]['hal_rem' if o not in hd else 'hal_kept']+=1;cl[o]['hal_removed']+=int(o not in hd)
        for o in hd-hb:cl[o]['hal_new']+=1
        k['cap_hal_b']+=bool(hb);k['cap_hal_d']+=bool(hd)
    print(f"\n#### {name}")
    print(f"recall {100*k['hit_b']/k['gt']:.1f} -> {100*k['hit_d']/k['gt']:.1f}; true objects lost {k['lost']} gained {k['gained']} (net {k['gained']-k['lost']} of {k['gt']}); hallucinated (image,class) {k['hal_b']} -> {k['hal_d']} (removed {k['hal_removed']}, new {k['hal_new']}); captions with hal {k['cap_hal_b']} -> {k['cap_hal_d']}; hal removed per true lost {k['hal_removed']/max(k['lost'],1):.2f}, net hal per net true {(k['hal_b']-k['hal_d'])/max(k['lost']-k['gained'],1):.2f}")
    if not detail:return
    print('Base first-mention word index: bucket kept lost lost% | hal removed hal kept removed%')
    for b in (20,50,100,200,10**9):
        p=pos[b];print(f"  <{b:<10d} {p['kept']:5d} {p['lost']:4d} {100*p['lost']/max(p['kept']+p['lost'],1):5.1f}% | {p['hal_rem']:4d} {p['hal_kept']:4d} {100*p['hal_rem']/max(p['hal_rem']+p['hal_kept'],1):5.1f}%")
    print('Mention order among the true objects in Base: rank kept lost lost%')
    for b in range(4):
        p=rank[b];print(f"  {b+1 if b<3 else '4+'}  {p['kept']:5d} {p['lost']:4d} {100*p['lost']/max(p['kept']+p['lost'],1):5.1f}%")
    print('Lost object:',dict(why))
    print('Classes by net true lost: class gt hit_b hit_d lost gained | hal_removed hal_new | Base mention precision')
    rows=sorted(cl.items(),key=lambda x:x[1]['gained']-x[1]['lost'])
    for o,v in rows[:14]:
        print(f"  {o:13s} {v['gt']:4d} {v['hit_b']:4d} {v['hit_d']:4d} {v['lost']:3d} {v['gained']:3d} | {v['hal_removed']:3d} {v['hal_new']:3d} | {100*v['men_true']/max(v['men_true']+v['men_hal'],1):5.1f}%")
    print('Classes grouped by Base mention precision: group classes lost gained hal_removed hal_new')
    for lo,hi in ((.9,1.01),(.7,.9),(.4,.7),(0,.4)):
        g=[v for v in cl.values() if v['men_true']+v['men_hal']>=3 and lo<=v['men_true']/(v['men_true']+v['men_hal'])<hi]
        print(f"  [{lo:.1f},{hi:.2f}) n={len(g):2d} lost {sum(v['lost'] for v in g):4d} gained {sum(v['gained'] for v in g):4d} hal_removed {sum(v['hal_removed'] for v in g):4d} hal_new {sum(v['hal_new'] for v in g):4d}")
    cb=collections.Counter();cd=collections.Counter();nb=nd=0
    for i in ids:
        wb,wd=words(B[i]['prediction']),words(D[i]['prediction']);cb.update(wb);cd.update(wd);nb+=len(wb);nd+=len(wd)
    lo={w:np.log((cd[w]+5)/(nd+5*len(cb)))-np.log((cb[w]+5)/(nb+5*len(cb))) for w in set(cb)|set(cd) if cb[w]+cd[w]>=30}
    s=sorted(lo,key=lo.get);print('Word shift, RBD down:',[(w,cb[w],cd[w]) for w in s[:22]]);print('Word shift, RBD up:',[(w,cb[w],cd[w]) for w in s[::-1][:22]])
    at=[(i,D[i]) for i in ids if D[i]['generated_tokens']>=512];bt=[i for i in ids if B[i]['generated_tokens']>=512]
    def rep(cap):
        s=[x.strip() for x in re.split(r'(?<=[.!?\n])\s+',cap) if len(x.strip())>15];return 1-len(set(s))/max(len(s),1)
    print(f"@512: Base {len(bt)} RBD {len(at)}; repeated-sentence share in RBD@512 mean {np.mean([rep(r['prediction']) for _,r in at]) if at else 0:.2f}, RBD<512 mean {np.mean([rep(D[i]['prediction']) for i in ids if D[i]['generated_tokens']<512]):.2f}, Base@512 {np.mean([rep(B[i]['prediction']) for i in bt]) if bt else 0:.2f}")
    def sub(sel):
        h=[bool({o for o in men(D[i]['prediction'])[0] if o not in GT[IM[i]]}) for i in sel];hb=[bool({o for o in men(B[i]['prediction'])[0] if o not in GT[IM[i]]}) for i in sel];return 100*np.mean(h) if sel else 0,100*np.mean(hb) if sel else 0
    s1=[i for i,_ in at];s2=[i for i in ids if D[i]['generated_tokens']<512];print(f"CHAIR_s on RBD@512 images: RBD {sub(s1)[0]:.1f} Base {sub(s1)[1]:.1f} (n {len(s1)}); on RBD<512 images: RBD {sub(s2)[0]:.1f} Base {sub(s2)[1]:.1f} (n {len(s2)})")
    if at:print('tail of a RBD@512 caption:',repr(at[0][1]['prediction'][-260:]))
Q=[('Qwen2.5-VL RBD s16a2bl1p05',X/'chair_test500_all/base-0-500.jsonl',X/'chair_test500_all/s16a2bl1p05-0-500.jsonl'),
   ('Gemma E4B RBD s24a1bl1p05',X/'gchair_test500_all/base-0-500.jsonl',X/'gchair_test500_all/s24a1bl1p05-0-500.jsonl'),
   ('Qwen3-VL RBD s20a1bl1p05',X/'qchair_test500_all/base-0-500.jsonl',X/'qchair_test500_all/s20a1bl1p05-0-500.jsonl')]
for n,b,d in Q:account(n,rd(b),rd(d))
print('\n==== Qwen2.5 ablation (accounting only)')
b=rd(X/'chair_abl500_all/base-0-500.jsonl')
for t in ('abl_steer','abl_pen','abl_belief','abl_nosteer','abl_nopen','abl_nobelief','abl_const','rbd_full'):account(t,b,rd(X/f'chair_abl500_all/{t}-0-500.jsonl'),False)
print('ANALYSIS_DONE')
