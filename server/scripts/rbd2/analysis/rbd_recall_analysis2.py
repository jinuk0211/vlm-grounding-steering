"""RBD recall, second pass (CPU, existing rows): (1) how much of the CHAIR_s gain is class 'orange'; (2) per-class Base mention
precision measured on THRONE Base captions (other images, so independent of the CHAIR test labels) and its transfer to CHAIR;
(3) first-order counterfactual if classes above a precision threshold were exempt from belief/penalty; (4) generic-noun
substitution and V_obj coverage of lost objects; examples.   usage: rbd_recall_analysis2.py"""
import collections,glob,hashlib,json,re
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from vlm_grounding.evaluation import _load_external_module
L=Path('/home/project-b/LJW');R=L/'vlm-grounding-steering';O=R/'outputs/rbd2-20261102';X=O/'runs'
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_rec2');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
GT={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
refs=[json.loads(l) for l in open(O/'sets/chair_test500.jsonl')];IM={r['id']:r['metadata']['image_id'] for r in refs};ids=[r['id'] for r in refs]
rd=lambda p:{r['id']:r for r in map(json.loads,open(p))}
def men(cap):
    w,n,i,raw=sc.caption_to_words(cap);f={};wd={}
    for a,b,e in zip(n,i,w):
        f.setdefault(a,b);wd.setdefault(a,e)
    return f,wd,raw
Q={'qwen25':('Qwen2.5',X/'chair_test500_all/base-0-500.jsonl',X/'chair_test500_all/s16a2bl1p05-0-500.jsonl','qwen','models--Qwen--Qwen2.5-VL-7B-Instruct'),
   'gemma':('Gemma',X/'gchair_test500_all/base-0-500.jsonl',X/'gchair_test500_all/s24a1bl1p05-0-500.jsonl','gemma','models--google--gemma-4-E4B-it'),
   'qwen3':('Qwen3',X/'qchair_test500_all/base-0-500.jsonl',X/'qchair_test500_all/s20a1bl1p05-0-500.jsonl','qwen3','models--Qwen--Qwen3-VL-8B-Instruct')}
M={k:(rd(v[1]),rd(v[2])) for k,v in Q.items()};MEN={k:({i:men(B[i]['prediction']) for i in ids},{i:men(D[i]['prediction']) for i in ids}) for k,(B,D) in M.items()}
def boot(a,b,n=2000):
    g=np.random.default_rng(0);d=[100*(a[k].mean()-b[k].mean()) for k in (g.integers(0,len(a),len(a)) for _ in range(n))];return np.quantile(d,[.025,.975])
print('=== 1. CHAIR_s and class orange')
for k,v in Q.items():
    mb,md=MEN[k];hb=np.array([bool(set(mb[i][0])-GT[IM[i]]) for i in ids],float);hd=np.array([bool(set(md[i][0])-GT[IM[i]]) for i in ids],float)
    xb=np.array([bool(set(mb[i][0])-GT[IM[i]]-{'orange'}) for i in ids],float);xd=np.array([bool(set(md[i][0])-GT[IM[i]]-{'orange'}) for i in ids],float)
    co,ci_=boot(hd,hb),boot(xd,xb)
    ob=sum('orange' in set(mb[i][0])-GT[IM[i]] for i in ids);od=sum('orange' in set(md[i][0])-GT[IM[i]] for i in ids)
    print(f"{v[0]:8s} CHAIR_s Base {100*hb.mean():.1f} RBD {100*hd.mean():.1f} [{co[0]:+.1f},{co[1]:+.1f}] | without class orange Base {100*xb.mean():.1f} RBD {100*xd.mean():.1f} [{ci_[0]:+.1f},{ci_[1]:+.1f}] | captions with hallucinated orange {ob} -> {od}")
    ctx=collections.Counter()
    for i in ids:
        if 'orange' in set(mb[i][0])-GT[IM[i]]:
            cap=M[k][0][i]['prediction'].lower().split()
            for a,w in enumerate(cap[:-1]):
                if w.strip('.,*:;') =='orange':ctx[cap[a+1].strip('.,*:;')]+=1
    print('   word after hallucinated "orange" in Base:',ctx.most_common(14))
print('=== 2. class precision on THRONE Base captions (images not in the CHAIR test)')
cats={};img=collections.defaultdict(set)
for f in ('instances_val2014.json','instances_train2014.json'):
    a=json.load(open(R/'data/hallucination-suite/annotations'/f));cat={x['id']:x['name'] for x in a['categories']}
    for x in a['annotations']:img[x['image_id']].add(cat[x['category_id']])
    del a
test={IM[i] for i in ids}
def prior(p):
    n=collections.Counter();t=collections.Counter();use=0
    for r in map(json.loads,open(p)):
        m=r['metadata'];m=m if isinstance(m,dict) else eval(m);im=m['image_id']
        if im in test or im not in img:continue
        use+=1;f,_,_=men(r['prediction'])
        for o in f:
            n[o]+=1;t[o]+=o in img[im]
    return n,t,use
P={}
for k,p in (('qwen25',X/'throne_full1000/base-0-1000.jsonl'),('gemma',X/'gemma_throne1000/base-0-1000.jsonl')):
    n,t,u=prior(p);P[k]=(n,t);print(k,'THRONE images used',u)
pn=P['qwen25'][0]+P['gemma'][0];pt=P['qwen25'][1]+P['gemma'][1];P['qwen3']=(pn,pt)
def prec(n,t,o,a=2,b=.75):return (t[o]+a*b)/(n[o]+a)   # shrink rare classes towards 0.75
def rank(x):return np.argsort(np.argsort(x))
for k,v in Q.items():
    mb,_=MEN[k];n=collections.Counter();t=collections.Counter()
    for i in ids:
        for o in mb[i][0]:n[o]+=1;t[o]+=o in GT[IM[i]]
    cl=[o for o in n if n[o]>=5];a=[t[o]/n[o] for o in cl];b=[prec(*P[k],o) for o in cl]
    print(f"{v[0]:8s} classes n>=5 {len(cl)}: Spearman(CHAIR precision, THRONE prior) {np.corrcoef(rank(a),rank(b))[0,1]:.2f}; lowest prior:",[(o,round(prec(*P[k],o),2)) for o in sorted(n,key=lambda o:prec(*P[k],o))[:12]])
print('=== 3. first-order counterfactual: classes with prior >= tau keep their Base mentions (no decoding; ignores chain effects)')
for k,v in Q.items():
    B,D=M[k];mb,md=MEN[k];N=sum(len(GT[IM[i]]) for i in ids)
    print(v[0],f"Base recall {100*sum(len(set(mb[i][0])&GT[IM[i]]) for i in ids)/N:.1f}")
    for tau in (2,.95,.9,.85,.8,.7,.6,0):
        ex=lambda o:prec(*P[k],o)>=tau
        hit=hal=halx=0
        for i in ids:
            g=GT[IM[i]];s={o for o in md[i][0] if not ex(o)}|{o for o in mb[i][0] if ex(o)};hit+=len(s&g);hal+=bool(s-g);halx+=bool(s-g-{'orange'})
        nex=sum(ex(o) for o in set(sc.inverse_synonym_dict.values()))
        print(f"  tau {tau:<4} exempt classes {nex:2d}: recall {100*hit/N:.1f} CHAIR_s {100*hal/500:.1f} (without orange {100*halx/500:.1f})")
print('=== 4. V_obj, lost objects, generic nouns')
G='item items piece pieces section sections unit units structure structures material materials type types object objects gear container containers figure figures individual individuals thing things area areas element elements equipment device devices'.split()
for k,v in Q.items():
    B,D=M[k];mb,md=MEN[k];snap=glob.glob(str(L/'.cache/huggingface/hub'/v[4]/'snapshots/*'))[0];tok=AutoTokenizer.from_pretrained(snap,local_files_only=True)
    vo=json.load(open(R/f'outputs/rbd-20261030/collect/{v[3]}-vobj2.json'));vs={tok.convert_ids_to_tokens(i)[1:] for i in vo}
    print(f"{v[0]}: |V_obj| {len(vo)}; generic nouns in V_obj: {[w for w in G if w in vs]}; not in V_obj: {[w for w in G if w not in vs]}")
    cw=[w for w in sc.mscoco_objects if ' ' not in w];print(f"  CHAIR single-word synonyms in V_obj {sum(w in vs for w in cw)}/{len(cw)}; missing e.g. {[w for w in cw if w not in vs][:30]}")
    inv=ouT=lostg=keepg=0;nl=nk=0;cnt=lambda s:sum(s.lower().split().count(w) for w in G)
    for i in ids:
        g=GT[IM[i]];lost=set(mb[i][0])&g-set(md[i][0]);dg=cnt(D[i]['prediction'])-cnt(B[i]['prediction'])
        if lost:lostg+=dg;nl+=1
        else:keepg+=dg;nk+=1
        for o in lost:
            w=mb[i][1][o]
            if w in vs or ' ' in w and w.split()[0] in vs:inv+=1
            else:ouT+=1
    print(f"  lost true objects: Base word in V_obj {inv}, not in V_obj {ouT}; generic-noun count change per caption: images with a lost object {lostg/max(nl,1):+.2f} (n {nl}), others {keepg/max(nk,1):+.2f} (n {nk})")
    tb=sum(cnt(B[i]['prediction']) for i in ids);td=sum(cnt(D[i]['prediction']) for i in ids);print(f"  generic nouns total Base {tb} RBD {td}")
def sents(c):return [s.strip() for s in re.split(r'(?<=[.!?\n])\s+',c) if len(s.strip())>12]
print('=== examples Qwen3: Base sentence with a lost true object -> most similar RBD sentence')
B,D=M['qwen3'];mb,md=MEN['qwen3'];shown=0
for i in ids:
    lost=set(mb[i][0])&GT[IM[i]]-set(md[i][0])
    for o in lost:
        if shown>=9:break
        w=mb[i][1][o];a=[s for s in sents(B[i]['prediction']) if w in s.lower()]
        if not a:continue
        sa=set(a[0].lower().split());b=max(sents(D[i]['prediction']),key=lambda s:len(sa&set(s.lower().split())))
        print(f"[{o}/{w}] B: {a[0][:210]!r}\n    R: {b[:210]!r}");shown+=1
print('ANALYSIS2_DONE')
