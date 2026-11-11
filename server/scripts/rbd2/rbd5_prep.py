"""RBD recall round (rbd5): V_obj files for the class gate. A token outside V_obj gets neither belief nor penalty, so a gated run is
vdd4_decode.py --vobj FILE with the unchanged decoder.
Class prior = Base mention precision on the 1000 THRONE Base captions, CHAIR test images excluded (qwen: its own; gemma: its own;
qwen3 has no THRONE Base: both pooled), shrunk towards 0.75 with weight 2. Classes with prior >= 0.8 are removed from V_obj:
every word CHAIR maps to the class and its plain plural; a token shared with a class that stays is kept.
tau = 0.8 was chosen after a counterfactual that used CHAIR test labels (rbd_recall_analysis2.py); disclose.
Writes outputs/rbd2-20261102/vdd5/{model}-gate80.json (from vobj2) and {model}-cfgate80.json (from the colour-fixed vobj2cf)."""
import collections,glob,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from vlm_grounding.evaluation import _load_external_module
L=Path('/home/project-b/LJW');R=L/'vlm-grounding-steering';O=R/'outputs/rbd2-20261102';X=O/'runs';V=O/'vdd5';V.mkdir(exist_ok=True)
c=json.loads((R/'data/hallucination-suite/chair-scorer-state.json').read_text());src=R/'upstream/chair/chair.py'
assert c['scorer_sha256']==hashlib.sha256(src.read_bytes()).hexdigest()
mod=_load_external_module(src,'chair_rbd5');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
test={json.loads(l)['metadata']['image_id'] for l in open(O/'sets/chair_test500.jsonl')}
img=collections.defaultdict(set)
for f in ('instances_val2014.json','instances_train2014.json'):
    a=json.load(open(R/'data/hallucination-suite/annotations'/f));cat={x['id']:x['name'] for x in a['categories']}
    for x in a['annotations']:img[x['image_id']].add(cat[x['category_id']])
    del a
def prior(p):
    n=collections.Counter();t=collections.Counter()
    for r in map(json.loads,open(p)):
        m=r['metadata'];m=m if isinstance(m,dict) else eval(m);im=m['image_id']
        if im in test or im not in img:continue
        w,nw,_,_=sc.caption_to_words(r['prediction'])
        for o in set(nw):n[o]+=1;t[o]+=o in img[im]
    return n,t
Pq=prior(X/'throne_full1000/base-0-1000.jsonl');Pg=prior(X/'gemma_throne1000/base-0-1000.jsonl')
PR={'qwen':Pq,'gemma':Pg,'qwen3':(Pq[0]+Pg[0],Pq[1]+Pg[1])}
SNAP={'qwen':'models--Qwen--Qwen2.5-VL-7B-Instruct','gemma':'models--google--gemma-4-E4B-it','qwen3':'models--Qwen--Qwen3-VL-8B-Instruct'}
TAU=.8
for m,(n,t) in PR.items():
    tok=AutoTokenizer.from_pretrained(glob.glob(str(L/'.cache/huggingface/hub'/SNAP[m]/'snapshots/*'))[0],local_files_only=True)
    vo=json.load(open(R/f'outputs/rbd-20261030/collect/{m}-vobj2.json'));cf=json.load(open(R/f'outputs/rbd-20261030/collect/{m}-vobj2cf.json'))
    byword={tok.convert_ids_to_tokens(i)[1:]:i for i in vo}
    prec=lambda o:(t[o]+2*.75)/(n[o]+2)
    tokclass=collections.defaultdict(set)
    for word,node in sc.inverse_synonym_dict.items():
        for part in word.split():
            for f in (part,part+'s',part+'es'):
                if f in byword:tokclass[byword[f]].add(node)
    ex={i for i,cl in tokclass.items() if all(prec(o)>=TAU for o in cl)}
    kept={o for cl in tokclass.values() for o in cl if prec(o)<TAU}
    exc=sorted({o for i in ex for o in tokclass[i]})
    (V/f'{m}-gate80.json').write_text(json.dumps(sorted(set(vo)-ex)));(V/f'{m}-cfgate80.json').write_text(json.dumps(sorted(set(cf)-ex)))
    print(f"{m}: V_obj {len(vo)} -> gate80 {len(set(vo)-ex)}, cf {len(cf)} -> cfgate80 {len(set(cf)-ex)}; {len(ex)} tokens removed, {len(exc)} classes exempt\n  exempt: {exc}\n  still suppressed: {sorted(kept)}\n  sample removed: {[tok.convert_ids_to_tokens(i) for i in sorted(ex)[:24]]}")
print('PREP_DONE')
