"""Per-object breakdown of the AMBER-100 runs with the official matching (same loop as score_amber100.per_image):
which truth objects Base covered and a run lost, whether those words are in V_obj / the COCO-80 (CHAIR) list, and
where in the Base caption they were mentioned; hallucinated nouns removed/added."""
import collections,json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/rbd_pilot')
import score_amber100 as S
from transformers import AutoTokenizer

R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd-test-pilot-20261101/amber'
vobj=set(json.loads((R/'outputs/rbd-20261030/collect/qwen-vobj2.json').read_text()))
tok=AutoTokenizer.from_pretrained('/workspace/models/qwen2.5-vl-7b-instruct')
coco=set(json.loads(Path('/workspace/rbd_pilot/chair-test100-scorer-state.json').read_text())['state']['mscoco_objects'])
in_vobj=lambda w:tok(' '+w,add_special_tokens=False)['input_ids'][0] in vobj

def detail(id,resp,sim=0.8):
    """official generative loop; returns covered truth words and hallucinated nouns"""
    g=S.ground_truth[id-1];a=S.association;nouns=[n for n in S.O.extract_nouns(resp) if n in S.hallucination_words]
    sw=[];sl=[]
    for k,w in enumerate(g['truth']):sw+=a[w];sl+=[k]*len(a[w])
    sw+=g['truth'];n=len(g['truth']);sl+=[0]*n;hal=[]
    for noun in nouns:
        if noun in S.global_safe_words:continue
        j=next((j for j,x in enumerate(sw) if noun==x),None) if noun in sw else None
        if j is None:j=next((j for j,x in enumerate(sw) if S.O.check_synonyms_word(noun,x,sim)),None)
        if j is None:hal.append(noun);continue
        if j<len(sl)-n:sl[sl[j]+len(sl)-n]=1
        else:sl[j]=1
    return {g['truth'][k] for k,v in enumerate(sl[-n:]) if v==1},hal

def pos(text,w):
    ws=re.findall(r"[a-z]+",text.lower());names={w,*S.association.get(w,[])}
    for i,x in enumerate(ws):
        if x in names or x.rstrip('s') in names:return i/len(ws)
    return None

ids=[r['metadata']['amber_id'] for r in S.read('/workspace/amber100.jsonl')];D={};T={}
for n in ['base','rbd01','unif01']:
    rows={r['metadata']['amber_id']:r for r in S.read(O/f'qwen-{n}-0-100.jsonl')};T[n]=rows
    D[n]={i:detail(i,rows[i]['prediction']) for i in ids}
    print(f"{n:7s} truth objects covered {sum(len(D[n][i][0]) for i in ids)}  hallucinated nouns {sum(len(D[n][i][1]) for i in ids)}  words/caption {np.mean([len(rows[i]['prediction'].split()) for i in ids]):.0f}")
allpos=[p for i in ids for w in D['base'][i][0] if (p:=pos(T['base'][i]['prediction'],w)) is not None]
print(f"\nposition (fraction of Base caption) of all Base-covered objects: median {np.median(allpos):.2f}")
for n in ['rbd01','unif01']:
    lost=[(i,w) for i in ids for w in D['base'][i][0]-D[n][i][0]];gain=[(i,w) for i in ids for w in D[n][i][0]-D['base'][i][0]]
    lp=[p for i,w in lost if (p:=pos(T['base'][i]['prediction'],w)) is not None]
    print(f"\n{n} vs base: truth objects lost {len(lost)} gained {len(gain)}")
    print(f"  lost words in V_obj {sum(in_vobj(w) for _,w in lost)}/{len(lost)};  in COCO-80 list {sum(w in coco for _,w in lost)}/{len(lost)}")
    print(f"  lost: median position in Base caption {np.median(lp):.2f}; in last half {sum(p>=.5 for p in lp)}/{len(lp)}")
    print(f"  top lost {collections.Counter(w for _,w in lost).most_common(12)}")
    hr=collections.Counter(w for i in ids for w in set(D['base'][i][1])-set(D[n][i][1]));ha=collections.Counter(w for i in ids for w in set(D[n][i][1])-set(D['base'][i][1]))
    print(f"  hallucinated nouns removed {sum(hr.values())} added {sum(ha.values())}; added {ha.most_common(8)}")
