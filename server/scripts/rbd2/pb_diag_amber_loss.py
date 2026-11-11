"""Which AMBER truth objects does a run lose against Base, and where were they in the Base caption?
Per image, 'covered' = truth objects the official generative matching marks as mentioned (score_amber100.per_image
logic, kept here with the object names). usage: pb_diag_amber_loss.py RUNDIR TAG [TAG ...]   (Base = 'base')"""
import collections,json,os,re,sys
from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import score_amber100 as S
def covered(i,resp,sim=0.8):
    g=S.ground_truth[i-1];nouns=[n for n in S.O.extract_nouns(resp) if n in S.hallucination_words];hit=set();hal=set()
    for n in nouns:
        if n in S.global_safe_words:continue
        for t in g['truth']:
            if n==t or n in S.association[t] or S.O.check_synonyms_word(n,t,sim):hit.add(t);break
        else:
            for h in g['hallu']:
                if n==h or n in S.association[h] or S.O.check_synonyms_word(n,h,sim):hal.add(h);break
    return hit,hal
def first_pos(resp,word):
    w=resp.lower();m=re.search(r'\b'+re.escape(word.lower()),w)
    return None if not m else len(w[:m.start()].split())/max(1,len(w.split()))
run=Path(sys.argv[1]);rows={t:{r['metadata']['amber_id']:r for r in map(json.loads,open(run/f'{t}-0-60.jsonl'))} for t in ['base']+sys.argv[2:]}
ids=sorted(rows['base'])
for tag in sys.argv[2:]:
    lost=collections.Counter();gained=collections.Counter();pos=[];fixed=collections.Counter();newhal=collections.Counter();shorter=0
    for i in ids:
        b,m=rows['base'][i]['prediction'],rows[tag][i]['prediction']
        cb,hb=covered(i,b);cm,hm=covered(i,m)
        for o in cb-cm:
            lost[o]+=1;p=first_pos(b,o)
            if p is not None:pos.append(p)
        for o in cm-cb:gained[o]+=1
        for o in hb-hm:fixed[o]+=1
        for o in hm-hb:newhal[o]+=1
    pos.sort()
    print(f'== {tag}: lost {sum(lost.values())} truth objects, gained {sum(gained.values())}; hallucinations removed {sum(fixed.values())}, added {sum(newhal.values())}')
    print('   lost   :',lost.most_common(25))
    print('   gained :',gained.most_common(12))
    print('   removed hallucinations:',fixed.most_common(12));print('   new hallucinations    :',newhal.most_common(8))
    if pos:print(f'   lost objects: first mention in the Base caption at relative position median {pos[len(pos)//2]:.2f}, quartiles {pos[len(pos)//4]:.2f} / {pos[3*len(pos)//4]:.2f}')
