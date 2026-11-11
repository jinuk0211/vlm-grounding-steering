"""AMBER 'why' diagnostic (request of session 논문 업데이트 amber, 11-05): CPU only, existing rows, no decoding.
Per-noun version of score_amber100.per_image (same matching code, same order of checks), so a noun is 'flagged' exactly when
the official generative scorer counts it as a hallucination (the Hal flag: safe_flag_list entry 1), not the Cog list.
The per-image vector is checked against score_amber100.per_image on a sample and the aggregate Hal against score.json.
Mentions = the nouns of extract_nouns that are in the AMBER vocabulary (after_process_nouns, the CHAIR denominator of the
scorer; global-safe words are mentions that are never flagged). Mention order = rank in that list; relative position =
token index / number of tokens (nltk word_tokenize, as the scorer)."""
import collections,json,os,sys
from pathlib import Path
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
sys.path.insert(0,'/home/project-b/LJW/vlm-grounding-steering/scripts/rbd2')
import score_amber100 as S
OF=S.O
R=Path('/home/project-b/LJW/vlm-grounding-steering/outputs/rbd2-20261102')
STUFF=set(('sky ground sea grass forest cloud water road sand snow mountain beach desert field wall floor lake river ocean hill '
           'sun street sidewalk dirt rock wave tree stone soil land shore cliff valley meadow').split())
HEDGE=set('possibly perhaps maybe likely probably potentially presumably might'.split());HEDGE2=HEDGE|{'appears','seems','suggests'}
lem=WordNetLemmatizer()
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def by_id(p):return {r['metadata']['amber_id']:r for r in read(p)}

def analyze(i,text,sim=0.8):
    tokens=nltk.word_tokenize(text);tagged=nltk.pos_tag(tokens)
    nouns=[(lem.lemmatize(w),k) for k,(w,p) in enumerate(tagged) if p.startswith('NN')]
    ap=[(n,k) for n,k in nouns if n in S.hallucination_words]
    g=S.ground_truth[i-1];safe_words=[];safe_list=[]
    for idx,word in enumerate(g['truth']):safe_words+=S.association[word];safe_list+=[idx]*len(S.association[word])
    safe_words+=g['truth'];sl=len(g['truth']);safe_list+=[0]*sl;flags=[0]*len(ap)
    for idx,(noun,_) in enumerate(ap):
        if noun in S.global_safe_words:continue
        if noun in safe_words:
            for j in range(len(safe_words)):
                if noun==safe_words[j]:
                    if j<(len(safe_list)-sl):safe_list[safe_list[j]+len(safe_list)-sl]=1
                    else:safe_list[j]=1
                    break
            continue
        hit=False
        for j,cw in enumerate(safe_words):
            if OF.check_synonyms_word(noun,cw,sim):
                hit=True
                if j<(len(safe_list)-sl):safe_list[safe_list[j]+len(safe_list)-sl]=1
                else:safe_list[j]=1
                break
        if hit:continue
        flags[idx]=1
    cov={g['truth'][k] for k in range(sl) if safe_list[len(safe_list)-sl+k]==1}
    return {'tokens':tokens,'ap':ap,'flags':flags,'cov':cov,'hal':int(sum(flags)>0),'sum':sum(flags),'len':len(flags),'pred':text}

def run(path,ids=None):
    rows=by_id(path);ids=ids or sorted(rows);return {i:analyze(i,rows[i]['prediction']) for i in ids}
def hal(an):return 100*np.mean([a['hal'] for a in an.values()])
def check(an,path,n=40):
    rows=by_id(path)
    for i in sorted(an)[:n]:
        v=S.per_image(i,rows[i]['prediction']);a=an[i]
        assert v[0]==a['sum'] and v[1]==a['len'] and v[6]==int(a['sum']==0),(i,v,a['sum'],a['len'])

def order_tables(an,title):
    print(f'\n== {title}: n captions {len(an)}, mentions {sum(a["len"] for a in an.values())}, flagged {sum(a["sum"] for a in an.values())}')
    oc=collections.Counter();of=collections.Counter();pc=collections.Counter();pf=collections.Counter();st=collections.Counter()
    for a in an.values():
        nt=max(len(a['tokens']),1)
        for r,((n,k),f) in enumerate(zip(a['ap'],a['flags'])):
            b=min(r,5);oc[b]+=1;of[b]+=f;q=min(int(5*k/nt),4);pc[q]+=1;pf[q]+=f
            st['m_stuff' if n in STUFF else 'm_thing']+=1;st['f_stuff' if n in STUFF else 'f_thing']+=f
    tf=sum(of.values())
    print('  by mention order        mentions  flagged  flag rate  share of all flagged')
    for b in range(6):print(f'   {"%d%s"%(b+1,"+" if b==5 else ""):4s}                  {oc[b]:7d} {of[b]:8d}  {100*of[b]/max(oc[b],1):7.1f}%  {100*of[b]/max(tf,1):8.1f}%')
    print('  by relative position (token index / tokens), fifths of the caption')
    for q in range(5):print(f'   {20*q:2d}-{20*q+20:3d}%              {pc[q]:7d} {pf[q]:8d}  {100*pf[q]/max(pc[q],1):7.1f}%  {100*pf[q]/max(tf,1):8.1f}%')
    print(f'  STUFF nouns: mentions {st["m_stuff"]} ({100*st["m_stuff"]/max(sum(oc.values()),1):.1f}% of mentions), flagged {st["f_stuff"]} = {100*st["f_stuff"]/max(tf,1):.1f}% of flagged, flag rate {100*st["f_stuff"]/max(st["m_stuff"],1):.1f}% (things: {100*st["f_thing"]/max(st["m_thing"],1):.1f}%)')
    fl=collections.Counter(n for a in an.values() for (n,_),f in zip(a['ap'],a['flags']) if f)
    print('  most-flagged nouns:',fl.most_common(15))

def compare(ab,ar,name):
    c=collections.Counter()
    for i in ab:
        b,r=ab[i],ar[i]
        cb=collections.Counter(n for (n,_),f in zip(b['ap'],b['flags']) if f);cr=collections.Counter(n for (n,_),f in zip(r['ap'],r['flags']) if f)
        for n,v in (cb-cr).items():c['rem_stuff' if n in STUFF else 'rem_thing']+=v
        for n,v in (cr-cb).items():c['add_stuff' if n in STUFF else 'add_thing']+=v
        for t in b['cov']-r['cov']:c['lost_stuff' if t in STUFF else 'lost_thing']+=1
        for t in r['cov']-b['cov']:c['gain_stuff' if t in STUFF else 'gain_thing']+=1
        c['same']+=int(b['pred']==r['pred']);c['hal10']+=int(b['hal'] and not r['hal']);c['hal01']+=int(r['hal'] and not b['hal'])
        c['nb']+=b['sum'];c['nr']+=r['sum']
    print(f'  {name:8s} flagged nouns removed {c["rem_stuff"]+c["rem_thing"]:3d} (stuff {c["rem_stuff"]:3d}, things {c["rem_thing"]:3d}) added {c["add_stuff"]+c["add_thing"]:3d} (stuff {c["add_stuff"]:3d}, things {c["add_thing"]:3d})'
          f' | truth lost {c["lost_stuff"]+c["lost_thing"]:3d} (stuff {c["lost_stuff"]:3d}, things {c["lost_thing"]:3d}) gained {c["gain_stuff"]+c["gain_thing"]:3d} (stuff {c["gain_stuff"]:3d}, things {c["gain_thing"]:3d})'
          f' | identical captions {c["same"]}/{len(ab)} | Hal 1->0 {c["hal10"]}, 0->1 {c["hal01"]} | flagged total {c["nb"]} -> {c["nr"]}')

def hedge_counts(an,name):
    tot=collections.Counter();caps=0
    for a in an.values():
        t=[w.lower() for w in a['tokens']];h=collections.Counter(w for w in t if w in HEDGE2);caps+=int(bool(h));tot.update(h)
    print(f'  {name:9s} hedge tokens {sum(tot.values())} in {caps}/{len(an)} captions; {dict(tot.most_common())}')

def hedged_flagged(ab,ag,delims,label):
    n=hed=still_any=still_flag=0;o_n=o_any=o_flag=0
    for i,b in ab.items():
        t=[w.lower() for w in b['tokens']];cl=[];c=0
        for w in b['tokens']:
            cl.append(c)
            if w in delims:c+=1
        hedged_clauses={cl[k] for k,w in enumerate(t) if w in HEDGE2}
        g=ag[i];gnouns={n_ for n_,_ in g['ap']};gflag=collections.Counter(n_ for (n_,_),f in zip(g['ap'],g['flags']) if f)
        for (noun,k),f in zip(b['ap'],b['flags']):
            if not f:continue
            if cl[k] in hedged_clauses:
                n+=1;still_any+=int(noun in gnouns);still_flag+=int(gflag[noun]>0)
            else:o_n+=1;o_any+=int(noun in gnouns);o_flag+=int(gflag[noun]>0)
    print(f'  {label}: flagged nouns in vdd Base {n+o_n}; inside a hedged clause {n}: same noun still in the g5 caption {still_any} ({100*still_any/max(n,1):.0f}%), still flagged in g5 {still_flag} ({100*still_flag/max(n,1):.0f}%)'
          f' | outside hedged clauses {o_n}: still in g5 {o_any} ({100*o_any/max(o_n,1):.0f}%), still flagged {o_flag} ({100*o_flag/max(o_n,1):.0f}%)')

if __name__=='__main__':
    vb1004=R/'runs/amber_full1004_ours_all/base-0-1004.jsonl'
    print('##### 1. vdd Base on AMBER 1004: where do the flagged nouns sit');a1004=run(vb1004);check(a1004,vb1004)
    print(f'   Hal {hal(a1004):.1f} (score.json vdd Base on 1004 for comparison below), per-image check against score_amber100.per_image passed on 40 captions')
    try:print('   score.json:',json.load(open(R/'runs/amber_full1004_ours_all/score.json'))['results']['base']['Hal'])
    except Exception as e:print('   (score.json not read:',e,')')
    order_tables(a1004,'vdd Base, AMBER 1004')
    print('\n##### 2. dev 200 (AMBER ids 101-300): each candidate vs vdd Base')
    D=R/'runs/amber_v3_dev_all';tags=['base','g1','g4','g5','mainMed'];dev={t:run(D/f'{t}-0-200.jsonl') for t in tags}
    print('   Hal by tag:',{t:round(hal(dev[t]),1) for t in tags},'(dev table: base 33.0, g1 25.5, g4 29.5, g5 27.5, mainMed 28.0)')
    for t in tags[1:]:compare(dev['base'],dev[t],t)
    print('\n##### 3. hedging, dev 200');hedge_counts(dev['base'],'vdd Base');hedge_counts(dev['g5'],'g5')
    hedged_flagged(dev['base'],dev['g5'],{'.','!','?',';',':'},'hedged clause = between . ! ? ; :')
    hedged_flagged(dev['base'],dev['g5'],{'.','!','?',';',':',','},'hedged clause = between . ! ? ; : ,')
    print('\n##### 4. AMBER test 804: bench Base vs vdd Base')
    bb=R/'runs/amber_v3_test_vs_bench/base-0-804.jsonl';vv=R/'runs/amber_v3_test_vs_vdd/base-0-804.jsonl'
    ids804=sorted(by_id(vv));ab=run(bb,ids804);check(ab,bb);av=run(vv,ids804)
    print(f'   Hal bench Base {hal(ab):.1f}, vdd Base {hal(av):.1f} (score tables: 27.6 and 29.4)')
    compare(ab,av,'vdd/bench')
    flips=[(i,ab[i]['sum'],av[i]['sum']) for i in ids804 if ab[i]['hal']!=av[i]['hal']]
    print(f'   flips: {len(flips)} images change Hal flag; net {sum(1 for _,x,y in flips if x==0 and y>0)-sum(1 for _,x,y in flips if x>0 and y==0):+d} (positive = vdd Base worse)')
