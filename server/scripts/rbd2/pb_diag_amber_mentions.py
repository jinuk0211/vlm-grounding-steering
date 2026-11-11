"""Stage A (AMBER env, CPU): every noun mention in the Base captions, as AMBER's official matching sees it:
truth object (kept or lost by RUN) or hallucinated, plus a scene/stuff flag.
usage: pb_diag_amber_mentions.py SET_JSONL BASE_ROWS RUN_ROWS OUT_JSON"""
import json,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import score_amber100 as S

STUFF=set(('sky ground sea grass forest cloud water road sand snow mountain beach desert field wall floor lake river ocean hill '
           'sun street sidewalk dirt rock wave tree stone soil land shore cliff valley meadow').split())

def mentions(i,resp):
    g=S.ground_truth[i-1];out=[]
    for n in dict.fromkeys(n for n in S.O.extract_nouns(resp) if n in S.hallucination_words):
        if n in S.global_safe_words:continue
        tr=next((t for t in g['truth'] if n==t or n in S.association[t] or S.O.check_synonyms_word(n,t,0.8)),None)
        out.append({'noun':n,'object':tr or n,'kind':'truth' if tr else 'hallucinated',
                    'words':sorted({n,*(S.association[tr] if tr else [])})})
    return out

refs=[json.loads(l) for l in open(sys.argv[1])]
base={r['id']:r for r in map(json.loads,open(sys.argv[2]))};run={r['id']:r for r in map(json.loads,open(sys.argv[3]))};res=[]
for r in refs:
    i=r['metadata']['amber_id'];mb=mentions(i,base[r['id']]['prediction'])
    kept={m['object'] for m in mentions(i,run[r['id']]['prediction']) if m['kind']=='truth'}
    for m in mb:
        if m['kind']=='truth':m['kind']='truth_kept' if m['object'] in kept else 'truth_lost'
        m['stuff']=m['object'] in STUFF
    res.append({'id':r['id'],'image':r['image'],'prompt':r['prompt'],'caption':base[r['id']]['prediction'],'mentions':mb})
json.dump(res,open(sys.argv[4],'w'))
c=[m for x in res for m in x['mentions']]
print(sys.argv[4],{k:sum(m['kind']==k for m in c) for k in ('truth_kept','truth_lost','hallucinated')},
      'stuff among lost:',sum(m['stuff'] for m in c if m['kind']=='truth_lost'))
