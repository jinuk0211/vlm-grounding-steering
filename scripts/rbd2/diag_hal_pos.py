"""Where do the hallucinations sit? For each hallucinated mention in Base (and given runs): word position in the
caption (fraction and approx. token index), the sentence around it. AMBER: official matching (diag_amber_set.detail);
CHAIR: unmodified scorer from the cached state.  usage: diag_hal_pos.py amber|chair TAG [TAG ...]"""
import json,re,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,'/workspace/rbd_pilot')
R=Path('/home/project-b/LJW/vlm-grounding-steering');KIND=sys.argv[1];TAGS=['base',*sys.argv[2:]]
SET={'amber':'amber_dev60','chair':'chair_dev60'}[KIND];O=Path('/workspace/vdd')/SET
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
refs=read(f'/workspace/sets/{SET}.jsonl')

def locate(text,word):
    ws=re.findall(r"[A-Za-z]+",text);lw=[w.lower() for w in ws];word=word.lower().split()[-1]
    for i,w in enumerate(lw):
        if w==word or w.rstrip('s')==word.rstrip('s'):
            sents=re.split(r'(?<=[.!?])\s+',text);c=0
            for s in sents:
                n=len(re.findall(r"[A-Za-z]+",s))
                if c+n>i:return i,len(ws),s.strip()
                c+=n
            return i,len(ws),''
    return None

if KIND=='amber':
    import score_amber100 as S
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

def hallucinated(tag):
    rows={r['id']:r for r in read(O/f'{tag}-0-60.jsonl')};out=[]
    if KIND=='amber':
        for r in refs:out.append((r['id'],rows[r['id']]['prediction'],detail(r['metadata']['amber_id'],rows[r['id']]['prediction'])[1]))
    else:
        from vlm_grounding.evaluation import _load_external_module
        c=json.loads((R/'data/chair-dev-train2014-100/chair-scorer-state.json').read_text())
        mod=_load_external_module(R/'upstream/chair/chair.py','chair_pos');sc=mod.CHAIR.__new__(mod.CHAIR);sc.__dict__.update(c['state'])
        sc.imid_to_objects={int(k):set(v) for k,v in c['state']['imid_to_objects'].items()}
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:
            json.dump([{'image_id':r['metadata']['image_id'],'caption':rows[r['id']]['prediction']} for r in refs],f)
        s={x['image_id']:x for x in sc.compute_chair(f.name,'image_id','caption')['sentences']}
        for r in refs:out.append((r['id'],rows[r['id']]['prediction'],[h[0] for h in s[r['metadata']['image_id']]['mscoco_hallucinated_words']]))
    return out

for tag in TAGS:
    res=hallucinated(tag);pos=[];ctx=[]
    for i,text,hal in res:
        for w in dict.fromkeys(hal):
            L=locate(text,w)
            if L:pos.append((L[0]/L[1],L[0]*1.3));ctx.append((i,w,round(L[0]/L[1],2),L[2][:160]))
    p=np.array(pos) if pos else np.zeros((0,2))
    print(f"\n== {KIND} {tag}: hallucinated mentions {len(p)} in {sum(bool(h) for _,_,h in res)} captions; "
          f"position fraction median {np.median(p[:,0]):.2f}; approx token idx median {np.median(p[:,1]):.0f}; "
          f"in first 30 tokens {int((p[:,1]<30).sum())}, 30-100 {int(((p[:,1]>=30)&(p[:,1]<100)).sum())}, >=100 {int((p[:,1]>=100).sum())}")
    if tag=='base':
        for c in ctx:print('  ',c)
