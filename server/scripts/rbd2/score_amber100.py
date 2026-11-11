"""AMBER generative metrics (CHAIR, Cover, Hal, Cog) on the first 100 AMBER images.

Per-image counts reuse the official inference.py (extract_nouns, check_synonyms_word, data files); the loop body
below is its generative branch unchanged, only accumulated per image so a paired bootstrap over images is possible.
Aggregates are checked against the official script run on the same responses.
usage: score_amber100.py OUT_JSON name=rows.jsonl [...]   (first name = Base)"""
import importlib.util,json,os,re,subprocess,sys,tempfile
from pathlib import Path
import numpy as np

A=Path(os.environ.get('AMBER_REPO','/tmp/AMBER'))  # project-b: data/amber/repo
spec=importlib.util.spec_from_file_location('amber_official',A/'inference.py');O=importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

association=json.load(open(A/'data/relation.json',encoding='utf-8'))
hallucination_words=[w for k,v in association.items() for w in [k,*v]]
global_safe_words=[l.split('\n')[0] for l in open(A/'data/safe_words.txt',encoding='utf-8')]
ground_truth=json.load(open(A/'data/annotations.json',encoding='utf-8'))

def per_image(id,response,sim=0.8):
    g=ground_truth[id-1];assert g['type']=='generative'
    nouns=O.extract_nouns(response);after_process_nouns=[n for n in nouns if n in hallucination_words]
    safe_words=[];safe_list=[]
    for idx,word in enumerate(g['truth']):safe_words+=association[word];safe_list+=[idx]*len(association[word])
    ha_words=[];ha_list=[]
    for idx,word in enumerate(g['hallu']):ha_words+=association[word];ha_list+=[idx]*len(association[word])
    safe_words+=g['truth'];safe_len=len(g['truth']);safe_list+=[0]*safe_len;safe_flag_list=[0]*len(after_process_nouns)
    ha_words+=g['hallu'];ha_len=len(g['hallu']);ha_list+=[0]*ha_len
    for idx,noun in enumerate(after_process_nouns):
        if noun in global_safe_words:continue
        if noun in safe_words:
            for j in range(len(safe_words)):
                if noun==safe_words[j]:
                    if j<(len(safe_list)-safe_len):safe_list[safe_list[j]+len(safe_list)-safe_len]=1
                    else:safe_list[j]=1
                    break
            continue
        if noun in ha_words:
            for j in range(len(ha_words)):
                if noun==ha_words[j]:
                    if j<(len(ha_list)-ha_len):ha_list[ha_list[j]+len(ha_list)-ha_len]=1
                    else:ha_list[j]=1
                    break
        for j,check_word in enumerate(ha_words):
            if O.check_synonyms_word(noun,check_word,sim):
                if j<(len(ha_list)-ha_len):ha_list[ha_list[j]+len(ha_list)-ha_len]=1
                else:ha_list[j]=1
                break
        flag=False
        for j,check_word in enumerate(safe_words):
            if O.check_synonyms_word(noun,check_word,sim):
                flag=True
                if j<(len(safe_list)-safe_len):safe_list[safe_list[j]+len(safe_list)-safe_len]=1
                else:safe_list[j]=1
                break
        if flag:continue
        safe_flag_list[idx]=1
    return [sum(safe_flag_list),len(safe_flag_list),sum(safe_list[-safe_len:]),len(safe_list[-safe_len:]),
            sum(ha_list[-ha_len:]),len(ha_list[-ha_len:]),int(sum(safe_flag_list)==0)]

def metrics(X):
    s=X.sum(0);return {'CHAIR':100*s[0]/s[1],'Cover':100*s[2]/s[3],'Hal':100-100*s[6]/len(X),'Cog':100*s[4]/s[5]}

def official(resp):
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:json.dump(resp,f)
    out=subprocess.run([sys.executable,'inference.py','--inference_data',f.name,'--evaluation_type','g'],cwd=A,capture_output=True,text=True,check=True).stdout
    return {k:float(v) for k,v in re.findall(r'(CHAIR|Cover|Hal|Cog):\s+([\d.]+)',out)}

def main(out,specs):
    ids=[r['metadata']['amber_id'] for r in read('/workspace/amber100.jsonl')];res={};per={}
    for spec in specs:
        name,path=spec.split('=',1);rows={r['metadata']['amber_id']:r for r in read(path)}
        if any(i not in rows for i in ids):print(name,'incomplete',len(rows));continue
        resp=[{'id':i,'response':rows[i]['prediction']} for i in ids]
        X=np.array([per_image(r['id'],r['response']) for r in resp],float);m=metrics(X);o=official(resp)
        assert all(abs(round(m[k],1)-o[k])<1e-6 for k in o),(name,m,o)
        per[name]=X;res[name]={**m,'official':o,'mean_tokens':float(np.mean([rows[i]['generated_tokens'] for i in ids])),
                               'at_512':int(sum(rows[i]['generated_tokens']>=512 for i in ids))}
    base=next(iter(per));rng=np.random.default_rng(0);B=[rng.integers(0,len(ids),len(ids)) for _ in range(2000)]
    for name in per:
        if name==base:continue
        d=np.array([[metrics(per[name][k])[q]-metrics(per[base][k])[q] for q in ('CHAIR','Cover','Hal','Cog')] for k in B])
        for j,q in enumerate(('CHAIR','Cover','Hal','Cog')):res[name][f'd{q}_ci95']=[float(np.quantile(d[:,j],.025)),float(np.quantile(d[:,j],.975))]
    Path(out).write_text(json.dumps({'n_images':len(ids),'results':res},indent=1))
    for k,v in res.items():
        ci=''.join(f" d{q} [{v[f'd{q}_ci95'][0]:+.1f},{v[f'd{q}_ci95'][1]:+.1f}]" for q in ('CHAIR','Cover','Hal','Cog')) if 'dHal_ci95' in v else ''
        print(f"{k:22s} CHAIR {v['CHAIR']:5.1f} Cover {v['Cover']:5.1f} Hal {v['Hal']:5.1f} Cog {v['Cog']:5.1f} tok {v['mean_tokens']:4.0f} @512 {v['at_512']}{ci}")

if __name__=='__main__':main(sys.argv[1],sys.argv[2:])
