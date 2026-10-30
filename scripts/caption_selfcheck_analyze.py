"""Analyze E0-caption rows: separation of hallucinated vs correct mentions by the
model's own Yes/No margin, and the CHAIR effect of dropping mentions the model
itself denies (offline simulation, not a generation result)."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from read_gap_analyze_v3 import auroc

def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def simulate(rows,tau):
    caps={}
    for r in rows:caps.setdefault(r['image_id'],[]).append(r)
    hs=0;hall=0;men=0;cov=0;gt=0
    for rs in caps.values():
        keep=[r for r in rs if r['kind']!='missed' and r['margin']>=tau]
        hall_c=sum(r['mention_count'] for r in keep if r['kind']=='hallucinated')
        hall+=hall_c;men+=sum(r['mention_count'] for r in keep);hs+=hall_c>0
        cov+=sum(1 for r in keep if r['kind']=='correct');gt+=sum(1 for r in rs if r['kind']!='hallucinated')
    return {'CHAIR_s':hs/len(caps),'CHAIR_i':hall/max(men,1),'recall':cov/gt}

def main(path):
    rows=read(path);m=lambda k:[r['margin'] for r in rows if r['kind']==k]
    h,c,mi=m('hallucinated'),m('correct'),m('missed')
    out={'n':{'hallucinated':len(h),'correct':len(c),'missed':len(mi)},
         'auroc_correct_vs_hallucinated':auroc(c+h,[1]*len(c)+[0]*len(h)),
         'denied_at_0':{'hallucinated':sum(x<0 for x in h)/len(h),'correct':sum(x<0 for x in c)/len(c),'missed_affirmed':sum(x>0 for x in mi)/len(mi)},
         'simulated_drop_denied':{str(t):simulate(rows,t) for t in (float('-inf'),-2,-1,0,1,2)}}
    # hallucination rate by order of first mention among object nodes in the caption
    by={}
    for r in rows:
        if r['kind']=='missed':continue
        o=min(r['mention_order'],5);b=by.setdefault(o,[0,0]);b[0]+=r['kind']=='hallucinated';b[1]+=1
    out['hallucination_rate_by_mention_order']={str(k):[v[0],v[1],round(v[0]/v[1],3)] for k,v in sorted(by.items())}
    print(json.dumps(out,indent=1));Path(str(path).replace('-rows.jsonl','-analysis.json')).write_text(json.dumps(out,indent=1))

if __name__=='__main__':
    for p in sys.argv[1:]:main(Path(p))
