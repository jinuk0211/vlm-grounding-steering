"""Gemma CHAIR dev pick for gchair_chain.sh. Rule, fixed before any dev result: among candidates with dev recall >= vdd Base
recall - 2, lowest CHAIR_s, ties by CHAIR_i; if none qualifies, highest F1 (P = 100 - CHAIR_i, R = recall).
usage: gchair_pick.py SCORE_JSON   -> prints the picked tag"""
import json,sys
s=json.load(open(sys.argv[1]))['results'];b=s['base'];c={t:v for t,v in s.items() if t!='base'}
ok=[t for t,v in c.items() if v['recall']>=b['recall']-2]
f1=lambda v:2*(100-v['CHAIR_i'])*v['recall']/(100-v['CHAIR_i']+v['recall'])
print(min(ok,key=lambda t:(c[t]['CHAIR_s'],c[t]['CHAIR_i'])) if ok else max(c,key=lambda t:f1(c[t])))
