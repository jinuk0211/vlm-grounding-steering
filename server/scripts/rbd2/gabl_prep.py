"""Gemma 4 E4B CHAIR ablation of the paper's RBD row, same design as abl_prep.py (Qwen2.5-VL). User 11-05: after HAT + ALOHa and the Qwen3 baselines.
Full RBD = s24a1bl1p05: steering alpha 1 at layer 24 + belief lam 1 + penalty 0.5, decay schedules, measured r(t), raw evidence, eager, old V_obj
(runs/gchair_test500_all; CHAIR_s 26.2 / CHAIR_i 5.5 / recall 63.6). Its setting was picked on test rows 0-60 (gchair3.sh), the variants were not.
Variants (one change each, everything else as the full row):
  abl_nosteer   alpha 0              (belief + penalty)         rows 0-59 reused from runs/gchair_t60/bl1p05
  abl_nobelief  lam 0                (steering + penalty)
  abl_nopen     pen 0                (steering + belief)
  abl_steer     steering only        (alpha 1)
  abl_belief    belief only          (lam 1)
  abl_pen       penalty only         (pen 0.5)
  abl_const     schedules constant   (--bsched const --ssched const: w(t) = s(t) = 1 instead of 1 - r(t))
Rows are reused only where the stored setting is identical. vdd Base and the full row come from runs/gchair_test500_all, the table Base from
runs/gchair_test500_tbl.
  check      run when the chain is scheduled: every reused file is complete and has the setting it should have
  queue      queues/gabl_chair.txt (units of 50; reused prefixes skipped), queues/gabl_repro.txt (full setting, rows 0-5)
  repro      rows 0-5 decoded now (vdd4_decode.py) vs the stored full row (vdd_decode.py); exit 1 if none is identical
  join       runs/gabl_chair500_all/<tag>-0-500.jsonl (base = vdd Base, rbd_full, variants) and
             runs/gabl_chair500_vsfull/ (base = the full RBD row, so the bootstrap gives each variant's change vs full; vddbase, tblbase)
  table DIR  the paper's ablation table in LaTeX from DIR/score.json (a *_vsfull directory)"""
import json,os,sys
from pathlib import Path
O=Path(os.environ.get('GABL_L','/home/project-b/LJW'))/'vlm-grounding-steering/outputs/rbd2-20261102'
FULL='s24a1bl1p05';S='--alpha 1 --layer 24'
V={'abl_nosteer':'--lam 1 --pen 0.5','abl_nobelief':f'{S} --pen 0.5','abl_nopen':f'{S} --lam 1',
   'abl_steer':S,'abl_belief':'--lam 1','abl_pen':'--pen 0.5',
   'abl_const':f'{S} --lam 1 --pen 0.5 --bsched const --ssched const'}
REUSE={'abl_nosteer':(O/'runs/gchair_t60/bl1p05-0-60.jsonl',60)}
FF=O/f'runs/gchair_test500_all/{FULL}-0-500.jsonl';BF=O/'runs/gchair_test500_all/base-0-500.jsonl';TF=O/'runs/gchair_test500_tbl/base-0-500.jsonl'
def read(p):return [json.loads(l) for l in open(p) if l.strip()]
def st(r):s={'bpos':False,'norm':'raw',**(r.get('setting') or {})};s.pop('vobj',None);return json.dumps(s,sort_keys=True)
def want(**k):return json.dumps({'lam':0.0,'clip':5.0,'pen':0.0,'bsched':'decay','alpha':0.0,'layer':16,'ssched':'decay','r':'measured','attn':'eager','bpos':False,'norm':'raw',**k},sort_keys=True)
ref=[r['id'] for r in read(O/'sets/chair_test500.jsonl')];assert len(ref)==500
def spans(t):
    a=REUSE[t][1] if t in REUSE else 0;b=[a]+list(range((a//50+1)*50,500,50)) if a%50 else list(range(a,500,50));return [(s,min(s+50,500) if s%50==0 else (s//50+1)*50) for s in b]
cmd=sys.argv[1]
if cmd=='check':
    f=read(FF);assert sorted(x['id'] for x in f)==sorted(ref),'full row incomplete';assert {st(x) for x in f}=={want(alpha=1.0,layer=24,lam=1.0,pen=0.5)},f'full row setting: {st(f[0])}'
    b=read(BF);assert sorted(x['id'] for x in b)==sorted(ref),'vdd Base incomplete';s=json.loads(st(b[0]));assert s['alpha']==s['lam']==s['pen']==0 and s['attn']=='eager',f'vdd Base setting: {s}'
    t=read(TF);assert sorted(x['id'] for x in t)==sorted(ref),'table Base incomplete'
    for k,(p,n) in REUSE.items():
        r=read(p);assert [x['id'] for x in r]==ref[:n],f'{p} ids';assert {st(x) for x in r}=={want(lam=1.0,pen=0.5)},f'{p} setting: {st(r[0])}'
    print('check ok: full, vdd Base, table Base complete; reused',{k:n for k,(p,n) in REUSE.items()})
elif cmd=='queue':
    q=['# Gemma CHAIR ablation of RBD s24a1bl1p05 (gabl_prep.py)']
    for t,a in V.items():
        for s,e in spans(t):q.append(f'chair_test500 gabl_chair500 {t} {s} {e} --attn eager {a}')
    (O/'queues/gabl_chair.txt').write_text('\n'.join(q)+'\n');print(len(q)-1,'units',{t:sum(e-s for s,e in spans(t)) for t in V})
    (O/'queues/gabl_repro.txt').write_text(f'chair_test500 gabl_repro {FULL} 0 5 --attn eager {S} --lam 1 --pen 0.5\n')
elif cmd=='repro':
    full={r['id']:r for r in read(FF)};same=0;new=read(O/f'runs/gabl_repro/{FULL}-0-5.jsonl');assert [x['id'] for x in new]==ref[:5],'repro rows'
    for r in new:
        a,b=r['prediction'],full[r['id']]['prediction'];same+=a==b
        print(r['id'],'identical' if a==b else f'differs at char {next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))} of {len(b)}')
    print(f'repro {same}/5 identical to the stored full row');sys.exit(0 if same else 1)
elif cmd=='join':
    full=read(FF);base=read(BF);tbl=read(TF)
    A=O/'runs/gabl_chair500_all';Fd=O/'runs/gabl_chair500_vsfull';A.mkdir(exist_ok=True);Fd.mkdir(exist_ok=True)
    def w(d,t,rows):(d/f'{t}-0-500.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    w(A,'base',base);w(A,'rbd_full',full);w(Fd,'base',full);w(Fd,'vddbase',base);w(Fd,'tblbase',tbl)
    for t in V:
        rows=[]
        if t in REUSE:rows=read(REUSE[t][0])
        for s,e in spans(t):
            f=O/f'runs/gabl_chair500/{t}-{s}-{e}.jsonl';r=read(f) if f.exists() else [];assert len(r)==e-s,f'{f} incomplete';rows+=r
        assert [x['id'] for x in rows]==ref,t
        assert len({st(x) for x in rows})==1,f'{t}: settings differ between reused and new rows'
        w(A,t,rows);w(Fd,t,rows)
    print('joined',['base','rbd_full',*V])
elif cmd=='table':
    X=json.loads((Path(sys.argv[2])/'score.json').read_text())['results']
    def nums(t):v=X[t];return f"{v['CHAIR_s']:.1f} & {v['CHAIR_i']:.1f} & {v['recall']:.1f}"
    def row(n,t):c=X[t]['ci'];f=lambda q:f"[${c[q][0]:+.1f}$, ${c[q][1]:+.1f}$]";return f"    {n} & {nums(t)} & {f('CHAIR_s')} & {f('recall')} \\\\"
    print('\n'.join([r'\begin{table*}[t]',r'  \centering',r'  \caption{Ablation Study on Proposed RBD Components (Gemma 4 E4B)}',r'  \label{tab:ablation_study_gemma}',r'  \small',
      r'  \setlength{\tabcolsep}{4pt}',r'  \begin{tabular}{lccccc}',r'    \toprule',
      r'    \textbf{Configuration} & \textbf{CHAIR$_s$ ($\downarrow$)} & \textbf{CHAIR$_i$ ($\downarrow$)} & \textbf{Recall ($\uparrow$)} & \textbf{$\Delta$CHAIR$_s$ vs. Full} & \textbf{$\Delta$Recall vs. Full} \\',
      r'    \midrule',f"    Full (RBD) & {nums('base')} & -- & -- \\\\",r'    \midrule',
      row('$-$ steering','abl_nosteer'),row('$-$ penalty','abl_nopen'),row('$-$ belief','abl_nobelief'),r'    \midrule',
      row('w/o decay schedule','abl_const'),row('belief only','abl_belief'),row('penalty only','abl_pen'),row('steering only','abl_steer'),r'    \midrule',
      row('Base (Vanilla)','vddbase'),r'    \bottomrule',r'  \end{tabular}',r'\end{table*}']))
