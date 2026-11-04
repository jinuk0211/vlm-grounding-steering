"""HTML report (inline SVG, no plotting library) from visual_decay_analyze.py summaries.

usage: visual_decay_plot.py OUT.html SUMMARY.json [SUMMARY.json ...]
Each chart has a hover tooltip per point/cell and a table view.
"""
import html,json,math,sys
from pathlib import Path

SERIES=['var(--s1)','var(--s2)','var(--s3)','var(--s4)']
SEQ=['#cde2fb','#b7d3f6','#9ec5f4','#86b6ef','#6da7ec','#5598e7','#3987e5','#2a78d6','#256abf','#1c5cab','#184f95','#104281','#0d366b']
DIV_NEG=['#0d366b','#1c5cab','#2a78d6','#5598e7','#86b6ef','#b7d3f6']  # blue arm, dark -> light
DIV_POS=['#f6c9c8','#f0a3a2','#e87c7b','#e34948','#c13a39','#8f2a29']  # red arm, light -> dark
MID='#f0efec'
CSS="""
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);color:var(--text-primary);margin:0;padding:24px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;
 --grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}p.note{color:var(--text-secondary);font-size:13px;max-width:1100px;margin:4px 0 12px}
.row{display:flex;flex-wrap:wrap;gap:16px}.card{background:var(--surface-1);border-radius:8px;padding:12px 14px;box-shadow:0 0 0 1px rgba(11,11,11,.10)}
.card h3{font-size:14px;margin:0}.card .sub{font-size:12px;color:var(--text-secondary);margin:2px 0 6px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--text-secondary);margin:4px 0}.legend i{display:inline-block;width:14px;height:2px;margin-right:5px;vertical-align:middle;border-radius:1px}
svg text{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}svg .lab{fill:var(--text-secondary)}
svg .hit:hover+.dot,svg .dot:hover{stroke-width:3}
details{font-size:12px;color:var(--text-secondary);margin-top:4px}table{border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:4px}
td,th{padding:2px 8px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}
"""

def fmt(v,d=3):
    if v is None or (isinstance(v,float) and math.isnan(v)):return '–'
    return f'{v:.{d}f}'

def nice_ticks(lo,hi,n=5):
    if hi<=lo:hi=lo+1
    step=10**math.floor(math.log10((hi-lo)/n));step*=next(m for m in (1,2,2.5,5,10) if (hi-lo)/(step*m)<=n)
    a=math.floor(lo/step)*step;t=[];x=a
    while x<=hi+step*1e-9:t.append(round(x,10));x+=step
    if t[-1]<hi:t.append(round(x,10))
    return t

def table_html(head,rows):
    h=''.join(f'<th>{html.escape(str(x))}</th>' for x in head)
    b=''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in r)+'</tr>' for r in rows)
    return f'<details><summary>Table</summary><table><tr>{h}</tr>{b}</table></details>'

def line_chart(title,sub,x,series,ylab,digits=3,zero=False,W=500,H=250):
    """series: list of (name, [value|None], [[lo,hi]|None] or None)"""
    ml,mr,mt,mb=52,16,12,42;pw,ph=W-ml-mr,H-mt-mb
    vals=[v for _,ys,cs in series for v in ys if v is not None]+[c for _,_,cs in series if cs for p in cs if p for c in p if c is not None]
    if not vals:return ''
    lo,hi=min(vals),max(vals)
    if zero:lo=min(lo,0);hi=max(hi,0)
    pad=(hi-lo)*0.06 or abs(hi)*0.1 or 1;ticks=nice_ticks(lo-pad,hi+pad)
    # shares and AUROCs live in [0,1]: keep the axis inside the range the data can take
    if min(vals)>=0 and ticks[0]<0:ticks=[t for t in ticks if t>=0]
    if max(vals)<=1 and ticks[-1]>1:ticks=[t for t in ticks if t<=1]
    lo,hi=ticks[0],ticks[-1]
    X=lambda i:ml+(pw*(i+0.5)/len(x));Y=lambda v:mt+ph*(1-(v-lo)/(hi-lo))
    s=[f'<svg width="{W}" height="{H}" role="img" aria-label="{html.escape(title)}">']
    for t in ticks:s.append(f'<line x1="{ml}" x2="{W-mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" stroke="var(--grid)" stroke-width="1"/><text x="{ml-6}" y="{Y(t)+4:.1f}" text-anchor="end">{t:g}</text>')
    if lo<0<hi:s.append(f'<line x1="{ml}" x2="{W-mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1"/>')
    for i,l in enumerate(x):s.append(f'<text x="{X(i):.1f}" y="{H-mb+16}" text-anchor="middle">{html.escape(l)}</text>')
    s.append(f'<text x="{ml+pw/2}" y="{H-6}" text-anchor="middle" class="lab">{html.escape(ylab[0])}</text>')
    s.append(f'<text x="12" y="{mt+ph/2}" text-anchor="middle" class="lab" transform="rotate(-90 12 {mt+ph/2})">{html.escape(ylab[1])}</text>')
    for k,(name,ys,cs) in enumerate(series):
        col=SERIES[k]
        if cs:
            up=[(X(i),Y(c[1])) for i,c in enumerate(cs) if c and c[1] is not None];dn=[(X(i),Y(c[0])) for i,c in enumerate(cs) if c and c[0] is not None]
            if up:s.append('<polygon points="'+' '.join(f'{a:.1f},{b:.1f}' for a,b in up+dn[::-1])+f'" fill="{col}" fill-opacity="0.1" stroke="none"/>')
        seg=[];segs=[]
        for i,v in enumerate(ys):
            if v is None:
                if seg:segs.append(seg);seg=[]
            else:seg.append((X(i),Y(v)))
        if seg:segs.append(seg)
        for g in segs:s.append('<polyline points="'+' '.join(f'{a:.1f},{b:.1f}' for a,b in g)+f'" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i,v in enumerate(ys):
            if v is None:continue
            c=cs[i] if cs and cs[i] else None;tip=f'{name} · {x[i]}: {fmt(v,digits)}'+(f' (95% CI {fmt(c[0],digits)} to {fmt(c[1],digits)})' if c and c[0] is not None else '')
            s.append(f'<g><title>{html.escape(tip)}</title><circle class="hit" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="9" fill="transparent"/>'
                     f'<circle class="dot" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" stroke="var(--surface-1)" stroke-width="2"/></g>')
    s.append('</svg>')
    leg='' if len(series)<2 else '<div class="legend">'+''.join(f'<span><i style="background:{SERIES[k]}"></i>{html.escape(n)}</span>' for k,(n,_,_) in enumerate(series))+'</div>'
    tab=table_html(['position']+[n for n,_,_ in series],[[l]+[fmt(ys[i],digits) for _,ys,_ in series] for i,l in enumerate(x)])
    return f'<div class="card"><h3>{html.escape(title)}</h3><div class="sub">{html.escape(sub)}</div>{leg}{"".join(s)}{tab}</div>'

def color_for(v,lo,hi,diverging):
    if v is None or math.isnan(v):return 'var(--grid)'
    if diverging:
        m=max(abs(lo),abs(hi)) or 1;f=v/m
        if abs(f)<0.08:return MID
        arm=DIV_POS if f>0 else DIV_NEG[::-1];return arm[min(len(arm)-1,int(abs(f)*len(arm)))]
    f=(v-lo)/((hi-lo) or 1);return SEQ[min(len(SEQ)-1,int(f*len(SEQ)))]

def heatmap(title,sub,x,ylabels,M,diverging=False,digits=3,W=500,ytitle='decoder layer'):
    flat=[v for r in M for v in r if v is not None and not math.isnan(v)]
    if not flat:return ''
    lo,hi=min(flat),max(flat);ml,mt,mb=52,8,40;cw=(W-ml-16)/len(x);ch=max(4,min(14,220/len(M)));H=int(mt+ch*len(M)+mb)
    s=[f'<svg width="{W}" height="{H}" role="img" aria-label="{html.escape(title)}">']
    for j,r in enumerate(M):
        yy=mt+ch*j
        if j%max(1,len(M)//7)==0 or j==len(M)-1:s.append(f'<text x="{ml-6}" y="{yy+ch/2+4:.1f}" text-anchor="end">{html.escape(str(ylabels[j]))}</text>')
        for i,v in enumerate(r):
            s.append(f'<rect x="{ml+cw*i+1:.1f}" y="{yy+0.5:.1f}" width="{cw-2:.1f}" height="{ch-1:.1f}" rx="1" fill="{color_for(v,lo,hi,diverging)}">'
                     f'<title>{html.escape(f"layer {ylabels[j]} · {x[i]}: {fmt(v,digits)}")}</title></rect>')
    for i,l in enumerate(x):s.append(f'<text x="{ml+cw*(i+0.5):.1f}" y="{H-mb+14}" text-anchor="middle">{html.escape(l)}</text>')
    s.append(f'<text x="{ml+(W-ml)/2}" y="{H-6}" text-anchor="middle" class="lab">caption position (tokens)</text>')
    s.append(f'<text x="12" y="{mt+ch*len(M)/2}" text-anchor="middle" class="lab" transform="rotate(-90 12 {mt+ch*len(M)/2})">{html.escape(ytitle)}</text></svg>')
    scale=(f'<div class="legend"><span>blue = {fmt(-max(abs(lo),abs(hi)),digits)}</span><span>gray = 0</span><span>red = {fmt(max(abs(lo),abs(hi)),digits)}</span></div>' if diverging
           else f'<div class="legend"><span>light = {fmt(lo,digits)}</span><span>dark = {fmt(hi,digits)}</span></div>')
    tab=table_html(['layer']+x,[[ylabels[j]]+[fmt(v,digits) for v in r] for j,r in enumerate(M)])
    return f'<div class="card"><h3>{html.escape(title)}</h3><div class="sub">{html.escape(sub)}</div>{scale}{"".join(s)}{tab}</div>'

def col(tab,key='mean'):return [r[key] for r in tab]
def by_bin(tab,bins):
    d={r['bin']:r for r in tab};return [d[b]['mean'] if b in d else None for b in bins],[d[b]['ci95'] if b in d else None for b in bins]

def model_section(s):
    bins=[b for b in s['bins'] if any(r['bin']==b for r in s['E1_attention']['bands']['all']['image'])]
    e1=s['E1_attention'];e2=s['E2_lens_repr'];e3=s['E3_causal'];out=[]
    out.append(f'<h2>{html.escape(s["model"])} — {s["captions"]} captions, {s["tokens"]:,} caption tokens, mean length {s["caption_length"]["mean"]:.0f}, '
               f'{s["image_tokens"]} image tokens</h2>')
    # E1
    ser=[(b,*by_bin(e1['bands'][b]['image_share'],bins)) for b in ('early','middle','late')]
    c=[line_chart('Attention on image tokens, by layer band','share of content attention (pre-image sink tokens removed) from the step that writes each token',bins,ser,('caption position (tokens)','image share'))]
    a=e1['bands']['all'];ser=[(n,*by_bin(a[k],bins)) for n,k in (('image','image_share'),('prompt text','prompt_share'),('caption so far','caption_share'))]
    c.append(line_chart('Where the content attention goes (all layers)','the three shares add up to 1; text = prompt text + caption so far',bins,ser,('caption position (tokens)','share')))
    c.append(line_chart('Attention on pre-image tokens','raw share on the template/system tokens before the image (where sinks sit), all layers',bins,[('pre-image',*by_bin(a['pre_image'],bins))],('caption position (tokens)','share')))
    hm=e1['per_layer_image_share'];idx=[s['bins'].index(b) for b in bins]
    c.append(heatmap('Image share per layer','content attention on image tokens',bins,list(range(1,len(hm['matrix'])+1)),[[r[i] for i in idx] for r in hm['matrix']]))
    c.append(line_chart('Attention rollout reaching the image','share of attention paths (0.5·A + 0.5·I, head mean) that end on an image token',bins,[('rollout',*by_bin(e1['rollout_absorbing_image']['image'],bins))],('caption position (tokens)','path share')))
    if 'window_limited_layers' in e1['bands']:
        ser=[('sliding-window layers',*by_bin(e1['bands']['window_limited_layers']['image_keys_visible'],bins)),('full-attention layers',*by_bin(e1['bands']['full_attention_layers']['image_keys_visible'],bins))]
        c.append(line_chart('Image tokens the step can still see','share of image keys left unmasked by the layer\'s attention mask',bins,ser,('caption position (tokens)','visible share')))
    if 'most_attended_prompt_positions' in e1:
        rows=[[x['position'],x['group'],fmt(x['attention_received']),fmt(x['hidden_norm_over_median_at_mid_layer'],1)] for x in e1['most_attended_prompt_positions']]
        c.append('<div class="card"><h3>Where the sinks sit</h3><div class="sub">prompt positions receiving the most attention from caption steps (mean over layers); '
                 'a large hidden-state norm marks a massive-activation sink</div>'+table_html(['position','group','attention','norm ÷ median'],rows).replace('<details>','<details open>')+'</div>')
    out.append('<h3>E1 · attention</h3><div class="row">'+''.join(c)+'</div>')
    # E2
    c=[];pl=e2['probe_layers'];idx=[s['bins'].index(b) for b in bins]
    M=e2['per_layer']['lens_image_contribution']['matrix']
    c.append(heatmap('Logit lens: image contribution per layer','log p(token | real image) − log p(token | gray image), same caption; red = the image helps',bins,list(range(len(M))),[[r[i] for i in idx] for r in M],diverging=True,digits=2,ytitle='layer (0 = embedding)'))
    for key,t,yl in (('lens_image_contribution','Logit-lens image contribution at four depths','log p real − gray'),
                     ('delta_norm_ratio','How much the image changes the hidden state','‖h_real − h_gray‖ / ‖h_real‖'),
                     ('cos_to_image_mean_centred','Similarity to the mean image-token state','centred cosine')):
        ser=[(f'layer {l}',*by_bin(e2['at_probe_layers'][key][str(l)],bins)) for l in pl]
        c.append(line_chart(t,'at 25 / 50 / 75 / 100 % depth',bins,ser,('caption position (tokens)',yl),zero=key=='lens_image_contribution'))
    pr=e2['probes'];coh=pr['cohort_results']
    steps=sorted({int(k) for l in coh for k in coh[l]})
    ser=[(f'layer {l}',[coh[str(l)].get(str(t),{}).get('image_gain_unmentioned') for t in steps],None) for l in pl]
    c.append(line_chart('Probe: object presence the image adds','AUROC(real) − AUROC(gray), objects not yet mentioned; '
                        f'fixed cohort of {pr["cohort"]["n"]} captions reaching step {pr["cohort"]["max_step"]}',[str(t) for t in steps],ser,('caption step','AUROC gain'),zero=True))
    mid=str(pl[1]);ser=[(n,[coh[mid].get(str(t),{}).get(k) for t in steps],None) for n,k in (('real image','real_auroc_unmentioned'),('gray image (text only)','gray_auroc_unmentioned'))]
    c.append(line_chart(f'Probe AUROC at layer {mid}','objects not yet mentioned; ceiling from pooled image tokens = '+fmt(pr.get('image_tokens_ceiling',{}).get(mid)),[str(t) for t in steps],ser,('caption step','macro AUROC')))
    out.append('<h3>E2 · logit lens and representations</h3><div class="row">'+''.join(c)+'</div>')
    # E3
    c=[];conds=[x for x in ('gray','swap','ko0','ko64') if x in e3['by_bin']]
    names={'gray':'gray image','swap':'another image','ko0':'image keys blocked from step 0','ko64':'blocked from step 64'}
    for key,t,yl in (('kl','Change in the next-token distribution','KL(unmodified ‖ intervened)'),('greedy_agreement','Greedy choice unchanged','agreement')):
        ser=[(names[x],*by_bin(e3['by_bin'][x][key],bins)) for x in conds]
        c.append(line_chart(t,'teacher-forced Base caption',bins,ser,('caption position (tokens)',yl)))
    rows=[]
    for k,v in e3['direct_vs_total'].items():
        for cnd in [x for x in v if x.endswith('_over_gray')]:
            rows.append([k,cnd.replace('_over_gray',''),fmt(v[cnd]['ratio']),f'{fmt(v[cnd]["ci95"][0])} to {fmt(v[cnd]["ci95"][1])}',v['n']])
    c.append('<div class="card"><h3>Direct vs total image effect</h3><div class="sub">KL with image keys blocked ÷ KL with a gray image, tokens after k. '
             'Near 1: late tokens read the image directly. Near 0: the image reaches them only through earlier tokens, or not at all.</div>'
             +table_html(['tokens','blocked from','ratio','95% CI','captions'],rows).replace('<details>','<details open>')+'</div>')
    g=e3['teacher_forced_greedy_reproduction']
    out.append(f'<h3>E3 · causal interventions</h3><p class="note">Teacher-forced greedy reproduction of the analysed captions: '
               f'{g["with_generation_config_repetition_penalty"]:.3f} with the model\'s repetition penalty, {g["raw_argmax"]:.3f} by raw argmax.</p><div class="row">'+''.join(c)+'</div>')
    return ''.join(out)

def main():
    out=Path(sys.argv[1]);sums=[json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    body=''.join(model_section(s) for s in sums)
    doc=(f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Visual information along the caption</title><style>{CSS}</style></head>'
         f'<body class="viz-root"><h1>Does the image fade as the caption gets longer?</h1>'
         f'<p class="note">Base CHAIR captions, teacher-forced. Position bins are caption tokens. Shaded bands are 95% bootstrap CIs over captions. '
         f'Hover a point or cell for its value; each chart has a table view.</p>{body}</body></html>')
    out.write_text(doc,encoding='utf-8');print('wrote',out)

if __name__=='__main__':main()
