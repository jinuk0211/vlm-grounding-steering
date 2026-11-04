"""Shared pieces of the RBD (reliability-weighted belief decoding) pilot.

V_obj   word-initial vocabulary tokens whose word is a physical-object noun (WordNet, same filter as open_vocab.py).
r(t)    image reliance at caption position t: the E3 gray-image KL curve of visual_decay, normalised to 1 at the
        first bin, piecewise-linear between bin centres, held flat after the last one.
"""
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/project-b/LJW/vlm-grounding-steering')
DECAY=ROOT/'outputs/visual-decay-20261030'
BINS=[(0,15),(16,31),(32,63),(64,95),(96,127),(128,191),(192,255),(256,383),(384,512)]

def build_vobj(tok):
    """Sorted ids of word-initial tokens (leading space / SentencePiece marker) that spell a physical-object noun."""
    from open_vocab import physical,lemma,noun_dominant
    from nltk.corpus import wordnet as wn
    def freq(w):return sum(l.count() for s in wn.synsets(w,pos=wn.NOUN) for l in s.lemmas() if l.name().lower()==w)
    ids=[]
    for i in range(len(tok)):
        s=tok.convert_ids_to_tokens(i)
        if not s or s[0] not in ('Ġ','▁'):continue
        w=s[1:]
        # lowercase whole words only (mentions are mid-sentence); proper nouns and word pieces drop out
        if not (w.isalpha() and w.islower() and len(w)>=3):continue
        lw=lemma(w)
        if lw==w or lw+'s'==w or lw+'es'==w:   # the token is the word itself or its plain plural
            if physical(lw) and noun_dominant(lw) and freq(lw)>=1:ids.append(i)
    return ids

def r_curve(model,kind='measured'):
    """Return f(t)->r for t=0..; kind in measured | const | step0 | late_only (control)."""
    if kind=='const':return lambda t:1.0
    if kind=='step0':return lambda t:1.0 if t==0 else 0.0
    s=json.loads((DECAY/f'{model}-visual-decay-summary.json').read_text())
    kl={r['bin']:r['mean'] for r in s['E3_causal']['by_bin']['gray']['kl']}
    keys=[k for k in (f'{a}-{b}' for a,b in BINS) if k in kl]
    xs=[(int(k.split('-')[0])+int(k.split('-')[1]))/2 for k in keys];ys=np.array([kl[k] for k in keys]);ys=ys/ys[0]
    return lambda t:float(np.interp(t,xs,ys))

def r_array(model,kind,T):
    f=r_curve(model,kind);return np.array([f(t) for t in range(T)],dtype=np.float32)
