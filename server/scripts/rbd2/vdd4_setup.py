"""One-off setup of the colour-fixed-V_obj THRONE run (vdd4; request of session h-35, user-approved 11-04):
  1. outputs/rbd-20261030/collect/<model>-vobj2cf.json = <model>-vobj2.json minus the tokens whose word is mainly a colour
     term in captions (DROP below); the old files stay. Prints the removed tokens and the colour-like words kept.
  2. scripts/rbd2/vdd4_decode.py = vdd_decode.py + `--vobj PATH` (default: the old file). Nothing else changes.
  3. sets/throne_cfdev200.jsonl = throne_full rows 4000-4199 (disjoint from throne_full1000 and the Gemma dev 4940-4999)."""
import glob,json,sys
from pathlib import Path
from transformers import AutoProcessor
R=Path('/home/project-b/LJW/vlm-grounding-steering');O=R/'outputs/rbd2-20261102';C=R/'outputs/rbd-20261030/collect'
# words used mainly as colours in captions; 'oranges' (fruit plural), rose, wine, chocolate, lemon, mint, sand, salmon etc. stay
DROP={'orange','cream','creams','ivory','violet','coral','bronze','copper'}
LOOKED_AT='orange gold golden silver lime olive peach plum rose salmon cream coral ivory amber tan navy maroon violet lavender chocolate wine berry bronze copper mint lemon sand'.split()
for model,pat in (('qwen','Qwen--Qwen2.5-VL-7B-Instruct'),('gemma','google--gemma-4-E4B-it')):
    snap=glob.glob(f'/home/project-b/LJW/.cache/huggingface/hub/models--{pat}/snapshots/*')[0];tok=AutoProcessor.from_pretrained(snap).tokenizer
    v=json.loads((C/f'{model}-vobj2.json').read_text());word=lambda i:tok.decode([i]).strip().lower()
    out=[i for i in v if word(i) not in DROP];gone=[(i,tok.decode([i])) for i in v if word(i) in DROP]
    (C/f'{model}-vobj2cf.json').write_text(json.dumps(out))
    print(f'{model}: vobj {len(v)} -> {len(out)}; removed {gone}')
    print(f'   colour-like words checked and kept: {sorted({word(i) for i in out if word(i) in LOOKED_AT or word(i).rstrip("s") in LOOKED_AT})}')
src=(R/'scripts/rbd2/vdd_decode.py').read_text()
old="vobj=json.loads((ROOT/f'outputs/rbd-20261030/collect/{a.model}-vobj2.json').read_text())"
new="vobj=json.loads((Path(a.vobj) if a.vobj else ROOT/f'outputs/rbd-20261030/collect/{a.model}-vobj2.json').read_text())"
assert src.count(old)==1;src=src.replace(old,new)
old2="    main(p.parse_args())";assert src.count(old2)==1;src=src.replace(old2,"    p.add_argument('--vobj',default=None)\n    main(p.parse_args())")
old3="'decoder':'vdd_decode.py','setting':setting}";assert src.count(old3)==1;src=src.replace(old3,"'decoder':'vdd4_decode.py','setting':{**setting,'vobj':a.vobj or 'vobj2'}}")
(R/'scripts/rbd2/vdd4_decode.py').write_text(src);compile(src,'vdd4_decode.py','exec')
full=[json.loads(l) for l in open(O/'sets/throne_full.jsonl') if l.strip()];dev=full[4000:4200]
test={r['id'] for r in full[:1000]};assert len(dev)==200 and not test&{r['id'] for r in dev}
(O/'sets/throne_cfdev200.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dev));print('throne_cfdev200: throne_full rows 4000-4199')
