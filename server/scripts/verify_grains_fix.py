import json
from pathlib import Path
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import extract_baseline
import torch
r=Path("/home/project-b/LJW/vlm-grounding-steering")
c=json.loads((r/"configs/gemma4_e4b.local.json").read_text())
record=json.loads((r/"data/mmvp-full/fit.jsonl").read_text().splitlines()[0])
a=load_adapter(c["model"])
artifact=extract_baseline("grains",a,[record],c["methods"]["grains"])
norms=[float(v.norm()) for v in artifact["vectors"].values()]
assert all(torch.isfinite(v).all() for v in artifact["vectors"].values()) and max(norms)>0
print(json.dumps({"status":"passed","quadrature":artifact["ig_quadrature"],"counts":artifact["attribution_token_counts"],"max_vector_norm":max(norms),"nonzero_layers":sum(v>0 for v in norms)}),flush=True)
