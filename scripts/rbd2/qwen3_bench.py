"""Qwen3-VL-8B baseline runner for CHAIR 500 (RUDDER, REVIS, VTI).
"""
import argparse, contextlib, hashlib, json, os, time, traceback
from pathlib import Path
import torch
from author_clone_bridge import load_authors, rudder_context
from vlm_grounding.backend import load_adapter
from vlm_grounding.baselines import build_baseline
from vlm_grounding.cli import _seed

ROOT = Path('/home/project-b/LJW/vlm-grounding-steering')
CFG_PATH = ROOT / 'configs/qwen3_vl_8b.json'

def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def setup(m, adapter, config):
    if m == 'base': return lambda batch, diag: contextlib.nullcontext()
    if m == 'rudder_clone':
        authors = load_authors()
        return lambda batch, diag: rudder_context(adapter, batch, 'chair', authors, diag)
    if m in ('revis', 'vti'):
        p = ROOT / f'outputs/qwen3-vl8b/artifacts/{m}.pt'
        art = torch.load(p, map_location='cpu', weights_only=True)
        for g in ('vectors', 'vision_vectors'):
            if g in art: art[g] = {k: v.to(adapter.device) for k, v in art[g].items()}
        return lambda batch, diag: build_baseline(m, adapter, art, config['methods'][m])
    raise ValueError(f"Unsupported baseline method for qwen3: {m}")

def check(m, diag):
    if m == 'rudder_clone': assert diag.get('calls', 0) > 0

def main(a):
    m = a.method
    assert os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.device_count() == 1
    config = json.loads(CFG_PATH.read_text())
    cfg = dict(config['model'], dtype='bfloat16', attn_implementation=a.attn)
    refs = read(a.set)[a.start:a.end]
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    name = f'{m}-{a.start}-{a.end}'
    rows_file = out / f'{name}.jsonl'
    status = out / f'{name}-status.json'
    done = read(rows_file) if rows_file.exists() else []
    for r, ref in zip(done, refs): assert r['id'] == ref['id'] and r['method'] == m
    def update(**k):
        s = json.loads(status.read_text()) if status.exists() else {}
        s.update(k, updated_at=time.time())
        status.write_text(json.dumps(s, indent=1))
    update(stage='loading', pid=os.getpid(), method=m, set=str(a.set), dtype=cfg['dtype'], attn=a.attn,
           total=len(refs), done=len(done), resumed_from=len(done))
    try:
        adapter = load_adapter(cfg)
        assert adapter.model_type == 'qwen3_vl'
        ctx = setup(m, adapter, config)
        update(stage='running', gpu=torch.cuda.get_device_name(0))
        with rows_file.open('a') as f:
            for i, ref in enumerate(refs[len(done):], start=len(done)):
                assert sha(ref['image']) == ref['image_sha256'], ref['image']
                _seed(config)
                batch = adapter.prepare(ref['image'], ref['prompt'])
                batch_cuda = {k: v.to(adapter.device) for k, v in batch.items() if torch.is_tensor(v)}
                diag = {}
                t0 = time.perf_counter()
                with ctx(batch_cuda, diag):
                    text = adapter.generate(batch_cuda, max_new_tokens=int(ref['max_new_tokens']))
                check(m, diag)
                f.write(json.dumps({**ref, 'method': m, 'prediction': text, 'generated_tokens': adapter.last_generated_tokens,
                    'elapsed_seconds': time.perf_counter() - t0, 'dtype': cfg['dtype'], 'attn_implementation': a.attn,
                    'gpu': torch.cuda.get_device_name(0), 'author_diagnostics': {k: str(v) for k, v in diag.items()}}) + '\n')
                f.flush()
                update(done=i + 1)
        update(stage='complete')
    except BaseException:
        update(stage='failed', error=traceback.format_exc()); raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['base', 'rudder_clone', 'revis', 'vti'], required=True)
    p.add_argument('--set', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--attn', choices=['eager', 'sdpa'], default='eager')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=500)
    main(p.parse_args())
