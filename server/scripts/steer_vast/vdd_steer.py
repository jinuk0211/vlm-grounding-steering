"""Steering variants for TETHER (RBD round 2): belief and penalty exactly as vdd_decode.py, only the steering step changes.

Rows decoded in lockstep: 0 = real image, 1 = 448x448 gray image (the belief's control, as vdd_decode.py), and with
--ctrl blur a third row = Gaussian-blurred copy of the real image. Every row takes row 0's token. Belief/penalty:
A(o) += r(t) (log p0(o) - log p1(o)); logits0(o) += w(t) (lam clip(A/R) - pen) for objects not yet written (vdd_decode.py).

--steer (decode steps only; prompt prefill untouched; s(t) = 1 - r(t) 'decay' or 1 'const'; c = control row):
  resid   h_L[0] += alpha s(t) (h_L[0] - h_L[c])                   output of each decoder layer L in --layers (vdd_decode: one L)
  norm    as resid, difference rescaled to rho |h_L[0]|            (constant strength relative to h along the caption)
  anchor  resid for caption steps 1..K (--anchor-steps); afterwards the mean difference of those steps is added
          (re-injects the early visual state; real-vs-gray evidence is strongest at the start of a caption)
  attn    u_L[0] += alpha s(t) (u_L[0] - u_L[c])                   u_L = attention update (Qwen: o_proj output; Gemma: post-attention norm)
  pai     attention logits of the real row to image keys += log(1 + beta s(t)) in --layers (PAI-style: read the image more)
  sel     h_L[0] += s(t) rho |h_L[0]| v/|v|,  v = sum_o (clip(A_o/R) - pen/lam) g*W_U[o]/|g*W_U[o]| over objects not yet written
          (belief-guided: toward objects with evidence above the penalty threshold, away from the others)
  none    no steering (the -steer ablation)
Layers are 1-based decoder-layer numbers (vdd_decode.py: --layer 16 = output of decoder layer 16).
"""
import argparse, json, math, os, sys, tempfile, time, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from transformers import LogitsProcessor, LogitsProcessorList

from vlm_grounding.backend import load_adapter, _first_path
import read_gap_collect_v2 as C
from rbd_common import ROOT, r_curve

GRAY = Path(os.environ.get('VDD_GRAY', '/workspace/rbd_pilot/gray448.png'))

_linear = F.linear
def _rowsplit_linear(x, w, b=None):
    # Turing: a bf16 matmul with 2-3 rows is several times slower than 1 row; decode steps go row by row (as vdd_decode.py)
    if x.dtype == torch.bfloat16 and x.dim() == 3 and x.shape[0] in (2, 3) and x.shape[1] == 1:
        return torch.cat([_linear(x[i:i + 1], w, b) for i in range(x.shape[0])], 0)
    return _linear(x, w, b)
F.linear = _rowsplit_linear


class State:
    def __init__(self):
        self.t = 0; self.on = False; self.proc = None; self.anchor = {}; self.anchor_n = {}; self.img_pos = None; self.pai_calls = 0


class Evidence(LogitsProcessor):
    """belief + penalty on row 0 (rows 0 vs 1), then every row gets row 0's scores so all rows take the same token"""
    def __init__(self, st, vidx, rf, a, pl):
        self.st = st; self.vidx = vidx; self.rf = rf; self.a = a; self.pl = pl
        self.A = torch.zeros(len(vidx), device=vidx.device); self.R = 0.0
        self.Am = torch.zeros(len(vidx), device=vidx.device)      # median-normalised belief, used only by sel --sel-belief median
        self.done = torch.zeros(len(vidx), dtype=torch.bool, device=vidx.device)
        self.kl_last = None; self.kl_sum = 0.0; self.kl_n = 0      # --ssched gate: per-step image reliance KL(p_real || p_gray)
    def w(self, t): return 1 - self.rf(t) if self.a.bsched == 'decay' else 1.0
    def __call__(self, input_ids, scores):
        t = input_ids.shape[1] - self.pl; self.st.t = t + 1
        a = self.a
        if a.ssched == 'gate':   # raw logits of this step, before any adjustment
            l0 = scores[0].float().log_softmax(-1); l1 = scores[1].float().log_softmax(-1)
            kl = float((l0.exp() * (l0 - l1)).sum()); self.kl_last = kl; self.kl_sum += kl; self.kl_n += 1
        if a.lam or 'sel' in (a.steer, a.steer2):
            lp = scores.float().log_softmax(-1)[:, self.vidx]
            d = lp[0] - lp[1]
            dm = d - d.median()
            if a.norm == 'median': d = dm
            r = self.rf(t); self.A += r * d; self.Am += r * dm; self.R += r
        self.done = torch.isin(self.vidx, input_ids[0, self.pl:])
        if a.lam or a.pen:
            adj = torch.zeros_like(self.A)
            if a.lam: adj += a.lam * self.w(t) * (self.A / max(self.R, 1e-6)).clamp(0.0 if a.bpos else -a.clip, a.clip)
            if a.pen: adj -= a.pen * self.w(t)
            adj = torch.where(self.done, torch.zeros_like(adj), adj); scores[0, self.vidx] += adj.to(scores.dtype)
        for r_ in range(1, scores.shape[0]): scores[r_] = scores[0]
        return scores


class Steer:
    def __init__(self, adapter, st, a, rf, ctrl_row, vidx):
        self.st = st; self.a = a; self.rf = rf; self.c = ctrl_row; self.calls = 0
        if a.steer in ('resid', 'norm', 'anchor', 'anchornorm', 'sel', 'restore'):
            for L in a.layers: adapter.layers[L - 1].register_forward_hook(self._resid_hook(L))
        elif a.steer == 'attn':
            for L in a.layers: adapter.attn_updates[L - 1].register_forward_hook(self._attn_hook(L))
        elif a.steer == 'pai':
            install_pai(adapter, st, a, rf, {L - 1 for L in a.layers})
        if a.steer == 'sel':
            W = adapter.model.get_output_embeddings().weight
            norm = _first_path(adapter.model, ('model.language_model.norm', 'language_model.model.norm', 'model.norm'))
            g = norm.weight.float()
            if 'gemma' in type(norm).__name__.lower(): g = 1.0 + g        # Gemma RMSNorm scales by (1 + weight)
            U = W[vidx].float() * g[None]
            self.U = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-6)     # [|V_obj|, d]

    def s(self, t):
        if self.a.ssched == 'gate':   # s(t) = 1 - r(t) only on steps whose previous-step reliance is below the caption's running mean
            p = self.st.proc
            if p is None or p.kl_n == 0 or p.kl_last is None: return 0.0
            return (1 - self.rf(t)) * (1.0 if p.kl_last < p.kl_sum / p.kl_n else 0.0)
        return 1 - self.rf(t) if self.a.ssched == 'decay' else (self.rf(t) if self.a.ssched == 'early' else 1.0)

    def _live(self, h): return self.st.on and h.shape[1] == 1 and h.shape[0] > self.c

    def _resid_hook(self, L):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if not self._live(h): return None
            a = self.a; t = self.st.t; s = self.s(t)
            h0 = h[0].float(); d = (h[0] - h[self.c]).float()        # same operation order as vdd_decode.py
            if a.steer == 'resid':
                add = a.alpha * s * d
            elif a.steer == 'norm':
                add = a.alpha * s * a.rho * h0.norm(dim=-1, keepdim=True) * d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            elif a.steer == 'restore':
                # alpha_t chosen so the image contribution regains alpha x its mean norm over caption steps 1..K (no steering
                # while that norm is measured); direction is the current d_t, alpha_t capped at rho. Eq. steer with alpha -> alpha_t.
                dn = d.norm(dim=-1, keepdim=True); k = ('rn', L)
                if t <= a.anchor_steps or k not in self.st.anchor_n:
                    self.st.anchor[k] = self.st.anchor.get(k, 0) + dn; self.st.anchor_n[k] = self.st.anchor_n.get(k, 0) + 1
                    return None
                m0 = self.st.anchor[k] / self.st.anchor_n[k]
                at = (a.alpha * m0 / dn.clamp_min(1e-6) - 1.0).clamp(0.0, a.rho)
                add = s * at * d
            elif a.steer in ('anchor', 'anchornorm'):
                if t <= a.anchor_steps or L not in self.st.anchor_n:
                    self.st.anchor[L] = self.st.anchor.get(L, 0) + d; self.st.anchor_n[L] = self.st.anchor_n.get(L, 0) + 1
                    v = d
                else:
                    v = self.st.anchor[L] / self.st.anchor_n[L]
                add = a.alpha * s * v if a.steer == 'anchor' else \
                    a.alpha * s * a.rho * h0.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            else:  # sel
                p = self.st.proc
                if p is None or p.R <= 0: return None
                B = p.Am if a.sel_belief == 'median' else p.A
                wts = (B / max(p.R, 1e-6)).clamp(-a.clip, a.clip) - (a.pen / a.lam if a.lam else 0.0)
                if a.sel_sign == 'neg': wts = wts.clamp(max=0.0)       # only push away from objects below the threshold
                wts = torch.where(p.done, torch.zeros_like(wts), wts)
                v = (wts[:, None] * self.U).sum(0)[None]
                if float(v.norm()) == 0.0: return None
                add = a.alpha * s * a.rho * h0.norm(dim=-1, keepdim=True) * v / v.norm()
            self.calls += 1
            h = h.clone(); h[0] = h[0] + add.to(h.dtype)               # vdd_decode.py: h[0] + (alpha s d).to(dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    def _attn_hook(self, L):
        def hook(mod, args, out):
            u = out[0] if isinstance(out, tuple) else out
            if not self._live(u): return None
            s = self.s(self.st.t)
            self.calls += 1
            u = u.clone(); u[0] = u[0] + (self.a.alpha * s * (u[0] - u[self.c]).float()).to(u.dtype)
            return (u,) + tuple(out[1:]) if isinstance(out, tuple) else u
        return hook


def install_pai(adapter, st, a, rf, layer_idx):
    """Raise the real row's attention to image keys by log(1 + beta s(t)) in the given layers (decode steps)."""
    def bias_mask(module, query, key, attention_mask):
        li = getattr(module, 'layer_idx', None)
        if not (st.on and query.shape[-2] == 1 and li in layer_idx and st.img_pos is not None): return attention_mask
        if getattr(module, 'is_sliding', False) or getattr(module, 'sliding_window', None): return attention_mask
        k = key.shape[-2]
        if int(st.img_pos.max()) >= k: return attention_mask
        s = 1 - rf(st.t) if a.ssched == 'decay' else 1.0
        bias = torch.zeros(query.shape[0], 1, 1, k, device=query.device, dtype=query.dtype)
        bias[0, 0, 0, st.img_pos] = math.log1p(a.beta * s)
        if attention_mask is None:
            m = bias
        else:
            m = attention_mask[..., -1:, :k]
            if m.dtype == torch.bool: m = torch.zeros_like(m, dtype=query.dtype).masked_fill(~m, torch.finfo(query.dtype).min)
            m = m.to(query.dtype) + bias
        st.pai_calls += 1
        return m
    from transformers import modeling_utils as mu
    AF = mu.ALL_ATTENTION_FUNCTIONS
    impl = adapter.layers[0].self_attn.config._attn_implementation
    if impl == 'eager':
        mod = sys.modules[type(adapter.layers[0].self_attn).__module__]
        orig = mod.eager_attention_forward
        def wrapped(module, query, key, value, attention_mask, *args, **kw):
            return orig(module, query, key, value, bias_mask(module, query, key, attention_mask), *args, **kw)
        mod.eager_attention_forward = wrapped
    else:
        orig = AF[impl]
        def wrapped(module, query, key, value, attention_mask, *args, **kw):
            return orig(module, query, key, value, bias_mask(module, query, key, attention_mask), *args, **kw)
        AF[impl] = wrapped


def blurred(path, size, radius):
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    out = Path(tempfile.gettempdir()) / f'vdd_blur_{radius}_{Path(path).stem}.png'
    im.filter(ImageFilter.GaussianBlur(radius)).save(out)
    return str(out)


def rows_for(adapter, ref, a):
    b = [adapter.prepare(ref['image'], ref['prompt']), adapter.prepare(str(GRAY), ref['prompt'])]
    if a.ctrl == 'blur':
        b.append(adapter.prepare(blurred(ref['image'], int(adapter.image_size or 448), a.blur_radius), ref['prompt']))
    for x in b[1:]:
        assert x.keys() == b[0].keys() and torch.equal(x['input_ids'], b[0]['input_ids'])
    return {k: torch.cat([x[k] for x in b], 0) for k in b[0] if k != 'labels'}


def main(a):
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') and torch.cuda.device_count() == 1
    torch.manual_seed(42)
    if not GRAY.exists(): GRAY.parent.mkdir(parents=True, exist_ok=True); Image.new('RGB', (448, 448), (128, 128, 128)).save(GRAY)
    config = dict(json.loads((C.MODELS[a.model] / 'config.json').read_text())['model'], attn_implementation=a.attn)
    refs = C.read(a.data)[a.start:a.end]
    name = f'{a.tag}-{a.start}-{a.end}'; rows_file = out / f'{name}.jsonl'; status = out / f'{name}-status.json'
    done = C.read(rows_file) if rows_file.exists() else []
    keys = ('steer', 'layers', 'alpha', 'rho', 'beta', 'anchor_steps', 'ctrl', 'blur_radius', 'sel_belief', 'sel_sign',
            'steer2', 'layers2', 'alpha2', 'rho2', 'lam', 'clip', 'pen', 'bsched', 'ssched', 'r', 'attn', 'bpos', 'norm')
    setting = {k: getattr(a, k) for k in keys}
    def update(**k):
        s = json.loads(status.read_text()) if status.exists() else {}
        s.update(k, updated_at=time.time()); status.write_text(json.dumps(s, indent=1))
    update(stage='loading', pid=os.getpid(), args={k: str(v) for k, v in vars(a).items()}, resumed_from=len(done))
    try:
        adapter = load_adapter(config); m = adapter.model; tok = adapter.processor.tokenizer
        vobj = json.loads((Path(a.vobj) if a.vobj else ROOT / f'outputs/rbd-20261030/collect/{a.model}-vobj2.json').read_text())
        vidx = torch.tensor(vobj, device=adapter.device); rf = r_curve(a.model, a.r); st = State()
        ctrl_row = 2 if a.ctrl == 'blur' else 1
        steer = Steer(adapter, st, a, rf, ctrl_row, vidx) if a.steer != 'none' else None
        if a.steer2 != 'none':   # a second steering, with its own layers / strength (e.g. resid at L16 + sel at L24)
            a2 = argparse.Namespace(**{**vars(a), 'steer': a.steer2, 'layers': a.layers2, 'alpha': a.alpha2, 'rho': a.rho2})
            steer2 = Steer(adapter, st, a2, rf, ctrl_row, vidx)
        else:
            steer2 = None
        img_id = getattr(m.config, 'image_token_id', None) or getattr(m.config, 'image_token_index', None)
        update(stage='running', vobj=len(vobj))
        with rows_file.open('a') as f:
            for i, ref in enumerate(refs[len(done):], start=len(done)):
                t0 = time.perf_counter(); batch = {k: v.to(adapter.device) for k, v in rows_for(adapter, ref, a).items()}
                pl = batch['input_ids'].shape[1]
                st.img_pos = (batch['input_ids'][0] == img_id).nonzero().flatten() if img_id is not None else None
                proc = Evidence(st, vidx, rf, a, pl); st.proc = proc; st.t = 0; st.anchor = {}; st.anchor_n = {}; st.on = True
                calls0, pai0 = (steer.calls if steer else 0), st.pai_calls
                calls2_0 = steer2.calls if steer2 else 0
                try:
                    with torch.no_grad():
                        o = m.generate(**batch, do_sample=False, num_beams=1, use_cache=True, max_new_tokens=int(ref['max_new_tokens']),
                                       logits_processor=LogitsProcessorList([proc]))
                finally:
                    st.on = False
                for r_ in range(1, o.shape[0]): assert torch.equal(o[0], o[r_])
                ids = o[0, pl:].tolist()
                if tok.pad_token_id in ids: ids = ids[:ids.index(tok.pad_token_id)]
                n_steer = (steer.calls - calls0) if steer else 0; n_pai = st.pai_calls - pai0
                n_steer2 = (steer2.calls - calls2_0) if steer2 else 0
                if a.steer in ('resid', 'norm', 'anchor', 'attn') and len(ids) > 2: assert n_steer > 0, 'steering hook never fired'
                if a.steer == 'pai' and len(ids) > 2: assert n_pai > 0, 'PAI bias never applied (attention function not wrapped)'
                row = {**ref, 'method': a.tag, 'prediction': tok.decode(ids, skip_special_tokens=True), 'generated_tokens': len(ids),
                       'elapsed_seconds': time.perf_counter() - t0, 'decoder': 'vdd_steer.py', 'setting': setting,
                       'steer_calls': n_steer, 'steer2_calls': n_steer2, 'pai_calls': n_pai}
                f.write(json.dumps(row) + '\n'); f.flush(); update(done=i + 1)
        update(stage='complete')
    except BaseException:
        update(stage='failed', error=traceback.format_exc()); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(C.MODELS), required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--tag', required=True)
    p.add_argument('--start', type=int, default=0); p.add_argument('--end', type=int, default=100)
    p.add_argument('--steer', choices=['resid', 'norm', 'anchor', 'anchornorm', 'attn', 'pai', 'sel', 'restore', 'none'], default='resid')
    p.add_argument('--sel-belief', choices=['raw', 'median'], default='raw'); p.add_argument('--sel-sign', choices=['both', 'neg'], default='both')
    p.add_argument('--steer2', choices=['resid', 'norm', 'anchor', 'anchornorm', 'sel', 'none'], default='none')
    p.add_argument('--layers2', type=lambda s: [int(x) for x in s.split(',')], default=[24])
    p.add_argument('--alpha2', type=float, default=1.0); p.add_argument('--rho2', type=float, default=0.2)
    p.add_argument('--layers', type=lambda s: [int(x) for x in s.split(',')], default=[16])
    p.add_argument('--alpha', type=float, default=2.0); p.add_argument('--rho', type=float, default=0.3)
    p.add_argument('--beta', type=float, default=1.0); p.add_argument('--anchor-steps', type=int, default=16)
    p.add_argument('--ctrl', choices=['gray', 'blur'], default='gray'); p.add_argument('--blur-radius', type=float, default=12.0)
    p.add_argument('--lam', type=float, default=1.0); p.add_argument('--clip', type=float, default=5.0)
    p.add_argument('--pen', type=float, default=0.5); p.add_argument('--bsched', choices=['decay', 'const'], default='decay')
    p.add_argument('--ssched', choices=['decay', 'const', 'early', 'gate'], default='decay')  # early: s(t) = r(t) (steering-hook variants only)
    p.add_argument('--r', choices=['measured', 'const', 'step0'], default='measured')
    p.add_argument('--attn', default='sdpa'); p.add_argument('--bpos', action='store_true')
    p.add_argument('--norm', choices=['raw', 'median'], default='raw'); p.add_argument('--vobj', default=None)
    main(p.parse_args())
