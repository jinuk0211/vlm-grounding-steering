"""Export out/jlens_pos100 (top-100 Jacobian / logit lens, every position, 31 layers) to a static, serverless viewer bundle.

Streams a tar to stdout (nothing is written on the box) with:
  data/<id>.js   JL_DATA(<id>, "<base64 of gzip(binary)>")  binary sections, little-endian, in this order:
                   ids   3 byte planes (lo, mid, hi) of the token ids [P, 31 layers, 2 lenses, 100]  (lens 0 = Jacobian, 1 = logit)
                   top1  float32 [P, 31, 2]          logit of rank 1
                   gap   uint8   [P, 31, 2, 100]   (top1 - logit) * 10, clipped to 255 (0.1-logit steps)
                   nrank int32   [P, 31, 2]          0-based rank of the token actually written at p + 1
                   fids  3 byte planes [P, 100]      model's own top 100 at p (final layer + lm_head)
                   frank int32   [P]                 rank of the written token in the model's own distribution
  index.js       JL_INDEX({...})  per image: ids, caption, n_prompt, image_span, n_caption, stopped, mentions, ground_truth
  vocab.js       JL_VOCAB({...})  token text, kind (out/jlens_view100/vocab.json)
  img/<id>.jpg   the 448x448 model input (out/jlens_viz/img)
Usage: nice -n 10 .venv35/bin/python jl_export.py [--only ID] [--level 9]
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/tether-pilot")
POS, TOP = ROOT / "out" / "jlens_pos", ROOT / "out" / "jlens_pos100"
VOCAB, IMG = ROOT / "out" / "jlens_view100" / "vocab.json", ROOT / "out" / "jlens_viz" / "img"
OUT = ROOT / "out" / "jlens_local"
INFO_KEYS = ("image_id", "ground_truth", "ids", "caption", "n_prompt", "image_span", "n_caption", "stopped", "mentions")


def planes(ids: np.ndarray) -> bytes:
    flat = ids.astype(np.uint32).ravel()
    assert flat.max() < (1 << 24)
    return b"".join(((flat >> s) & 0xFF).astype(np.uint8).tobytes() for s in (0, 8, 16))


def encode(npz: Path, level: int) -> tuple[bytes, dict]:
    with np.load(npz) as z:
        ids, val, nrank = z["top_ids"], z["top_val"].astype(np.float32), z["next_rank"]
        fids, frank = z["final_top_ids"], z["final_rank"]
    P, L, K, N = ids.shape
    assert (K, N) == (2, 100), ids.shape
    top1 = val[..., 0]
    gap = np.clip(np.round((top1[..., None] - val) * 10), 0, 255).astype(np.uint8)
    parts = [planes(ids), top1.astype("<f4").tobytes(), gap.tobytes(), nrank.astype("<i4").tobytes(), planes(fids),
             frank.astype("<i4").tobytes()]
    raw = b"".join(parts)
    return gzip.compress(raw, compresslevel=level, mtime=0), {"P": P, "L": L, "raw": len(raw)}


def main() -> None:
    import io
    import sys
    import tarfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int)
    ap.add_argument("--level", type=int, default=6)
    args = ap.parse_args()
    tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")

    def emit(name: str, data: bytes) -> None:
        entry = tarfile.TarInfo(name)
        entry.size, entry.mtime = len(data), int(time.time())
        tar.addfile(entry, io.BytesIO(data))

    objects = json.loads((TOP / "objects.json").read_text())
    files = sorted(TOP.glob("*.npz"), key=lambda p: int(p.stem))
    if args.only:
        files = [f for f in files if int(f.stem) == args.only]
    images, t0, total = [], time.time(), 0
    for i, f in enumerate(files):
        info = json.loads((POS / f"{f.stem}.json").read_text())
        blob, meta = encode(f, args.level)
        assert meta["P"] == len(info["ids"]), (f, meta, len(info["ids"]))
        text = f'JL_DATA({int(f.stem)},"{base64.b64encode(blob).decode()}");\n'.encode()
        emit(f"data/{f.stem}.js", text)
        total += len(text)
        src = IMG / f"{f.stem}.jpg"
        if src.exists():
            emit(f"img/{src.name}", src.read_bytes())
        images.append({**{k: info[k] for k in INFO_KEYS}, "has_img": src.exists(), "bytes": len(text)})
        print(f"{i + 1}/{len(files)} {f.stem} P={meta['P']} raw {meta['raw'] / 1e6:.1f} MB gzip {len(blob) / 1e6:.1f} MB "
              f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    index = {"layers": objects["layers"], "lens": objects["lens"], "topn": objects["topn"], "nodes": objects["nodes"],
             "node_ids": objects["node_ids"], "eos": objects["eos"], "images": images,
             "note": "top_ids at position p are the lens readouts of the residual at p, i.e. predictions for the token at p + 1"}
    emit("index.js", ("JL_INDEX(" + json.dumps(index, ensure_ascii=False, separators=(",", ":")) + ");\n").encode())
    emit("vocab.js", ("JL_VOCAB(" + VOCAB.read_text() + ");\n").encode())
    if (IMG / "gray.jpg").exists():
        emit("img/gray.jpg", (IMG / "gray.jpg").read_bytes())
    tar.close()
    print(f"EXPORT_DONE {len(files)} images, data {total / 1e6:.0f} MB", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
