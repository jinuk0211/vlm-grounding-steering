"""Top-100 rerun of jlens_positions.py on the captions it already wrote (no generation).

For every image in out/jlens_pos/ (its prompt + greedy caption token ids), one forward pass with the residual stream recorded
at the lens's 31 layers; per position, layer and lens (Jacobian, logit) the top N tokens and their logits, with the same
readout code as jlens_positions.py (only TOPN changes). Checks that the prompt rebuilds to the same ids and reports how
often the new top 10 matches the old one.
Usage: PILOT_MODEL=qwen35 .venv35/bin/python pilot/jlens_positions_top100.py --model-dir DIR --lens FILE [--topn 100]
Writes out/jlens_pos100/<image_id>.npz (top_ids, top_val, next_rank, final_top_ids, final_rank) and objects.json; images
already written are skipped.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import jlens_positions as jp
from common import CHAIR_STATE, COCO, ROOT, SPEC, configure_generation, fast_patch_embedding, load_chair, load_image, read_jsonl
from jlens_probe import PILOT_IMAGES, model_inputs, object_token_ids

SRC = ROOT / "out" / "jlens_pos"
KEEP = ("top_ids", "top_val", "next_rank", "final_top_ids", "final_rank")


def main() -> None:
    import jlens
    import transformers
    from jlens import JacobianLens

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--topn", type=int, default=100)
    parser.add_argument("--out-dir", default=str(ROOT / "out" / "jlens_pos100"))
    args = parser.parse_args()
    jp.TOPN = args.topn
    out_dir = jp.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = transformers.AutoProcessor.from_pretrained(args.model_dir)
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    fast_patch_embedding(model)
    configure_generation(model, processor, SPEC)
    lm = jlens.from_hf(model, processor.tokenizer)
    lens = JacobianLens.load(args.lens)
    jacobians = {layer: J.to("cuda") for layer, J in lens.jacobians.items()}
    scorer = load_chair(COCO / "annotations", CHAIR_STATE)
    node_ids = object_token_ids(processor.tokenizer, scorer.inverse_synonym_dict)
    old_objects = json.loads((SRC / "objects.json").read_text())
    assert list(node_ids) == old_objects["nodes"], "object list differs from out/jlens_pos/objects.json"
    reader = jp.AllPositions(model, lm, jacobians, node_ids)
    rows = {r["image_id"]: r for r in read_jsonl(PILOT_IMAGES / "manifest.jsonl")}
    infos = sorted(p for p in SRC.glob("*.json") if p.name != "objects.json")
    (out_dir / "objects.json").write_text(json.dumps({**old_objects, "topn": args.topn, "source": str(SRC)}))
    print(f"{lens!r}; {len(infos)} images; top {args.topn}; out {out_dir}", flush=True)
    began = time.perf_counter()
    for index, path in enumerate(infos):
        info = json.loads(path.read_text())
        image_id = info["image_id"]
        stem = out_dir / str(image_id)
        if stem.with_suffix(".npz").exists():
            continue
        inputs = model_inputs(processor, load_image(rows[image_id]["path"]), "cuda")
        prompt = inputs["input_ids"][0].tolist()
        if len(prompt) != info["n_prompt"] or prompt != info["ids"][:info["n_prompt"]]:
            print(f"{image_id}: prompt ids differ from out/jlens_pos, skipped", flush=True)
            continue
        full_ids = torch.tensor(info["ids"], device="cuda")
        t0 = time.perf_counter()
        data = reader(inputs, full_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with np.load(SRC / f"{image_id}.npz") as old:
            agree = float((data["top_ids"][..., :old["top_ids"].shape[-1]] == old["top_ids"]).mean())
        np.savez_compressed(stem.with_suffix(".npz"), **{k: data[k] for k in KEEP})
        print(f"{index + 1}/{len(infos)} {image_id} seq {len(info['ids'])} lens {t1 - t0:.0f}s top10 agree {agree:.4f} "
              f"total {time.perf_counter() - began:.0f}s", flush=True)
    print("TOP100_OK", flush=True)


if __name__ == "__main__":
    main()
