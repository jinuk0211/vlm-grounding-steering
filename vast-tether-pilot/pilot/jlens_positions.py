"""Jacobian lens and logit lens at EVERY token position: chat template, image tokens and Qwen3.5-4B's own greedy caption.

Per image: generate the caption (greedy, 512 new tokens, thinking off, stop at <|im_end|>), run prompt + caption once with
the residual stream recorded at the lens's 31 layers, and keep for every position, layer and lens:
  top_ids/top_val  top 10 tokens and their logits
  next_rank        rank of the token that actually comes next (0 = the lens's top token; -1 at the last position)
  obj_rank         rank of each COCO object's best single-token word (62 objects, capped at 65535)
plus the model's own output (final_top_ids, final_rank). Object mentions of the caption are labelled with CHAIR's parser
and the image's COCO objects (hallucinated = object not in the image).
Usage: PILOT_MODEL=qwen35 .venv35/bin/python pilot/jlens_positions.py --model-dir DIR --lens FILE [--n-images 100] [--out-dir DIR]
Writes <out-dir>/<image_id>.npz and .json; images already written are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import CHAIR_STATE, COCO, MAX_NEW_TOKENS, ROOT, SPEC, configure_generation, fast_patch_embedding, load_chair, \
    load_image, mention_token_positions, read_jsonl
from jlens_probe import PILOT_IMAGES, model_inputs, object_token_ids

TOPN = 10
CHUNK = 192


def full_batch(inputs: Any, full_ids: torch.Tensor) -> dict[str, Any]:
    """The processor's inputs for prompt + caption: per-token tensors are extended with zeros for the caption tokens."""
    n_prompt = inputs["input_ids"].shape[1]
    extra = full_ids.shape[0] - n_prompt
    batch = {}
    for key, value in inputs.items():
        if key == "input_ids":
            batch[key] = full_ids[None]
        elif key == "attention_mask":
            batch[key] = torch.ones((1, full_ids.shape[0]), dtype=value.dtype, device=value.device)
        elif torch.is_tensor(value) and value.dim() == 2 and value.shape == (1, n_prompt):
            batch[key] = torch.cat([value, torch.zeros((1, extra), dtype=value.dtype, device=value.device)], 1)
        else:
            batch[key] = value
    return batch


def ranks(logits: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Number of vocabulary logits strictly above each query value, per row. logits [n, V], queries [n, m]."""
    ordered = logits.sort(-1).values
    return logits.shape[-1] - torch.searchsorted(ordered, queries.contiguous(), right=True)


class AllPositions:
    def __init__(self, model: Any, lm: Any, jacobians: dict[int, torch.Tensor], node_ids: dict[str, list[int]]):
        self.model, self.lm, self.jacobians = model, lm, jacobians
        self.layers = sorted(jacobians)
        device = model.lm_head.weight.device
        self.flat_ids = torch.tensor([t for ids in node_ids.values() for t in ids], device=device)
        self.owner = torch.tensor([j for j, ids in enumerate(node_ids.values()) for _ in ids], device=device)
        self.n_nodes = len(node_ids)

    def node_scores(self, logits: torch.Tensor) -> torch.Tensor:
        out = torch.full((logits.shape[0], self.n_nodes), -torch.inf, device=logits.device)
        return out.scatter_reduce(1, self.owner.expand(logits.shape[0], -1), logits[:, self.flat_ids], reduce="amax", include_self=True)

    def readout(self, logits: torch.Tensor, nxt: torch.Tensor) -> tuple[torch.Tensor, ...]:
        top = logits.topk(TOPN, dim=-1)
        queries = torch.cat([logits.gather(1, nxt.clamp(min=0)[:, None]), self.node_scores(logits)], 1)
        r = ranks(logits, queries)
        next_rank = torch.where(nxt >= 0, r[:, 0], torch.full_like(r[:, 0], -1))
        return top.indices, top.values, next_rank, r[:, 1:].clamp(max=65535)

    @torch.no_grad()
    def __call__(self, inputs: Any, full_ids: torch.Tensor) -> dict[str, np.ndarray]:
        from jlens import ActivationRecorder

        with ActivationRecorder(self.lm.layers, at=self.layers) as recorder:
            output = self.model(**full_batch(inputs, full_ids), use_cache=False)
            states = {l: recorder.activations[l][0].detach() for l in self.layers}
        seq, n_layers = full_ids.shape[0], len(self.layers)
        nxt = torch.cat([full_ids[1:], full_ids.new_full((1,), -1)])
        out = {"top_ids": np.zeros((seq, n_layers, 2, TOPN), np.int32), "top_val": np.zeros((seq, n_layers, 2, TOPN), np.float16),
               "next_rank": np.zeros((seq, n_layers, 2), np.int32), "obj_rank": np.zeros((seq, n_layers, 2, self.n_nodes), np.uint16),
               "final_top_ids": np.zeros((seq, TOPN), np.int32), "final_rank": np.zeros(seq, np.int32),
               "final_obj_rank": np.zeros((seq, self.n_nodes), np.uint16)}
        for s in range(0, seq, CHUNK):
            e = min(seq, s + CHUNK)
            for i, layer in enumerate(self.layers):
                residual = states[layer][s:e].float()
                for k in range(2):
                    transported = residual @ self.jacobians[layer].T if k == 0 else residual
                    ids, vals, nr, orank = self.readout(self.lm.unembed(transported).float(), nxt[s:e])
                    out["top_ids"][s:e, i, k] = ids.cpu().numpy()
                    out["top_val"][s:e, i, k] = vals.cpu().numpy()
                    out["next_rank"][s:e, i, k] = nr.cpu().numpy()
                    out["obj_rank"][s:e, i, k] = orank.cpu().numpy()
            ids, _, nr, orank = self.readout(output.logits[0, s:e].float(), nxt[s:e])
            out["final_top_ids"][s:e], out["final_rank"][s:e], out["final_obj_rank"][s:e] = ids.cpu().numpy(), nr.cpu().numpy(), orank.cpu().numpy()
        return out


def mentions(tokenizer: Any, caption: list[int], scorer: Any, ground_truth: set[str], offset: int) -> list[dict[str, Any]]:
    try:
        found = mention_token_positions(tokenizer, caption, scorer)
    except Exception as error:  # the labels are an extra; the lens readout does not depend on them
        print(f"mention parsing failed: {error!r}", flush=True)
        return []
    return [{"node": node, "word": word, "pos": offset + t, "hallucinated": node not in ground_truth} for node, word, t in found]


def main() -> None:
    import jlens
    import transformers
    from jlens import JacobianLens

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--out-dir", default=str(ROOT / "out" / "jlens_pos"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
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
    reader = AllPositions(model, lm, jacobians, node_ids)
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
    rows = [r for r in read_jsonl(PILOT_IMAGES / "manifest.jsonl") if r["with_caption"]][:args.n_images]
    (out_dir / "objects.json").write_text(json.dumps({"nodes": list(node_ids), "node_ids": node_ids, "layers": reader.layers, "lens": repr(lens),
                                                       "eos": sorted(eos), "max_new_tokens": MAX_NEW_TOKENS}))
    print(f"{lens!r}; {len(rows)} images; eos {sorted(eos)}; out {out_dir}", flush=True)
    began = time.perf_counter()
    for index, row in enumerate(rows):
        stem = out_dir / str(row["image_id"])
        if stem.with_suffix(".json").exists():
            continue
        inputs = model_inputs(processor, load_image(row["path"]), "cuda")
        n_prompt = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        full_ids = generated[0]
        t1 = time.perf_counter()
        data = reader(inputs, full_ids)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ids = full_ids.tolist()
        caption = [t for t in ids[n_prompt:] if t not in eos]
        image_positions = [i for i, t in enumerate(ids) if t == model.config.image_token_id]
        np.savez_compressed(stem.with_suffix(".npz"), **data)
        info = {"image_id": row["image_id"], "ground_truth": row["ground_truth"], "ids": ids, "n_prompt": n_prompt,
                "image_span": [image_positions[0], image_positions[-1] + 1], "n_caption": len(ids) - n_prompt,
                "stopped": ids[-1] in eos, "caption": processor.tokenizer.decode(caption, skip_special_tokens=True),
                "mentions": mentions(processor.tokenizer, caption, scorer, set(row["ground_truth"]), n_prompt),
                "seconds": {"generate": round(t1 - t0, 1), "lens": round(t2 - t1, 1)}}
        stem.with_suffix(".json").write_text(json.dumps(info))
        print(f"{index + 1}/{len(rows)} {row['image_id']} seq {len(ids)} caption {info['n_caption']} gen {t1 - t0:.0f}s lens {t2 - t1:.0f}s "
              f"total {time.perf_counter() - began:.0f}s", flush=True)
    print("POSITIONS_OK", flush=True)


if __name__ == "__main__":
    main()
