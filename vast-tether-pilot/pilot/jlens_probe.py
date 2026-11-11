"""Which words can be read from Qwen3.5-4B's hidden states while it starts to describe an image?

Jacobian lens (Anthropic's `jlens` engine with JLensVL's fitted Qwen3.5-4B lens): unembed(J_l h). Logit lens: unembed(h).
Both read the top 50 tokens at every fitted layer, at three places: the position that writes the first caption token
("answer"), the first text position after the image ("post_image"), and each of the 196 image tokens. The gray image is the
language-only control. Labels: CHAIR's COCO objects on the pilot's 300 caption images (COCO train2014), object masks for
pointing. Prompt and preprocessing as in the pilot; thinking off.
Usage: .venv35/bin/python pilot/jlens_probe.py --model-dir DIR --lens FILE [--n-images 300]. Writes out/jlens/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from typing import Any

import numpy as np
import torch

from common import CHAIR_STATE, COCO, PROMPT, ROOT, fast_patch_embedding, gray_image, load_chair, load_image, read_jsonl

TOP_K = 50
SPAN_TOP = 20  # top tokens kept per image token
KINDS = ("jacobian", "logit")
PLACES = ("answer", "post_image")
OUT_DIR = ROOT / "out" / "jlens"
PILOT_IMAGES = ROOT / "out" / "qwen3"


def object_token_ids(tokenizer: Any, synonyms: dict[str, str]) -> dict[str, list[int]]:
    """Single-token ids per COCO object: each one-word CHAIR synonym and its plural, with and without a leading space and
    capitalized. A token claimed by two objects is dropped from both."""
    from pattern.en import pluralize

    by_node: dict[str, set[int]] = {}
    for word, node in synonyms.items():
        if not node or " " in word:
            continue
        for form in {word, pluralize(word)}:
            for variant in (form, " " + form, form.capitalize(), " " + form.capitalize()):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    by_node.setdefault(node, set()).add(ids[0])
    owners: dict[int, set[str]] = {}
    for node, ids in by_node.items():
        for token in ids:
            owners.setdefault(token, set()).add(node)
    kept = {node: sorted(t for t in ids if len(owners[t]) == 1) for node, ids in by_node.items()}
    return {node: ids for node, ids in sorted(kept.items()) if ids}


def model_inputs(processor: Any, image: Any, device: Any) -> Any:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return processor(text=[text], images=[image], return_tensors="pt").to(device)


class Readout:
    """Lens readouts for one model and lens. `jacobians` maps layer -> J_l on the model's device (float32)."""

    def __init__(self, model: Any, lm: Any, jacobians: dict[int, torch.Tensor], node_ids: dict[str, list[int]], image_token_id: int):
        self.model, self.lm, self.jacobians = model, lm, jacobians
        self.layers = sorted(jacobians)
        self.nodes = list(node_ids)
        device = model.lm_head.weight.device
        self.flat_ids = torch.tensor([t for ids in node_ids.values() for t in ids], device=device)
        self.owner = torch.tensor([j for j, ids in enumerate(node_ids.values()) for _ in ids], device=device)
        self.image_token_id = image_token_id

    def node_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Best logit over each object's tokens: [positions, objects]."""
        scores = logits[:, self.flat_ids]
        out = torch.full((logits.shape[0], len(self.nodes)), -torch.inf, device=logits.device)
        return out.scatter_reduce(1, self.owner.expand(logits.shape[0], -1), scores, reduce="amax", include_self=True)

    @torch.no_grad()
    def __call__(self, inputs: Any) -> dict[str, np.ndarray]:
        from jlens import ActivationRecorder

        ids = inputs["input_ids"][0]
        image_positions = (ids == self.image_token_id).nonzero().flatten()
        answer, post = len(ids) - 1, int(image_positions[-1]) + 1
        with ActivationRecorder(self.lm.layers, at=self.layers) as recorder:
            output = self.model(**inputs, use_cache=False)
            states = {l: recorder.activations[l][0].detach() for l in self.layers}
        final = output.logits[0, answer].float()
        places = torch.cat([torch.tensor([answer, post], device=ids.device), image_positions])
        n_layers, n_nodes, n_image = len(self.layers), len(self.nodes), len(image_positions)
        out = {"top_ids": np.zeros((n_layers, 2, 2, TOP_K), np.int32), "top_values": np.zeros((n_layers, 2, 2, TOP_K), np.float16),
               "rank": np.zeros((n_layers, 2, 2, n_nodes), np.int32), "score": np.zeros((n_layers, 2, 2, n_nodes), np.float32),
               "span_max": np.zeros((n_layers, 2, n_nodes), np.float32), "span_hits": np.zeros((n_layers, 2, n_nodes), np.int16),
               "span_argmax": np.zeros((n_layers, 2, n_nodes), np.int16), "span_top_ids": np.zeros((n_layers, 2, n_image, SPAN_TOP), np.int32)}
        for i, layer in enumerate(self.layers):
            residual = states[layer][places].float()
            for k, kind in enumerate(KINDS):
                transported = residual @ self.jacobians[layer].T if kind == "jacobian" else residual
                logits = self.lm.unembed(transported).float()
                top = logits.topk(TOP_K, dim=-1)
                scores = self.node_scores(logits)
                out["top_ids"][i, k] = top.indices[:2].cpu().numpy()
                out["top_values"][i, k] = top.values[:2].cpu().numpy()
                out["score"][i, k] = scores[:2].cpu().numpy()
                out["rank"][i, k] = (logits[:2, None, :] > scores[:2, :, None]).sum(-1).cpu().numpy()
                image_scores = scores[2:]
                threshold = top.values[2:, -1:]
                out["span_max"][i, k] = image_scores.max(0).values.cpu().numpy()
                out["span_argmax"][i, k] = image_scores.argmax(0).cpu().numpy()
                out["span_hits"][i, k] = (image_scores >= threshold).sum(0).cpu().numpy()
                out["span_top_ids"][i, k] = top.indices[2:, :SPAN_TOP].cpu().numpy()
        final_scores = self.node_scores(final[None])[0]
        out["final_top_ids"] = final.topk(TOP_K).indices.cpu().numpy().astype(np.int32)
        out["final_rank"] = (final[None, :] > final_scores[:, None]).sum(-1).cpu().numpy().astype(np.int32)
        out["final_score"] = final_scores.cpu().numpy()
        return out


def collect(readout: Readout, processor: Any, rows: list[dict[str, Any]], device: Any) -> dict[str, np.ndarray]:
    per_image = []
    began = time.perf_counter()
    for index, row in enumerate(rows):
        per_image.append(readout(model_inputs(processor, load_image(row["path"]), device)))
        if index % 25 == 0:
            print(f"{index}/{len(rows)} {time.perf_counter() - began:.0f}s", flush=True)
    gray = readout(model_inputs(processor, gray_image(), device))
    stacked = {key: np.stack([p[key] for p in per_image]) for key in per_image[0]}
    stacked.update({f"gray_{key}": value for key, value in gray.items()})
    return stacked


def main() -> None:
    import jlens
    import transformers
    from jlens import JacobianLens

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--n-images", type=int, default=300)
    args = parser.parse_args()

    processor = transformers.AutoProcessor.from_pretrained(args.model_dir)
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    fast_patch_embedding(model)
    lm = jlens.from_hf(model, processor.tokenizer)
    lens = JacobianLens.load(args.lens)  # weights_only=True
    jacobians = {layer: J.to("cuda") for layer, J in lens.jacobians.items()}
    scorer = load_chair(COCO / "annotations", CHAIR_STATE)
    node_ids = object_token_ids(processor.tokenizer, scorer.inverse_synonym_dict)
    print(f"{lens!r}; model layers {lm.n_layers}; objects with tokens {len(node_ids)}", flush=True)
    rows = [r for r in read_jsonl(PILOT_IMAGES / "manifest.jsonl") if r["with_caption"]][:args.n_images]
    readout = Readout(model, lm, jacobians, node_ids, model.config.image_token_id)
    data = collect(readout, processor, rows, "cuda")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "probe.npz", **data, image_ids=np.array([r["image_id"] for r in rows]),
                        layers=np.array(readout.layers))
    vocab = {int(t): processor.tokenizer.decode([int(t)]) for t in np.unique(np.concatenate(
        [data["top_ids"].ravel(), data["gray_top_ids"].ravel(), data["final_top_ids"].ravel(), data["span_top_ids"].ravel()]))}
    meta = {"nodes": readout.nodes, "node_ids": node_ids, "layers": readout.layers, "n_model_layers": lm.n_layers,
            "lens": repr(lens), "vocab": vocab, "rows": rows}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta))
    print("PROBE_OK", flush=True)


if __name__ == "__main__":
    main()
