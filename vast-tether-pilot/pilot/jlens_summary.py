"""Summarize out/jlens/probe.npz: what the top 50 holds per layer, Jacobian lens against logit lens.

  recall@50      share of (image, object in the image) pairs whose object word is in the top 50
  false@50       objects not in the image whose word is in the top 50, per image
  AUROC          within object, images with it against images without it (score: -rank in the vocabulary, or the best
                 image-token logit centred on the image's other objects); equal weight per object
  pointing       the image token where the object's word scores highest falls on the object's COCO mask
Usage: python pilot/jlens_summary.py   Writes out/jlens/summary.json.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from typing import Any

import numpy as np

from jlens_probe import KINDS, OUT_DIR, PILOT_IMAGES, TOP_K
from stats import PairedBootstrap, StratifiedAuroc

MIN_PER_CLASS = 5
N_BOOT = 500


def within_object(scores: np.ndarray, present: np.ndarray) -> StratifiedAuroc:
    """scores, present: [images, objects]."""
    n_images, n_objects = present.shape
    objects = np.tile(np.arange(n_objects), n_images)
    images = np.repeat(np.arange(n_images), n_objects)
    ok = (present.sum(0) >= MIN_PER_CLASS) & ((~present).sum(0) >= MIN_PER_CLASS)
    keep = ok[objects]
    return StratifiedAuroc(scores.ravel()[keep], present.ravel()[keep], objects[keep], images[keep], equal_strata=True)


def with_interval(estimator: StratifiedAuroc, boot: PairedBootstrap) -> dict[str, float]:
    low, high = boot.ci(estimator)
    return {"auroc": estimator.value(), "low": low, "high": high}


def auroc_tables(data: dict[str, np.ndarray], present: np.ndarray, layers: list[int], boot: PairedBootstrap) -> dict[str, Any]:
    span_centred = data["span_max"] - np.median(data["span_max"], axis=-1, keepdims=True)
    scores = {"answer": lambda i, k: -data["rank"][:, i, k, 0].astype(float),
              "post_image": lambda i, k: -data["rank"][:, i, k, 1].astype(float),
              "image_tokens": lambda i, k: span_centred[:, i, k]}
    table: dict[str, Any] = {}
    for place, score in scores.items():
        rows = []
        for i, layer in enumerate(layers):
            jac, logit = within_object(score(i, 0), present), within_object(score(i, 1), present)
            low, high = boot.difference_ci(jac, logit)
            rows.append({"layer": layer, "jacobian": with_interval(jac, boot), "logit": with_interval(logit, boot),
                         "jacobian_minus_logit": {"difference": jac.value() - logit.value(), "low": low, "high": high}})
        table[place] = rows
    final = within_object(-data["final_rank"].astype(float), present)
    table["model_output_answer"] = with_interval(final, boot)
    table["n_objects_scored"] = int(len(within_object(-data["final_rank"].astype(float), present).strata))
    return table


def recall_tables(data: dict[str, np.ndarray], present: np.ndarray, layers: list[int]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for p, place in enumerate(("answer", "post_image")):
        rows = []
        for i, layer in enumerate(layers):
            entry: dict[str, Any] = {"layer": layer}
            for k, kind in enumerate(KINDS):
                hit = data["rank"][:, i, k, p] < TOP_K
                gray_hit = data["gray_rank"][i, k, p] < TOP_K
                entry[kind] = {"recall": float(hit[present].mean()), "false_per_image": float((hit & ~present).sum(1).mean()),
                               "precision": float(hit[present].sum() / max(hit.sum(), 1)),
                               "recall_not_in_gray_top": float((hit & ~gray_hit[None])[present].mean()),
                               "objects_in_gray_top": int(gray_hit.sum())}
            rows.append(entry)
        table[place] = rows
    final_hit = data["final_rank"] < TOP_K
    table["model_output_answer"] = {"recall": float(final_hit[present].mean()), "false_per_image": float((final_hit & ~present).sum(1).mean())}
    return table


def first_layers(data: dict[str, np.ndarray], present: np.ndarray, layers: list[int]) -> dict[str, Any]:
    """For objects in the image: the first layer where their word reaches the top 50 at the answer position."""
    out = {}
    for k, kind in enumerate(KINDS):
        hit = data["rank"][:, :, k, 0] < TOP_K  # [images, layers, objects]
        ever = hit.any(1)
        first = np.where(ever, hit.argmax(1), -1)
        chosen = first[present & ever]
        out[kind] = {"share_ever_in_top": float(ever[present].mean()),
                     "first_layer_quartiles": [int(layers[int(q)]) for q in np.percentile(chosen, [25, 50, 75])] if len(chosen) else None,
                     "first_layer_histogram": {int(layers[j]): int((chosen == j).sum()) for j in range(len(layers))}}
    return out


def pointing_table(data: dict[str, np.ndarray], present: np.ndarray, nodes: list[str], rows: list[dict[str, Any]],
                   layers: list[int]) -> dict[str, Any]:
    masks = pickle.loads((PILOT_IMAGES / "object_masks.pkl").read_bytes())
    pairs = [(n, j, masks[r["image_id"]][node].ravel()) for n, r in enumerate(rows) for j, node in enumerate(nodes)
             if present[n, j] and node in masks.get(r["image_id"], {}) and masks[r["image_id"]][node].mean() >= 0.01]
    chance = float(np.mean([(cover >= 0.5).mean() for _, _, cover in pairs]))
    table = {"n_pairs": len(pairs), "chance": chance, "by_layer": []}
    for i, layer in enumerate(layers):
        entry = {"layer": layer}
        for k, kind in enumerate(KINDS):
            entry[kind] = float(np.mean([cover[data["span_argmax"][n, i, k, j]] >= 0.5 for n, j, cover in pairs]))
        table["by_layer"].append(entry)
    return table


def content_table(data: dict[str, np.ndarray], vocab: dict[str, str], object_tokens: set[int], layers: list[int]) -> dict[str, Any]:
    """What fills the top 50 at the answer position: frequent tokens, object words, and tokens the gray image lacks."""
    table = []
    for i, layer in enumerate(layers):
        entry: dict[str, Any] = {"layer": layer}
        for k, kind in enumerate(KINDS):
            tops = data["top_ids"][:, i, k, 0]
            gray = set(data["gray_top_ids"][i, k, 0].tolist())
            counts = Counter(tops.ravel().tolist())
            entry[kind] = {"frequent": [[vocab[str(t)], round(c / len(tops), 3)] for t, c in counts.most_common(15)],
                           "gray_top": [vocab[str(t)] for t in data["gray_top_ids"][i, k, 0][:15]],
                           "object_word_share": float(np.isin(tops, list(object_tokens)).mean()),
                           "not_in_gray_share": float(np.mean([[t not in gray for t in row] for row in tops])),
                           "distinct_tokens": len(counts)}
        table.append(entry)
    return table


def examples(data: dict[str, np.ndarray], meta: dict[str, Any], present: np.ndarray, n_examples: int = 4) -> list[dict[str, Any]]:
    vocab, nodes, layers = meta["vocab"], meta["nodes"], meta["layers"]
    owner = {t: node for node, ids in meta["node_ids"].items() for t in ids}
    picked = [n for n in range(len(meta["rows"])) if present[n].sum() >= 3][:n_examples]
    shown_layers = sorted({layers[j] for j in range(3, len(layers), 4)} | {layers[-1]})
    out = []
    for n in picked:
        row = meta["rows"][n]
        entry = {"image_id": row["image_id"], "objects": [nodes[j] for j in np.flatnonzero(present[n])], "layers": {}}
        for layer in shown_layers:
            i = layers.index(layer)
            entry["layers"][layer] = {kind: [[vocab[str(t)], owner.get(int(t)), owner.get(int(t)) in entry["objects"] if owner.get(int(t)) else None]
                                             for t in data["top_ids"][n, i, k, 0]] for k, kind in enumerate(KINDS)}
        entry["model_output"] = [vocab[str(t)] for t in data["final_top_ids"][n][:15]]
        out.append(entry)
    return out


def main() -> None:
    with np.load(OUT_DIR / "probe.npz") as loaded:
        data = {k: loaded[k] for k in loaded.files}
    meta = json.loads((OUT_DIR / "meta.json").read_text())
    nodes, rows, layers = meta["nodes"], meta["rows"], meta["layers"]
    present = np.array([[node in set(r["ground_truth"]) for node in nodes] for r in rows])
    boot = PairedBootstrap(len(rows), N_BOOT)
    object_tokens = {t for ids in meta["node_ids"].values() for t in ids}
    summary = {"lens": meta["lens"], "n_images": len(rows), "n_objects_with_tokens": len(nodes), "layers": layers,
               "n_model_layers": meta["n_model_layers"], "present_pairs": int(present.sum()),
               "recall": recall_tables(data, present, layers), "first_layer": first_layers(data, present, layers),
               "auroc": auroc_tables(data, present, layers, boot), "pointing": pointing_table(data, present, nodes, rows, layers),
               "content": content_table(data, meta["vocab"], object_tokens, layers), "examples": examples(data, meta, present)}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("SUMMARY_OK")


if __name__ == "__main__":
    main()
