"""Paper-derived, local baseline implementations (not author reproductions).

See docs/baseline_implementation.md for equations, extraction requirements, and
architecture adaptations. Hooks are temporary and never modify model weights.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import random
from typing import Any, Iterator

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from torch.nn import functional as F


PAPERS = {
    "revis": "https://arxiv.org/html/2602.11824v2",
    "rudder": "https://arxiv.org/html/2511.10292v2",
    "vti": "https://arxiv.org/html/2410.15778v2",
    "grains": "https://arxiv.org/html/2507.18043v2",
}
EPS = 1e-8


def _tensor(output: Any) -> Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, Tensor):
        raise TypeError("A hook expected a tensor or a tuple whose first item is a tensor.")
    return value


def _replace(output: Any, value: Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    return value


def _input(args: tuple, kwargs: dict) -> Tensor:
    value = args[0] if args else kwargs.get("hidden_states")
    if not isinstance(value, Tensor):
        raise TypeError("Decoder block must receive hidden_states as first argument or keyword.")
    return value


def _revis_pc1(samples: Tensor) -> Tensor:
    """REVIS's centered PC1, scaled by the mean input-vector norm."""
    samples = samples.detach().float().cpu()
    if len(samples) < 2:
        return samples.mean(0)
    if torch.allclose(samples.std(0), torch.zeros_like(samples[0]), atol=1e-6):
        return torch.zeros_like(samples[0])
    component = PCA(n_components=1).fit(samples.numpy()).components_[0]
    return torch.from_numpy(component) * samples.norm(dim=-1).mean()


def _vti_pc1_plus_mean(samples: Tensor) -> Tensor:
    """VTI's PC1 plus sample mean, with its batched SVD sign convention."""
    if samples.ndim == 3 and samples.shape[0] > 16:
        # Each leading position is an independent PCA in the released code.
        # Bound SVD workspace while preserving its layer/token reshape order.
        return torch.cat([_vti_pc1_plus_mean(chunk) for chunk in samples.split(16)], dim=0)
    samples = samples.detach().float().cpu()
    centered = samples - samples.mean(dim=-2, keepdim=True)
    u, _, vh = torch.linalg.svd(centered, full_matrices=False)
    pivot = u[..., 0].abs().argmax(dim=-1, keepdim=True)
    sign = u[..., 0].gather(-1, pivot).squeeze(-1).sign()
    return vh[..., 0, :] * sign.unsqueeze(-1) + samples.mean(dim=-2)


def _vti_update(hidden: Tensor, vector: Tensor, alpha: float) -> Tensor:
    """VTI's unit-direction addition and original-norm restoration."""
    norm = hidden.float().norm(dim=-1, keepdim=True)
    direction = F.normalize(vector.to(device=hidden.device).float(), dim=-1)
    steered = F.normalize(F.normalize(hidden.float(), dim=-1) + 0.1 * (alpha * direction),
                          dim=-1) * norm
    return steered.to(hidden.dtype)


def _grains_update(hidden: Tensor, vector: Tensor, alpha: float) -> Tensor:
    """Match the released GrAInS hook's 10x scale and sequence-axis norm."""
    candidate = hidden + 10 * alpha * vector.to(device=hidden.device)
    return candidate * (hidden.norm(dim=1, keepdim=True)
                        / (candidate.norm(dim=1, keepdim=True) + EPS))


def beta_gate(similarity: Tensor, k: float = 5.0, c: float = 1.0,
              minimum: float = 0.05, maximum: float = 1.0) -> Tensor:
    """RUDDER Eq. (3): higher alignment means stronger reinforcement."""
    if k < 0 or not 0 <= minimum <= maximum <= 1:
        raise ValueError("Gate requires k >= 0 and 0 <= minimum <= maximum <= 1.")
    positive = F.softplus(k * similarity + c)
    negative = F.softplus(-k * similarity + c)
    return (positive / (positive + negative)).clamp(minimum, maximum)


def revis_update(hidden: Tensor, vector: Tensor, alpha: float, threshold: float,
                 gamma: float = 1.0) -> Tensor:
    risk = -gamma * (F.normalize(hidden, dim=-1) @ F.normalize(vector.to(hidden), dim=-1))
    return hidden + alpha * (risk > threshold).unsqueeze(-1).to(hidden) * vector.to(hidden)


def _check_batch(hidden: Tensor) -> None:
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError("This initial implementation requires batch_size=1 and [1, tokens, hidden].")


@contextmanager
def _hook(module: Any, callback: Any, *, pre: bool = False) -> Iterator[None]:
    handle = (module.register_forward_pre_hook(callback, with_kwargs=True)
              if pre else module.register_forward_hook(callback))
    try:
        yield
    finally:
        handle.remove()


def _indices(config: dict, key: str, count: int) -> list[int]:
    values = config.get(key)
    indices = list(range(count)) if values is None else [int(x) for x in values]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError(f"{key} must be a nonempty list of unique layer indices.")
    if min(indices) < 0 or max(indices) >= count:
        raise ValueError(f"{key} contains an out-of-range layer; available count={count}.")
    return indices


def _capture(adapter: Any, batch: dict, layers: list[int], *, vision: bool = False,
             keep_positions: bool = False, text_only: bool = False,
             native_dtype: bool = False) -> dict[int, Tensor]:
    """Author hidden tuples: -1 is embedding; block i uses hidden_states[i+1]."""
    result: dict[int, Tensor] = {}
    if not vision:
        with torch.no_grad():
            output = (adapter.forward_text(batch, output_hidden_states=True, attention_mask=False)
                      if text_only else adapter.forward({**batch, "output_hidden_states": True}))
        states = output.hidden_states
        for index in layers:
            value = states[index + 1][0, -1].detach().cpu().clone()
            result[index] = value if native_dtype else value.float()
    else:
        # Gemma's returned final vision state is pooled. Capture the encoder
        # boundary to retain the same patch positions for embedding + all blocks.
        def save(index, hidden):
            if hidden.ndim == 3:
                _check_batch(hidden)
                hidden = hidden[0]
            value = hidden if keep_positions else hidden.mean(0)
            value = value.detach().cpu().clone()
            result[index] = value if native_dtype else value.float()
        with ExitStack() as stack:
            if -1 in layers:
                def embedding(_module, args, kwargs):
                    save(-1, _input(args, kwargs))
                stack.enter_context(_hook(adapter.vision_layers[0], embedding, pre=True))
            for index in layers:
                if index == -1:
                    continue
                def capture(_module, _args, output, index=index):
                    save(index, _tensor(output))
                stack.enter_context(_hook(adapter.vision_layers[index], capture))
            with torch.no_grad():
                adapter.forward_vision(batch)
    if set(result) != set(layers):
        raise RuntimeError("Some requested layers were not executed by the model.")
    return result


def _metadata(name: str, records: list[dict], config: dict) -> dict:
    return {
        "name": name,
        "implementation": "paper-derived-local-adaptation",
        "paper": PAPERS[name],
        "author_reproduction": False,
        "gpu_verified": False,
        "record_ids": [str(record.get("id", i)) for i, record in enumerate(records)],
        "config": dict(config),
        "format_version": 1,
    }


def _require_preferences(records: list[dict]) -> None:
    for record in records:
        for key in ("image", "prompt", "chosen", "rejected"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Extraction record {record.get('id')} needs nonempty {key!r}.")
        if record["chosen"] == record["rejected"]:
            raise ValueError("Chosen and rejected responses must differ.")


def _revis_extract(adapter: Any, records: list[dict], config: dict) -> dict:
    extraction = [r for r in records if r.get("role") != "calibration"]
    calibration = [r for r in records if r.get("role") == "calibration"]
    _require_preferences(extraction)
    if not extraction:
        raise ValueError("REVIS requires vector extraction records.")
    if {r["image"] for r in extraction} & {r["image"] for r in calibration}:
        raise ValueError("REVIS vector extraction and calibration images must be disjoint.")
    layers = _indices(config, "layers", len(adapter.layers))
    extraction_layers = list(range(-1, len(adapter.layers)))
    raw: dict[int, list[Tensor]] = {i: [] for i in extraction_layers}
    priors: dict[int, list[Tensor]] = {i: [] for i in extraction_layers}
    truths: dict[int, list[Tensor]] = {i: [] for i in extraction_layers}
    languages: dict[int, list[Tensor]] = {i: [] for i in extraction_layers}
    unknowns = (
        "I cannot answer this question because there is no image provided.",
        "Without an image, I cannot describe the scene.",
        "Please provide an image so I can describe it.",
        "I don't see any image here.",
        "The visual information is missing.",
    )
    rng = random.Random(int(config.get("seed", 0)))
    for record in extraction:
        unknown = record.get("neutral") or rng.choice(unknowns)
        conditions = [
            (record["image"], record["chosen"]),
            (record["image"], record["rejected"]),
            (None, record["chosen"]),
            (None, record["rejected"]),
            (None, unknown),
        ]
        states = [_capture(adapter, adapter.prepare(image, record["prompt"], answer,
                                                   answer_format="dialogue"), extraction_layers)
                  for image, answer in conditions]
        for layer in extraction_layers:
            truths[layer].append(states[0][layer] - states[1][layer])
            languages[layer].append(states[4][layer] - states[3][layer])
            priors[layer].append(states[3][layer] - states[4][layer])
            raw[layer].append(states[0][layer] - states[2][layer])
    vectors, truth_vectors, language_vectors = {}, {}, {}
    for layer in extraction_layers:
        # sklearn's randomized solver consumes RNG in the released tuple order.
        truth_vectors[layer] = _revis_pc1(torch.stack(truths[layer]))
        language_vectors[layer] = _revis_pc1(torch.stack(languages[layer]))
        prior = F.normalize(_revis_pc1(torch.stack(priors[layer])), dim=-1)
        purified = torch.stack(raw[layer])
        purified = purified - (purified @ prior.unsqueeze(-1)) * prior.unsqueeze(0)
        vectors[layer] = _revis_pc1(purified)
    vectors = {i: vectors[i] for i in layers}
    factual = {i: [] for i in layers}
    hallucinated = {i: [] for i in layers}
    calibration_results = []
    for record in calibration:
        label = str(record["label"]).lower()
        if label not in ("yes", "no"):
            raise ValueError("REVIS calibration needs yes/no labels.")
        risks = {}
        with ExitStack() as stack:
            for layer in layers:
                def observe(_module, _args, output, layer=layer):
                    if layer not in risks:
                        hidden = _tensor(output)[0, -1]
                        risks[layer] = float(-(F.normalize(hidden, dim=-1)
                                               @ F.normalize(vectors[layer].to(hidden), dim=-1)))
                stack.enter_context(_hook(adapter.layers[layer], observe))
            prompt = record["prompt"] + " Please answer yes or no."
            prediction = adapter.generate(adapter.prepare(record["image"], prompt), max_new_tokens=5).lower()
        answer = "yes" if "yes" in prediction else "no" if "no" in prediction else "unknown"
        category = ("TP" if label == answer == "yes" else "TN" if label == answer == "no"
                    else "FP" if label == "no" and answer == "yes" else "Other")
        destination = factual if category in ("TP", "TN") else hallucinated if category == "FP" else None
        if destination is not None:
            for layer in layers:
                destination[layer].append(risks[layer])
        calibration_results.append({"id": str(record.get("id", "")), "prediction": prediction,
                                    "category": category, "risks": risks})
    if calibration:
        separation = {i: float(np.mean(hallucinated[i]) - np.mean(factual[i]))
                      for i in layers if hallucinated[i] and factual[i]}
        valid = [i for i in layers if i >= int(config.get("min_layer", 10))
                 and separation.get(i, 0) > 0 and vectors[i].norm() > EPS]
        if "layer" in config:
            layer = int(config["layer"])
            if layer not in layers:
                raise ValueError("REVIS configured layer was not extracted.")
            if layer not in valid:
                raise ValueError("REVIS configured layer does not separate Base TP/TN from FP.")
        elif valid:
            layer = max(valid)
        else:
            raise ValueError("No REVIS layer separates factual from hallucinated calibration states.")
        quantile = float(config.get("risk_quantile", 0.85))
        if not 0 <= quantile <= 1:
            raise ValueError("risk_quantile must be between zero and one.")
        threshold = float(np.percentile(factual[layer], 100 * quantile))
    elif "layer" in config and "threshold" in config:
        layer, threshold = int(config["layer"]), float(config["threshold"])
        if layer not in layers:
            raise ValueError("REVIS configured layer was not extracted.")
        separation = {}
    else:
        raise ValueError("REVIS needs disjoint role='calibration' records, or explicit layer and threshold.")
    if vectors[layer].norm() <= EPS:
        raise ValueError("REVIS selected a zero purified direction.")
    return {**_metadata("revis", records, config), "vectors": vectors, "layer": layer,
            "truth_vectors": {i: truth_vectors[i] for i in layers},
            "language_vectors": {i: language_vectors[i] for i in layers},
            "threshold": threshold, "calibration_separation": separation,
            "calibration_records": calibration_results,
            "calibration": "held-out-records" if calibration else "externally-specified"}


def _masked_pixels(batch: dict, generator: torch.Generator, fraction: float,
                   patch_size: int) -> dict:
    if "pixel_values" not in batch:
        raise ValueError("VTI requires adapter.prepare to return pixel_values.")
    pixels = batch["pixel_values"]
    masked = pixels.clone()
    if pixels.ndim == 2:
        # Qwen processor represents each spatial/temporal patch as a row.
        count = pixels.shape[0]
        chosen = torch.randperm(count, generator=generator)[:int(count * fraction)]
        # Qwen packs each RGB channel's temporal/spatial samples contiguously.
        # The author fills masked patches with one global mean per channel.
        channel_mean = pixels.reshape(count, 3, -1).mean(dim=(0, 2))
        masked[chosen.to(masked.device)] = channel_mean.repeat_interleave(pixels.shape[1] // 3)
    elif pixels.ndim == 3 and pixels.shape[0] == 1:
        # Gemma 4 keeps image patches in a batch dimension.
        valid = (batch["image_position_ids"][0] != -1).all(dim=-1).nonzero().flatten().cpu()
        count = len(valid)
        chosen = valid[torch.randperm(count, generator=generator)[:int(count * fraction)]]
        rgb = pixels[:, valid.to(pixels.device)].reshape(-1, 3).mean(0)
        masked[:, chosen.to(masked.device)] = rgb.repeat(pixels.shape[-1] // 3)
    elif pixels.ndim == 4 and pixels.shape[0] == 1:
        height, width = pixels.shape[-2:]
        rows, cols = (height + patch_size - 1) // patch_size, (width + patch_size - 1) // patch_size
        chosen = torch.randperm(rows * cols, generator=generator)[:int(rows * cols * fraction)]
        for index in chosen.tolist():
            row, col = divmod(index, cols)
            masked[..., row * patch_size:(row + 1) * patch_size,
                   col * patch_size:(col + 1) * patch_size] = pixels.mean(dim=(-2, -1), keepdim=True)
    else:
        raise ValueError("VTI masking supports Qwen/Gemma patch rows or a single [1,C,H,W] image.")
    return {**batch, "pixel_values": masked}


def _vti_extract(adapter: Any, records: list[dict], config: dict) -> dict:
    _require_preferences(records)
    text_layers = _indices(config, "layers", len(adapter.layers))
    vision_layers = _indices(config, "vision_layers", len(adapter.vision_layers))
    pooling = config.get("vision_pooling", "position")
    if pooling not in ("position", "mean"):
        raise ValueError("vision_pooling must be 'position' or the explicit 'mean' adaptation.")
    masks = int(config.get("masks_per_image", 50))
    fraction = float(config.get("mask_fraction", 0.99))
    patch_size = int(config.get("mask_patch_size", 14))
    if masks < 1 or not 0 < fraction < 1 or patch_size < 1:
        raise ValueError("VTI requires positive masks/patch size and 0 < mask_fraction < 1.")
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    all_vision = list(range(-1, len(adapter.vision_layers)))
    all_text = list(range(-1, len(adapter.layers)))
    visual_deltas = []
    for record in records:
        batch = adapter.prepare(record["image"], record["prompt"])
        original = _capture(adapter, batch, all_vision, vision=True,
                            keep_positions=pooling == "position", native_dtype=True)
        corrupted = [
            _capture(adapter, _masked_pixels(batch, generator, fraction, patch_size),
                     all_vision, vision=True, keep_positions=pooling == "position", native_dtype=True)
            for _ in range(masks)
        ]
        # Released get_demos orders [masked_trials, original].
        delta = torch.stack([torch.stack([states[i] for states in corrupted]).mean(0)
                             - original[i] for i in all_vision])
        if visual_deltas and delta.shape != visual_deltas[0].shape:
            raise ValueError("VTI position-specific PCA requires equal vision token grids.")
        visual_deltas.append(delta)
    visual = torch.stack(visual_deltas)
    if pooling == "position":
        count, layers_count, tokens, width = visual.shape
        # Preserve obtain_visual_vti's released layer/token flattening order.
        visual_fit = visual.reshape(count, tokens, layers_count * width).transpose(0, 1)
        visual_direction = _vti_pc1_plus_mean(visual_fit).reshape(layers_count, tokens, width)
    else:
        visual_direction = _vti_pc1_plus_mean(visual.reshape(len(records), -1)).reshape(
            len(all_vision), -1)
    vision_vectors = {i: visual_direction[i + 1] for i in vision_layers}
    # The author entry point installs visual steering before textual extraction.
    visual_artifact = {"name": "vti", "vectors": {}, "vision_vectors": vision_vectors}
    text_deltas = []
    with build_baseline("vti", adapter, visual_artifact, config):
        for record in records:
            chosen = _capture(adapter, adapter.prepare(
                record["image"], record["prompt"], record["chosen"], append_eos=False), all_text,
                native_dtype=True)
            rejected = _capture(adapter, adapter.prepare(
                record["image"], record["prompt"], record["rejected"], append_eos=False), all_text,
                native_dtype=True)
            text_deltas.append(torch.stack([chosen[i] - rejected[i] for i in all_text]))
    text_direction = _vti_pc1_plus_mean(torch.stack(text_deltas).flatten(1)).reshape(len(all_text), -1)
    return {**_metadata("vti", records, config),
            "vectors": {i: text_direction[i + 1] for i in text_layers},
            "vision_vectors": vision_vectors,
            "vision_pooling": pooling, "pixel_mask_baseline": "mean-in-processor-output-space"}


def _grains_attributions(adapter: Any, batch: dict, steps: int, sign: float) -> Tensor:
    """Released contrastive IG on token embeddings, without image features."""
    embeddings = adapter.model.get_input_embeddings()(batch["input_ids"]).detach()
    target_id = int(batch["input_ids"][0, -1])
    accumulated = torch.zeros_like(embeddings, dtype=torch.float32)
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    with torch.enable_grad():
        for node, weight in zip(nodes, weights):
            variable = (embeddings * float((node + 1) / 2)).detach().requires_grad_(True)
            output = adapter.forward_text(batch, inputs_embeds=variable)
            logits = output.logits
            objective = logits[0, -1].log_softmax(-1)[target_id]
            accumulated += sign * float(weight / 2) * torch.autograd.grad(objective, variable)[0].float()
    return (embeddings.float() * accumulated).sum(-1)[0].detach()


def _grains_pc1(samples: Tensor) -> Tensor:
    component = PCA(n_components=1).fit(samples.detach().float().cpu().numpy()).components_[0]
    return torch.from_numpy(component)


def _grains_mask_id(adapter: Any) -> int:
    tokenizer = adapter.processor.tokenizer
    token_id = tokenizer.pad_token_id or tokenizer.unk_token_id
    if token_id is None:
        raise ValueError("GrAInS needs a PAD or UNK token for author-style ablation.")
    return int(token_id)


def _grains_extract(adapter: Any, records: list[dict], config: dict) -> dict:
    _require_preferences(records)
    layer_config = config if "layers" in config else {"layers": [config.get("layer", len(adapter.layers) - 1)]}
    layers = _indices(layer_config, "layers", len(adapter.layers))
    steps = int(config.get("ig_steps", 20))
    top_k = int(config.get("top_k", 50))
    if steps < 1 or top_k < 1:
        raise ValueError("GrAInS needs ig_steps >= 1 and top_k >= 1.")
    deltas = [{i: [] for i in layers}, {i: [] for i in layers}]
    counts = []
    for record in records:
        positive = adapter.prepare(record["image"], record["prompt"], record["chosen"],
                                   append_eos=False, answer_format="plain")
        negative = adapter.prepare(record["image"], record["prompt"], record["rejected"],
                                   append_eos=False, answer_format="plain")
        pos_scores = _grains_attributions(adapter, positive, steps, 1.0)
        neg_scores = _grains_attributions(adapter, negative, steps, -1.0)
        indices_pair = tuple(torch.from_numpy(np.argsort(scores.cpu().numpy())[::-1][:top_k].copy())
                             for scores in (pos_scores, neg_scores))
        for destination, batch, indices in zip(deltas, (positive, negative), indices_pair):
            original = _capture(adapter, batch, layers, text_only=True)
            ablated_ids = batch["input_ids"].clone()
            ablated_ids[:, indices.to(ablated_ids.device)] = _grains_mask_id(adapter)
            masked = _capture(adapter, {"input_ids": ablated_ids}, layers, text_only=True)
            for layer in layers:
                destination[layer].append(original[layer] - masked[layer])
        counts.append({"id": str(record.get("id", "")),
                       "positive": len(indices_pair[0]), "negative": len(indices_pair[1])})
    return {**_metadata("grains", records, config),
            "vectors": {i: _grains_pc1(torch.stack(deltas[0][i] + deltas[1][i])) for i in layers},
            "attribution_token_counts": counts,
            "attribution_space": "token-embedding-zero-baseline-no-image-features",
            "objective": "last-logp-chosen-minus-last-logp-rejected"}


def extract_baseline(name: str, adapter: Any, records: list[dict],
                     config: dict | None = None) -> dict:
    """Extract offline artifacts; requires a model, but never a GPU specifically."""
    name, config = name.lower(), dict(config or {})
    if name not in PAPERS:
        raise ValueError(f"Unknown baseline {name!r}; choose {list(PAPERS)}.")
    if name == "rudder":
        layer = int(config.get("layer", len(adapter.layers) - 2))
        if not 0 <= layer < len(adapter.layers):
            raise ValueError("RUDDER layer index is out of range.")
        return {**_metadata(name, records, config), "layer": layer,
                "direction": "per-input-prefill; no offline direction"}
    if not records:
        raise ValueError(f"{name} requires nonempty extraction records.")
    training_states = [(module, module.training) for module in adapter.model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in adapter.model.parameters()]
    adapter.model.eval()
    try:
        if name == "grains":
            for parameter, _ in parameters:
                parameter.requires_grad_(False)
        return {"revis": _revis_extract, "vti": _vti_extract,
                "grains": _grains_extract}[name](adapter, records, config)
    finally:
        for module, training in training_states:
            module.training = training
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


@contextmanager
def _rudder(adapter: Any, config: dict, batch: dict, task: str) -> Iterator[None]:
    layer = int(config.get("layer", len(adapter.layers) - 2))
    if not 0 <= layer < len(adapter.layers):
        raise ValueError("RUDDER layer index is out of range.")
    pooling = config.get("pooling", "norm")
    if pooling not in ("mean", "norm"):
        raise ValueError("RUDDER pooling must be 'mean' or 'norm'.")
    state: dict[str, Any] = {}
    attention = adapter.layers[layer].self_attn
    answer_only = task in ("pope", "mme")

    def capture_input(_module, args, kwargs):
        state["before"] = _input(args, kwargs).detach()

    def capture_output(_module, _args, output):
        state["after"] = _tensor(output).detach()

    with _hook(attention, capture_input, pre=True), _hook(attention, capture_output), torch.no_grad():
        adapter.forward(batch)
    delta = state["after"] - state["before"]
    mask = torch.ones_like(delta[..., :1])
    if "attention_mask" in batch:
        if answer_only:
            padding = (batch["attention_mask"] == 0).sum(1)
            positions = torch.arange(delta.shape[1], device=delta.device)[None]
            mask *= (positions >= padding[:, None]).unsqueeze(-1).to(delta)
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(delta)
    magnitudes = (delta.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6
                  if answer_only else delta.norm(dim=-1, keepdim=True))
    weights = (magnitudes if pooling == "norm" else torch.ones_like(mask)) * mask
    denominator = weights.sum(1, keepdim=True)
    denominator = denominator + 1e-6 if answer_only else denominator.clamp_min(1e-6)
    card = F.normalize((delta * weights).sum(1, keepdim=True) / denominator, dim=-1)
    # set_vector in the release normalizes once more before the generation hook.
    card = F.normalize(card, dim=-1).detach()
    generation_mask = mask.clone()
    if answer_only:
        generation_mask.zero_()
        nonpad = (batch.get("attention_mask", mask[..., 0])[0] > 0).nonzero().flatten()
        if len(nonpad):
            generation_mask[:, nonpad[-1]] = 1

    def steer(_module, _args, output):
        update = _tensor(output)
        _check_batch(update)
        direction = F.normalize(card.to(update), dim=-1)
        alignment = F.cosine_similarity(F.normalize(update, dim=-1), direction,
                                        dim=-1, eps=1e-6).unsqueeze(-1)
        gate = beta_gate(alignment, float(config.get("gate_k", 5.0)), float(config.get("gate_c", 1.0)),
                         float(config.get("gate_min", 0.0)), float(config.get("gate_max", 0.8)))
        if update.shape[1] != 1:
            gate = gate * generation_mask.to(update)
        changed = torch.nan_to_num(update + float(config.get("alpha", 2.0)) * gate * direction,
                                   nan=0.0, posinf=1e4, neginf=-1e4)
        return _replace(output, changed)
    with _hook(attention, steer):
        yield


@contextmanager
def build_baseline(name: str, adapter: Any, artifact: dict | None = None,
                   config: dict | None = None, *, batch: dict | None = None,
                   task: str | None = None) -> Iterator[None]:
    """Apply one baseline for one generation call; always removes its hooks."""
    name, config = name.lower(), dict(config or {})
    if name in ("base", "none"):
        yield
        return
    if name not in PAPERS:
        raise ValueError(f"Unknown baseline {name!r}.")
    if artifact is not None and artifact.get("name") != name:
        raise ValueError("Artifact baseline name does not match the requested baseline.")
    config = {**(artifact or {}).get("config", {}), **config}
    if name == "rudder":
        if batch is None or task is None:
            raise ValueError("RUDDER requires the input batch and benchmark task for CARD and its mask.")
        merged = {**config, **config.get("tasks", {}).get(task, {})}
        if artifact is not None:
            merged.setdefault("layer", artifact["layer"])
        with _rudder(adapter, merged, batch, task):
            yield
        return
    if artifact is None:
        raise ValueError(f"{name} requires a calibrated artifact; run extract first.")
    vectors = artifact["vectors"]
    if name == "grains":
        selected = config.get("layer")
        if selected is None and len(vectors) == 1:
            selected = next(iter(vectors))
        if selected is None or int(selected) not in vectors:
            raise ValueError("GrAInS requires one extracted steering layer selected on development data.")
        vectors = {int(selected): vectors[int(selected)]}
    alpha = float(config.get("alpha", 2.5 if name == "revis" else 1.0))
    with ExitStack() as stack:
        if name == "revis":
            layer = int(config.get("layer", artifact["layer"]))
            threshold = float(config.get("threshold", artifact["threshold"]))
            vector = vectors[layer]
            def steer(_module, _args, output):
                hidden = _tensor(output)
                _check_batch(hidden)
                changed = hidden.clone()
                changed[:] = revis_update(hidden, vector, alpha, threshold,
                                          float(config.get("risk_gamma", 1.0)))
                return _replace(output, changed)
            stack.enter_context(_hook(adapter.layers[layer], steer))
        else:
            for layer, vector in vectors.items():
                def steer(_module, _args, output, vector=vector):
                    hidden = _tensor(output)
                    _check_batch(hidden)
                    changed = hidden.clone()
                    if name == "grains":
                        changed[:] = _grains_update(hidden, vector, alpha)
                    else:
                        changed[:] = _vti_update(hidden, vector, alpha)
                    return _replace(output, changed)
                target = adapter.layers[int(layer)]
                if name == "vti":
                    target = getattr(target, "mlp", target)
                stack.enter_context(_hook(target, steer))
            if name == "vti":
                visual_alpha = float(config.get("vision_alpha", alpha))
                for layer, vector in artifact["vision_vectors"].items():
                    def vision_steer(_module, _args, output, vector=vector):
                        hidden = _tensor(output)
                        if vector.ndim == 2 and hidden.shape[-2:] != vector.shape:
                            raise ValueError("VTI visual token grid differs from extraction grid.")
                        return _replace(output, _vti_update(hidden, vector, visual_alpha))
                    target = adapter.vision_layers[int(layer)]
                    stack.enter_context(_hook(getattr(target, "mlp", target), vision_steer))
        yield
