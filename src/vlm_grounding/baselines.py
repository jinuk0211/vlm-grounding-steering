"""Paper-derived, local baseline implementations (not author reproductions).

See docs/baseline_implementation.md for equations, extraction requirements, and
architecture adaptations. Hooks are temporary and never modify model weights.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Iterator

from numpy.polynomial.legendre import leggauss
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


def _replace_input(args: tuple, kwargs: dict, value: Tensor) -> tuple:
    if args:
        return (value, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": value}


def unit(vector: Tensor) -> Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(EPS)


def orthogonalize(visual: Tensor, prior: Tensor) -> Tensor:
    """REVIS Eq. (3), retaining the purified vector's original scale."""
    projection = (visual * prior).sum(-1, keepdim=True)
    projection = projection / prior.square().sum(-1, keepdim=True).clamp_min(EPS)
    return visual - projection * prior


def principal_direction(samples: Tensor) -> Tensor:
    """Centered PC1 along the sample axis, with deterministic mean-delta sign.

    Remaining axes may be [hidden] or [vision_position, hidden]. Degenerate
    constant nonzero samples use their mean; all-zero samples give zero.
    """
    if samples.ndim < 2 or samples.shape[0] < 1:
        raise ValueError("PCA needs at least one sample and a feature dimension.")
    samples = samples.detach().float().cpu()
    mean = samples.mean(0)
    centered = samples - mean
    matrices = centered.movedim(0, -2)
    _, singular, vh = torch.linalg.svd(matrices, full_matrices=False)
    direction = vh[..., 0, :]
    degenerate = singular[..., 0] < EPS
    direction = torch.where(degenerate[..., None], unit(mean), direction)
    sign = torch.where((direction * mean).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    return direction * sign


def norm_preserving_add(hidden: Tensor, vector: Tensor, alpha: float) -> Tensor:
    """GrAInS Eq. (5), preserving zero vectors and avoiding division by zero."""
    vector = vector.to(hidden)
    candidate = hidden + alpha * vector
    old_norm = hidden.float().norm(dim=-1, keepdim=True)
    new_norm = candidate.float().norm(dim=-1, keepdim=True)
    scaled = candidate * (old_norm / new_norm.clamp_min(EPS)).to(hidden)
    return torch.where(new_norm > EPS, scaled, hidden)


def beta_gate(similarity: Tensor, k: float = 5.0, c: float = 1.0,
              minimum: float = 0.05, maximum: float = 1.0) -> Tensor:
    """RUDDER Eq. (3): higher alignment means stronger reinforcement."""
    if k < 0 or not 0 <= minimum <= maximum <= 1:
        raise ValueError("Gate requires k >= 0 and 0 <= minimum <= maximum <= 1.")
    positive = F.softplus(k * similarity.float() + c)
    negative = F.softplus(-k * similarity.float() + c)
    return (positive / (positive + negative).clamp_min(EPS)).clamp(minimum, maximum)


def revis_update(hidden: Tensor, vector: Tensor, alpha: float, threshold: float) -> Tensor:
    risk = -F.cosine_similarity(hidden.float(), vector.to(hidden).float(), dim=-1)
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
             input_transform: Any = None, keep_positions: bool = False) -> dict[int, Tensor]:
    result: dict[int, Tensor] = {}
    modules = adapter.vision_layers if vision else adapter.layers
    with ExitStack() as stack:
        if input_transform is not None:
            def transform(_module, args, kwargs):
                hidden = _input(args, kwargs)
                return _replace_input(args, kwargs, input_transform(hidden))
            stack.enter_context(_hook(adapter.layers[0], transform, pre=True))
        for index in layers:
            def capture(_module, _args, output, index=index):
                hidden = _tensor(output)
                if vision:
                    if hidden.ndim == 3:
                        _check_batch(hidden)
                        hidden = hidden[0]
                    if hidden.ndim != 2:
                        raise ValueError("Vision activations must be [tokens, hidden] or [1, tokens, hidden].")
                    value = hidden if keep_positions else hidden.mean(0)
                else:
                    _check_batch(hidden)
                    value = hidden[0, -1]
                result[index] = value.detach().float().cpu().clone()
            stack.enter_context(_hook(modules[index], capture))
        with torch.no_grad():
            adapter.forward(batch)
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
    _require_preferences(records)
    extraction = [r for r in records if r.get("role") != "calibration"]
    calibration = [r for r in records if r.get("role") == "calibration"]
    if not extraction:
        raise ValueError("REVIS requires vector extraction records.")
    if {r["image"] for r in extraction} & {r["image"] for r in calibration}:
        raise ValueError("REVIS vector extraction and calibration images must be disjoint.")
    layers = _indices(config, "layers", len(adapter.layers))
    raw: dict[int, list[Tensor]] = {i: [] for i in layers}
    priors: dict[int, list[Tensor]] = {i: [] for i in layers}
    for record in extraction:
        conditions = [
            (record["image"], record["chosen"]),
            (None, record["chosen"]),
            (None, record["rejected"]),
            (None, record.get("neutral", "I cannot determine this without an image.")),
        ]
        states = [_capture(adapter, adapter.prepare(image, record["prompt"], answer), layers)
                  for image, answer in conditions]
        for layer in layers:
            raw[layer].append(states[0][layer] - states[1][layer])
            priors[layer].append(states[2][layer] - states[3][layer])
    vectors = {layer: orthogonalize(torch.stack(raw[layer]).mean(0),
                                    torch.stack(priors[layer]).mean(0))
               for layer in layers}
    factual = {i: [] for i in layers}
    hallucinated = {i: [] for i in layers}
    for record in calibration:
        for response, destination in ((record["chosen"], factual),
                                      (record["rejected"], hallucinated)):
            states = _capture(adapter, adapter.prepare(record["image"], record["prompt"], response), layers)
            for layer in layers:
                destination[layer].append(-F.cosine_similarity(states[layer], vectors[layer], dim=0))
    if calibration:
        separation = {i: float(torch.stack(hallucinated[i]).mean()
                               - torch.stack(factual[i]).mean()) for i in layers}
        valid = [i for i in layers if separation[i] > 0 and vectors[i].norm() > EPS]
        if "layer" in config:
            layer = int(config["layer"])
            if layer not in layers:
                raise ValueError("REVIS configured layer was not extracted.")
        elif valid:
            layer = max(valid)
        else:
            raise ValueError("No REVIS layer separates factual from hallucinated calibration states.")
        quantile = float(config.get("risk_quantile", 0.8))
        if not 0 <= quantile <= 1:
            raise ValueError("risk_quantile must be between zero and one.")
        threshold = float(torch.quantile(torch.stack(factual[layer]), quantile))
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
            "threshold": threshold, "calibration_separation": separation,
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
        chosen = torch.randperm(count, generator=generator)[:max(1, round(count * fraction))]
        masked[chosen.to(masked.device)] = 0
    elif pixels.ndim == 3 and pixels.shape[0] == 1:
        # Gemma 4 keeps image patches in a batch dimension.
        valid = (batch["image_position_ids"][0] != -1).all(dim=-1).nonzero().flatten().cpu()
        count = len(valid)
        chosen = valid[torch.randperm(count, generator=generator)[:max(1, round(count * fraction))]]
        masked[:, chosen.to(masked.device)] = 0
    elif pixels.ndim == 4 and pixels.shape[0] == 1:
        height, width = pixels.shape[-2:]
        rows, cols = (height + patch_size - 1) // patch_size, (width + patch_size - 1) // patch_size
        chosen = torch.randperm(rows * cols, generator=generator)[:max(1, round(rows * cols * fraction))]
        for index in chosen.tolist():
            row, col = divmod(index, cols)
            masked[..., row * patch_size:(row + 1) * patch_size,
                   col * patch_size:(col + 1) * patch_size] = 0
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
    masks = int(config.get("masks_per_image", 8))
    fraction = float(config.get("mask_fraction", 0.1))
    patch_size = int(config.get("mask_patch_size", 14))
    if masks < 1 or not 0 < fraction < 1 or patch_size < 1:
        raise ValueError("VTI requires positive masks/patch size and 0 < mask_fraction < 1.")
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    visual_deltas = {i: [] for i in vision_layers}
    text_deltas = {i: [] for i in text_layers}
    for record in records:
        batch = adapter.prepare(record["image"], record["prompt"])
        original = _capture(adapter, batch, vision_layers, vision=True,
                            keep_positions=pooling == "position")
        corrupted = [
            _capture(adapter, _masked_pixels(batch, generator, fraction, patch_size),
                     vision_layers, vision=True, keep_positions=pooling == "position")
            for _ in range(masks)
        ]
        for layer in vision_layers:
            delta = torch.stack([states[layer] for states in corrupted]).mean(0) - original[layer]
            if visual_deltas[layer] and delta.shape != visual_deltas[layer][0].shape:
                raise ValueError("VTI position-specific PCA requires equal vision token grids. "
                                 "Preprocess fixed-size images or explicitly set vision_pooling='mean'.")
            visual_deltas[layer].append(delta)
        chosen = _capture(adapter, adapter.prepare(record["image"], record["prompt"], record["chosen"]), text_layers)
        rejected = _capture(adapter, adapter.prepare(record["image"], record["prompt"], record["rejected"]), text_layers)
        for layer in text_layers:
            text_deltas[layer].append(chosen[layer] - rejected[layer])
    return {**_metadata("vti", records, config),
            "vectors": {i: principal_direction(torch.stack(v)) for i, v in text_deltas.items()},
            "vision_vectors": {i: principal_direction(torch.stack(v)) for i, v in visual_deltas.items()},
            "vision_pooling": pooling, "pixel_mask_baseline": "zero-in-processor-output-space"}


def _fused_embeddings(adapter: Any, batch: dict) -> Tensor:
    captured = []
    def capture(_module, args, kwargs):
        captured.append(_input(args, kwargs).detach().clone())
    with _hook(adapter.layers[0], capture, pre=True), torch.no_grad():
        adapter.forward(batch)
    if len(captured) != 1:
        raise RuntimeError("Expected one full teacher-forced decoder pass.")
    _check_batch(captured[0])
    return captured[0]


def sequence_log_probability(logits: Tensor, labels: Tensor) -> Tensor:
    """Sum conditional log probabilities over labeled completion tokens only."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("Labels must align with the expanded decoder token sequence.")
    targets = labels[:, 1:]
    valid = targets != -100
    if not valid.any():
        raise ValueError("No labeled completion token is available for the preference objective.")
    selected = logits[:, :-1].float().log_softmax(-1).gather(
        -1, targets.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    return selected.masked_select(valid).sum()


def _prompt_length(batch: dict, embeddings: Tensor) -> int:
    labels = batch.get("labels")
    if labels is None or labels.shape != embeddings.shape[:2]:
        raise ValueError("GrAInS requires labels matching fused sequence; prompt labels must be -100.")
    positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
    if not len(positions) or positions[0] == 0:
        raise ValueError("GrAInS requires a nonempty masked prompt and a labeled response.")
    return int(positions[0])


def _preference_attributions(adapter: Any, positive: dict, negative: dict, steps: int) -> Tensor:
    pos_embeddings = _fused_embeddings(adapter, positive)
    neg_embeddings = _fused_embeddings(adapter, negative)
    length = _prompt_length(positive, pos_embeddings)
    if _prompt_length(negative, neg_embeddings) != length:
        raise ValueError("Chosen and rejected completions must have identical prompt tokenization.")
    prompt = pos_embeddings[:, :length].detach()
    if not torch.allclose(prompt, neg_embeddings[:, :length], atol=1e-5, rtol=1e-4):
        raise ValueError("Preference pair fused prompt embeddings differ.")
    accumulated = torch.zeros_like(prompt, dtype=torch.float32)
    # Gauss-Legendre integrates the same zero-baseline path without evaluating
    # alpha=0, where real BF16 Gemma checkpoints can have non-finite gradients.
    # This is also Captum IntegratedGradients' default quadrature.
    nodes, weights = leggauss(steps)
    with torch.enable_grad():
        for node, weight in zip(nodes, weights):
            alpha = float((node + 1) / 2)
            variable = (prompt * alpha).detach().requires_grad_(True)
            gradient = torch.zeros_like(variable, dtype=torch.float32)
            for batch, embeddings, sign in ((positive, pos_embeddings, 1.0),
                                             (negative, neg_embeddings, -1.0)):
                def replace_prompt(_module, args, kwargs):
                    full = torch.cat((variable, embeddings[:, length:]), dim=1)
                    return _replace_input(args, kwargs, full)
                with _hook(adapter.layers[0], replace_prompt, pre=True):
                    output = adapter.forward(batch)
                    logits = output["logits"] if isinstance(output, dict) else output.logits
                    objective = sequence_log_probability(logits, batch["labels"])
                    if not torch.isfinite(objective):
                        raise ValueError(f"Non-finite GrAInS objective at IG alpha={alpha:.6g}.")
                    current = torch.autograd.grad(objective, variable)[0].float()
                    if not torch.isfinite(current).all():
                        raise ValueError(f"Non-finite GrAInS gradient at IG alpha={alpha:.6g}.")
                    gradient += sign * current
            accumulated += float(weight / 2) * gradient
    scores = (prompt.float() * accumulated).sum(-1)[0].detach()
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite GrAInS token attributions.")
    return scores


def _grains_extract(adapter: Any, records: list[dict], config: dict) -> dict:
    _require_preferences(records)
    layers = _indices(config, "layers", len(adapter.layers))
    steps = int(config.get("ig_steps", 10))
    top_k = int(config.get("top_k", 10))
    if steps < 1 or top_k < 1:
        raise ValueError("GrAInS needs ig_steps >= 1 and top_k >= 1.")
    positive_deltas = {i: [] for i in layers}
    negative_deltas = {i: [] for i in layers}
    counts = []
    for record in records:
        positive = adapter.prepare(record["image"], record["prompt"], record["chosen"])
        negative = adapter.prepare(record["image"], record["prompt"], record["rejected"])
        scores = _preference_attributions(adapter, positive, negative, steps)
        batch = adapter.prepare(record["image"], record["prompt"])
        embeddings = _fused_embeddings(adapter, batch)
        if embeddings.shape[1] != len(scores):
            raise ValueError("Generation prompt length differs from preference prompt length.")
        pos = (scores > 0).nonzero(as_tuple=False).flatten()
        neg = (scores < 0).nonzero(as_tuple=False).flatten()
        pos = pos[scores[pos].argsort(descending=True)[:top_k]]
        neg = neg[scores[neg].argsort()[:top_k]]
        original = _capture(adapter, batch, layers)
        for indices, destination in ((pos, positive_deltas), (neg, negative_deltas)):
            def mask_tokens(hidden, indices=indices):
                changed = hidden.clone()
                changed[:, indices.to(hidden.device)] = 0
                return changed
            masked = _capture(adapter, batch, layers, input_transform=mask_tokens)
            for layer in layers:
                destination[layer].append(original[layer] - masked[layer])
        counts.append({"id": str(record.get("id", "")), "positive": len(pos), "negative": len(neg)})
    plus = {i: principal_direction(torch.stack(v)) for i, v in positive_deltas.items()}
    minus = {i: principal_direction(torch.stack(v)) for i, v in negative_deltas.items()}
    return {**_metadata("grains", records, config),
            "vectors": {i: plus[i] - minus[i] for i in layers},
            "positive_vectors": plus, "negative_vectors": minus,
            "attribution_token_counts": counts,
            "attribution_space": "fused-input-embeddings-zero-baseline",
            "ig_quadrature": "gauss-legendre",
            "objective": "sum-logp-chosen-minus-sum-logp-rejected"}


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
def _rudder(adapter: Any, config: dict) -> Iterator[None]:
    layer = int(config.get("layer", len(adapter.layers) - 2))
    if not 0 <= layer < len(adapter.layers):
        raise ValueError("RUDDER layer index is out of range.")
    pooling = config.get("pooling", "norm")
    if pooling not in ("mean", "norm"):
        raise ValueError("RUDDER pooling must be 'mean' or 'norm'.")
    state: dict[str, Any] = {"card": None}
    def capture_input(_module, args, kwargs):
        hidden = _input(args, kwargs)
        _check_batch(hidden)
        state["hidden"] = hidden
    def steer(_module, _args, output):
        update = _tensor(output)
        _check_batch(update)
        hidden = state["hidden"]
        if state["card"] is None:
            weights = (update.float().norm(dim=-1, keepdim=True) if pooling == "norm"
                       else torch.ones_like(update[..., :1], dtype=torch.float32))
            pooled = (update.float() * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(EPS)
            state["card"] = unit(pooled).detach()
            if not config.get("steer_first_token", True):
                return output
        elif update.shape[1] != 1:
            raise ValueError("RUDDER requires cached single-token decode after prefill; use a fresh context per prompt.")
        card = state["card"].to(update)
        last = hidden[:, -1:, :]
        alignment = F.cosine_similarity(last.float(), card.float(), dim=-1).unsqueeze(-1)
        gate = beta_gate(alignment, float(config.get("gate_k", 5.0)), float(config.get("gate_c", 1.0)),
                         float(config.get("gate_min", 0.05)), float(config.get("gate_max", 1.0)))
        changed = update.clone()
        changed[:, -1:, :] += float(config.get("alpha", 8.0)) * gate.to(update) * card
        return _replace(output, changed)
    with _hook(adapter.layers[layer], capture_input, pre=True), \
            _hook(adapter.attn_updates[layer], steer):
        yield


@contextmanager
def build_baseline(name: str, adapter: Any, artifact: dict | None = None,
                   config: dict | None = None) -> Iterator[None]:
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
        merged = config
        if artifact is not None:
            merged.setdefault("layer", artifact["layer"])
        with _rudder(adapter, merged):
            yield
        return
    if artifact is None:
        raise ValueError(f"{name} requires a calibrated artifact; run extract first.")
    vectors = artifact["vectors"]
    alpha = float(config.get("alpha", 1.6 if name == "revis" else 1.0))
    with ExitStack() as stack:
        if name == "revis":
            layer = int(config.get("layer", artifact["layer"]))
            threshold = float(config.get("threshold", artifact["threshold"]))
            vector = vectors[layer]
            def steer(_module, _args, output):
                hidden = _tensor(output)
                _check_batch(hidden)
                changed = hidden.clone()
                changed[:, -1:, :] = revis_update(hidden[:, -1:, :], vector, alpha, threshold)
                return _replace(output, changed)
            stack.enter_context(_hook(adapter.layers[layer], steer))
        else:
            for layer, vector in vectors.items():
                def steer(_module, _args, output, vector=vector):
                    hidden = _tensor(output)
                    _check_batch(hidden)
                    changed = hidden.clone()
                    target = hidden[:, -1:, :]
                    changed[:, -1:, :] = (norm_preserving_add(target, vector, alpha) if name == "grains"
                                         else target + alpha * vector.to(target))
                    return _replace(output, changed)
                stack.enter_context(_hook(adapter.layers[int(layer)], steer))
            if name == "vti":
                visual_alpha = float(config.get("vision_alpha", alpha))
                for layer, vector in artifact["vision_vectors"].items():
                    def vision_steer(_module, _args, output, vector=vector):
                        hidden = _tensor(output)
                        if vector.ndim == 2 and hidden.shape[-2:] != vector.shape:
                            raise ValueError("VTI visual token grid differs from extraction grid.")
                        return _replace(output, hidden + visual_alpha * vector.to(hidden))
                    stack.enter_context(_hook(adapter.vision_layers[int(layer)], vision_steer))
        yield
