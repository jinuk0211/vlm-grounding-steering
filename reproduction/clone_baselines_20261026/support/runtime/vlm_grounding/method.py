"""Evidence-Preserving Head Steering (EPHS), a provisional research method.

Discovery uses clean activation donors as an oracle diagnostic. Deployment only
rescales selected attention heads in the current input's forward pass.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math
import random
from statistics import mean, median
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.nn import functional as F


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positions(value: Any) -> str:
    if value not in {"last", "all"}:
        raise ValueError("positions must be 'last' or 'all'")
    return value


def _adapter_shape(adapter: Any) -> tuple[int, int]:
    count = len(adapter.attn_projections)
    heads = _integer(adapter.num_heads, "adapter.num_heads")
    if not count or count != len(adapter.layers):
        raise ValueError("One attention output projection is required per decoder layer")
    return count, heads


def _head_view(value: Tensor, num_heads: int) -> Tensor:
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError("EPHS currently requires batch size 1 and [batch, sequence, hidden] inputs")
    if not value.shape[1] or value.shape[-1] % num_heads:
        raise ValueError("o_proj input width must be divisible by the query-head count")
    return value.reshape(*value.shape[:-1], num_heads, value.shape[-1] // num_heads)


def _last_logits(output: Any) -> Tensor:
    logits = output["logits"] if isinstance(output, dict) else output.logits
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Adapter logits must have shape [1, sequence, vocabulary]")
    return logits[0, -1].float()


def _margin(adapter: Any, batch: dict[str, Any]) -> Tensor:
    value = adapter.answer_margin(batch)
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise ValueError("adapter.answer_margin must return a scalar tensor")
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite answer margin")
    return value.reshape(())


@contextmanager
def _evaluation(adapter: Any, *, freeze: bool = False) -> Iterator[None]:
    states = [(module, module.training) for module in adapter.model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in adapter.model.parameters()]
    try:
        adapter.model.eval()
        if freeze:
            for parameter, _ in parameters:
                parameter.requires_grad_(False)
        yield
    finally:
        for module, training in states:
            module.training = training
        if freeze:
            for parameter, required in parameters:
                parameter.requires_grad_(required)


@contextmanager
def _capture(adapter: Any, layers: list[int]) -> Iterator[dict[int, Tensor]]:
    captures: dict[int, Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            value = _head_view(args[0], adapter.num_heads)
            captures[layer] = value[0, -1].detach().cpu().clone()
        return hook

    try:
        for layer in layers:
            handles.append(adapter.attn_projections[layer].register_forward_pre_hook(make_hook(layer)))
        yield captures
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _patch(adapter: Any, layer: int, head: int, donor: Tensor) -> Iterator[None]:
    def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        value = args[0]
        changed = _head_view(value, adapter.num_heads).clone()
        if changed.shape[-1] != donor.numel():
            raise ValueError("Donor and recipient head dimensions differ")
        changed[0, -1, head] = donor.to(device=value.device, dtype=value.dtype)
        return (changed.reshape_as(value), *args[1:])

    handle = adapter.attn_projections[layer].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _selected(adapter: Any, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    layers, heads = _adapter_shape(adapter)
    if artifact.get("num_heads", heads) != heads or artifact.get("layer_count", layers) != layers:
        raise ValueError("Artifact architecture does not match this adapter")
    entries = artifact.get("selected_heads", [])
    if not entries:
        raise ValueError("Artifact must contain at least one selected head")
    seen = set()
    normalized = []
    for entry in entries:
        layer = _integer(entry["layer"], "layer", 0)
        head = _integer(entry["head"], "head", 0)
        if layer >= layers or head >= heads or (layer, head) in seen:
            raise ValueError("Selected heads must be unique and within adapter dimensions")
        seen.add((layer, head))
        normalized.append({**entry, "layer": layer, "head": head})
    return normalized


@contextmanager
def _gains(
    adapter: Any, entries: list[dict[str, Any]], deltas: Any, positions: str,
) -> Iterator[None]:
    """`deltas` is a callable so every forward builds a fresh autograd graph."""
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        by_layer[entry["layer"]].append((entry["head"], index))
    handles = []

    def make_hook(members: list[tuple[int, int]]):
        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            value = args[0]
            shaped = _head_view(value, adapter.num_heads)
            current = deltas()
            indices = torch.tensor([head for head, _ in members], device=value.device)
            gain_indices = torch.tensor([index for _, index in members], device=current.device)
            updates = current[gain_indices].to(device=value.device, dtype=value.dtype)
            factors = torch.ones(adapter.num_heads, device=value.device, dtype=value.dtype)
            factors = factors.scatter(0, indices, 1 + updates)
            factors = factors.reshape(1, 1, -1, 1)
            if positions == "last":
                changed = torch.cat((shaped[:, :-1], shaped[:, -1:] * factors), dim=1)
            else:
                changed = shaped * factors
            return (changed.reshape_as(value), *args[1:])
        return hook

    try:
        for layer, members in by_layer.items():
            handles.append(adapter.attn_projections[layer].register_forward_pre_hook(make_hook(members)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def discover(adapter: Any, records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Rank heads by oracle restoration of an object-existence answer margin."""
    layer_count, head_count = _adapter_shape(adapter)
    layers = [_integer(value, "layer", 0) for value in config.get("layers", range(layer_count))]
    heads = [_integer(value, "head", 0) for value in config.get("heads", range(head_count))]
    if not layers or len(set(layers)) != len(layers) or max(layers) >= layer_count:
        raise ValueError("layers must be a nonempty unique list of valid decoder indices")
    if not heads or len(set(heads)) != len(heads) or max(heads) >= head_count:
        raise ValueError("heads must be a nonempty unique list of query-head indices")
    top_k = _integer(config.get("top_k", 8), "top_k")
    if top_k > len(layers) * len(heads):
        raise ValueError("top_k exceeds the number of candidate heads")
    penalty = _positive(config.get("control_penalty", 1.0), "control_penalty", allow_zero=True)
    floor = _positive(config.get("normalization_floor", 1.0), "normalization_floor")
    minimum = _positive(config.get("min_margin_drop", 0.05), "min_margin_drop", allow_zero=True)
    mode = config.get("selection_mode", "causal")
    if mode not in {"causal", "random", "magnitude"}:
        raise ValueError("selection_mode must be causal, random, or magnitude")
    limit = _integer(config.get("max_examples", len(records)), "max_examples")
    observations: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    used_ids, skipped = [], []

    with _evaluation(adapter), torch.no_grad():
        for index, record in enumerate(records[:limit]):
            record_id = str(record.get("id", index))
            if str(record.get("label", "yes")).lower() != "yes":
                raise ValueError("Discovery requires positive clean-image object-existence records")
            prompt = record["prompt"]
            clean = adapter.prepare(record["image"], prompt)
            masked = adapter.prepare(record["object_masked_image"], prompt)
            control = adapter.prepare(record["control_image"], prompt)
            with _capture(adapter, layers) as captures:
                clean_margin = float(_margin(adapter, clean))
            masked_margin = float(_margin(adapter, masked))
            control_margin = float(_margin(adapter, control))
            drop = clean_margin - masked_margin
            if clean_margin <= 0 or drop <= minimum:
                skipped.append({"id": record_id, "reason": "clean_wrong_or_insufficient_object_mask_effect"})
                continue
            if set(captures) != set(layers):
                raise ValueError("Some configured attention projections were not executed")
            denominator = max(drop, floor)
            group = str(record.get("group_id", record_id))
            for layer in layers:
                for head in heads:
                    donor = captures[layer][head]
                    with _patch(adapter, layer, head, donor):
                        rescue = float(_margin(adapter, masked)) - masked_margin
                        control_shift = float(_margin(adapter, control)) - control_margin
                    observations[layer, head].append({
                        "group": group,
                        "score": (rescue - penalty * abs(control_shift)) / denominator,
                        "recovery": rescue / denominator,
                        "control_shift": abs(control_shift) / denominator,
                        "magnitude": float(donor.float().square().mean().sqrt()),
                    })
            used_ids.append(record_id)

    if not used_ids:
        raise ValueError("No discovery examples passed the clean-correct and object-mask effect criteria")
    scores = []
    for (layer, head), values in observations.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            groups[value["group"]].append(value)
        scores.append({
            "layer": layer,
            "head": head,
            "score": median(mean(value["score"] for value in group) for group in groups.values()),
            "mean_recovery": mean(value["recovery"] for value in values),
            "mean_control_shift": mean(value["control_shift"] for value in values),
            "activation_magnitude": mean(value["magnitude"] for value in values),
            "n_examples": len(values),
            "n_groups": len(groups),
        })
    scores.sort(key=lambda item: (-item["score"], item["layer"], item["head"]))
    ranking = list(scores)
    if mode == "random":
        random.Random(config.get("seed", 0)).shuffle(ranking)
    elif mode == "magnitude":
        ranking.sort(key=lambda item: (-item["activation_magnitude"], item["layer"], item["head"]))
    return {
        "format_version": 1,
        "method": "EPHS",
        "stage": "discovery",
        "num_heads": head_count,
        "layer_count": layer_count,
        "selection_mode": mode,
        "selected_heads": [dict(item) for item in ranking[:top_k]],
        "head_scores": scores,
        "discovery_ids": used_ids,
        "skipped_records": skipped,
        "score_rule": "median_group_mean((object_mask_rescue - penalty*abs(control_rescue))/max(clean_mask_drop,floor))",
        "config": dict(config),
        "oracle_donors_used": True,
    }


def fit(
    adapter: Any,
    records: list[dict[str, Any]],
    selection: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fit bounded head gains; all base-model parameters remain frozen."""
    entries = _selected(adapter, selection)
    positions = _positions(config.get("positions", "last"))
    epochs = _integer(config.get("epochs", 2), "epochs")
    learning_rate = _positive(config.get("lr", 0.03), "lr")
    max_gain = _positive(config.get("max_gain", 0.25), "max_gain")
    if max_gain >= 1:
        raise ValueError("max_gain must be < 1 to avoid reversing or zeroing a head")
    kl_weight = _positive(config.get("kl_weight", 1.0), "kl_weight", allow_zero=True)
    l2_weight = _positive(config.get("l2_weight", 0.01), "l2_weight", allow_zero=True)
    maximum_steps = config.get("max_steps")
    if maximum_steps is not None:
        maximum_steps = _integer(maximum_steps, "max_steps", 2)
        if maximum_steps % 2:
            raise ValueError("max_steps must be even to preserve yes/no balance")
    labels: dict[str, list[int]] = {"yes": [], "no": []}
    for index, record in enumerate(records):
        label = str(record["label"]).lower()
        if label not in labels:
            raise ValueError("Fit labels must be yes or no")
        labels[label].append(index)
    if not labels["yes"] or not labels["no"]:
        raise ValueError("Fit data must include both yes and no object-existence examples")
    raw = torch.nn.Parameter(torch.zeros(len(entries), device=adapter.device, dtype=torch.float32))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    cache: dict[int, tuple[bool, Tensor | None]] = {}
    history = []
    steps = 0

    with _evaluation(adapter, freeze=True):
        with torch.no_grad():
            for index, record in enumerate(records):
                batch = adapter.prepare(record["image"], record["prompt"])
                baseline_margin = float(_margin(adapter, batch))
                target = 1 if str(record["label"]).lower() == "yes" else -1
                preserve = bool(record.get("preserve", True)) and target * baseline_margin > 0
                reference = None
                if preserve and kl_weight:
                    reference = F.log_softmax(_last_logits(adapter.forward(batch)), dim=-1).cpu()
                cache[index] = preserve, reference
        with _gains(adapter, entries, lambda: max_gain * raw.tanh(), positions):
            for epoch in range(epochs):
                rng = random.Random(int(config.get("seed", 0)) + epoch)
                positive, negative = list(labels["yes"]), list(labels["no"])
                rng.shuffle(positive)
                rng.shuffle(negative)
                order = []
                for index in range(max(len(positive), len(negative))):
                    pair = [positive[index % len(positive)], negative[index % len(negative)]]
                    rng.shuffle(pair)
                    order.extend(pair)
                losses, kls = [], []
                for index in order:
                    if maximum_steps is not None and steps >= maximum_steps:
                        break
                    record = records[index]
                    batch = adapter.prepare(record["image"], record["prompt"])
                    target = 1 if str(record["label"]).lower() == "yes" else -1
                    optimizer.zero_grad(set_to_none=True)
                    margin = _margin(adapter, batch).float()
                    if not margin.requires_grad:
                        raise ValueError("Adapter must preserve gradients through forward and answer_margin")
                    classification = F.softplus(-target * margin)
                    preservation = torch.zeros((), device=margin.device)
                    _, reference = cache[index]
                    if reference is not None:
                        current = F.log_softmax(_last_logits(adapter.forward(batch)), dim=-1)
                        preservation = F.kl_div(current, reference.to(current.device), reduction="sum", log_target=True)
                    regularization = (max_gain * raw.tanh()).square().mean()
                    loss = classification + kl_weight * preservation + l2_weight * regularization
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite gain-fitting loss")
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    kls.append(float(preservation.detach()))
                    steps += 1
                if losses:
                    history.append({"epoch": epoch + 1, "mean_loss": mean(losses), "mean_kl": mean(kls)})
                if maximum_steps is not None and steps >= maximum_steps:
                    break
    deltas = (max_gain * raw.detach().tanh()).cpu().tolist()
    return {
        "format_version": 1,
        "method": "EPHS",
        "stage": "fitted",
        "num_heads": adapter.num_heads,
        "layer_count": len(adapter.layers),
        "selection_mode": selection.get("selection_mode", "causal"),
        "selected_heads": [{**entry, "delta": delta} for entry, delta in zip(entries, deltas)],
        "max_gain": max_gain,
        "positions": positions,
        "fit_ids": [str(record.get("id", index)) for index, record in enumerate(records)],
        "fit_steps": steps,
        "preserved_examples": sum(preserve for preserve, _ in cache.values()),
        "history": history,
        "config": dict(config),
        "discovery_ids": list(selection.get("discovery_ids", [])),
        "deployment_requires_donor": False,
    }


@contextmanager
def steer(adapter: Any, artifact: dict[str, Any], config: dict[str, Any] | None = None) -> Iterator[None]:
    """Apply fitted gains to the current input only; hooks are always removed."""
    entries = _selected(adapter, artifact)
    if artifact.get("stage") != "fitted":
        raise ValueError("steer requires a fitted artifact")
    positions = _positions(artifact.get("positions", "last"))
    if "positions" in (config or {}) and config["positions"] != positions:
        raise ValueError("Deployment positions must match the fitted artifact")
    strength = _positive((config or {}).get("strength", 1.0), "strength", allow_zero=True)
    maximum = _positive(artifact["max_gain"], "max_gain")
    if maximum >= 1:
        raise ValueError("Artifact max_gain must be < 1")
    values = [float(entry["delta"]) * strength for entry in entries]
    if any(not math.isfinite(value) or abs(value) > maximum + 1e-7 for value in values):
        raise ValueError("Requested steering exceeds the artifact's bounded gain")
    tensor = torch.tensor(values, device=adapter.device, dtype=torch.float32)
    with _gains(adapter, entries, lambda: tensor, positions):
        yield
