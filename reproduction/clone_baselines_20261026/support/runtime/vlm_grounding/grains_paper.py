"""GrAInS paper equations on Qwen2.5-VL; separate from released-code controls.

Image patches are attributed after fusion, responses are teacher forced, and only
the shared prompt is interpolated. This module is not an author generation helper.
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import Tensor


class _Captured(Exception):
    """Stop after Qwen's multimodal frontend and before its language decoder."""


def fused_inputs(model: Any, batch: dict[str, Tensor]) -> dict[str, Any]:
    """Retain native fused embeddings, masks and multimodal rotary positions."""
    captured = {}

    def capture(_module, _args, kwargs):
        captured.update({key: value.detach() if torch.is_tensor(value) else value
                         for key, value in kwargs.items()})
        raise _Captured

    handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**batch, use_cache=False, return_dict=True)
    except _Captured:
        pass
    finally:
        handle.remove()
    if "inputs_embeds" not in captured or captured["past_key_values"] is not None:
        raise ValueError("Expected an uncached Qwen2.5 multimodal prefill")
    return captured


class QwenPreference:
    """Conditional response preference on one shared fused image/text prompt."""

    def __init__(self, model: Any, prompt: dict[str, Tensor],
                 positive: dict[str, Tensor], negative: dict[str, Tensor]):
        if model.config.model_type != "qwen2_5_vl" or model.training:
            raise ValueError("This paper bridge requires Qwen2.5-VL in eval mode")
        for batch in (prompt, positive, negative):
            if batch["input_ids"].shape[0] != 1 or not bool(batch["attention_mask"].all()):
                raise ValueError("Paper extraction requires one unpadded example")
        self.model = model
        self.length = prompt["input_ids"].shape[1]
        for batch in (positive, negative):
            if not torch.equal(batch["input_ids"][:, :self.length], prompt["input_ids"]):
                raise ValueError("Response tokenization changed the shared prompt")
            if batch["input_ids"].shape[1] <= self.length:
                raise ValueError("A preference response must contain tokens")
        self.prompt = fused_inputs(model, prompt)
        self.completions = [(fused_inputs(model, batch), batch["input_ids"][:, self.length:])
                            for batch in (positive, negative)]
        for inputs, _ in self.completions:
            if not torch.equal(inputs["inputs_embeds"][:, :self.length], self.embeddings):
                raise ValueError("The two responses do not share identical fused input embeddings")

    @property
    def embeddings(self) -> Tensor:
        return self.prompt["inputs_embeds"]

    def response_logp(self, completion: tuple[dict[str, Any], Tensor], prompt: Tensor) -> Tensor:
        """Sum log P(response token | prompt, preceding response), with causal shift."""
        inputs, targets = completion
        embeddings = torch.cat((prompt, inputs["inputs_embeds"][:, self.length:]), dim=1)
        hidden = self.model.model.language_model(**{**inputs, "inputs_embeds": embeddings}).last_hidden_state
        logits = self.model.lm_head(hidden[:, self.length - 1:-1]).float()
        return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).sum()

    def objective(self, prompt: Tensor) -> Tensor:
        return self.response_logp(self.completions[0], prompt) - self.response_logp(self.completions[1], prompt)

    def integrated_gradients(self, steps: int = 5) -> Tensor:
        """Zero-baseline Gauss-Legendre IG for the common prompt, including visual patches."""
        if steps < 1:
            raise ValueError("IG requires at least one quadrature step")
        accumulated = torch.zeros_like(self.embeddings, dtype=torch.float32)
        nodes, weights = np.polynomial.legendre.leggauss(steps)
        with torch.enable_grad():
            for node, weight in zip(nodes, weights):
                variable = (self.embeddings * float((node + 1) / 2)).detach().requires_grad_(True)
                # Linearity of differentiation lets us release each response graph
                # before building the other, reducing real-7B activation memory.
                for completion, sign in zip(self.completions, (1., -1.)):
                    gradient, = torch.autograd.grad(self.response_logp(completion, variable), variable)
                    accumulated += sign * float(weight / 2) * gradient.float()
        return (self.embeddings.float() * accumulated).sum(-1)[0].detach()

    def last_states(self, embeddings: Tensor) -> dict[int, Tensor]:
        """Decoder block outputs at the prompt's last token, before final model norm."""
        states, handles = {}, []

        def hook(index):
            def capture(_module, _args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                states[index] = tensor[0, -1].detach().float().cpu().clone()
            return capture

        try:
            for index, layer in enumerate(self.model.model.language_model.layers):
                handles.append(layer.register_forward_hook(hook(index)))
            with torch.no_grad():
                self.model.model.language_model(**{**self.prompt, "inputs_embeds": embeddings})
        finally:
            for handle in handles:
                handle.remove()
        return states


def signed_indices(scores: Tensor, top_k: int, positive: bool) -> Tensor:
    """Choose only positive or only negative prompt attributions, up to k each."""
    if top_k < 1:
        raise ValueError("Token selection requires top_k >= 1")
    valid = torch.where(scores > 0 if positive else scores < 0)[0]
    order = torch.argsort(scores[valid], descending=positive, stable=True)
    return valid[order[:top_k]]


def polarity_vectors(positive: dict[int, Tensor], negative: dict[int, Tensor]) -> dict[int, Tensor]:
    """Equations3–4: independent centered PC1 per polarity, followed by subtraction."""
    if positive.keys() != negative.keys():
        raise ValueError("Polarity layers differ")
    vectors = {}
    for layer in positive:
        components = []
        for samples in (positive[layer], negative[layer]):
            if samples.ndim != 2 or samples.shape[0] < 2 or not bool(torch.isfinite(samples).all()):
                raise ValueError("PCA requires at least two finite delta vectors")
            fitted = PCA(n_components=1, svd_solver="full").fit(samples.detach().float().cpu().numpy())
            if not np.isfinite(fitted.explained_variance_[0]) or fitted.explained_variance_[0] <= 0:
                raise ValueError("Polarity deltas have no PCA variance")
            components.append(torch.from_numpy(fitted.components_[0]))
        vectors[layer] = components[0] - components[1]
    return vectors


def extract_example(adapter: Any, record: dict, *, steps: int = 5, top_k: int = 20) -> dict:
    """One resumable extraction unit, retaining raw deltas and selection evidence."""
    prompt = adapter.prepare(record["image"], record["prompt"])
    responses = [adapter.prepare(record["image"], record["prompt"], record[key], append_eos=False)
                 for key in ("chosen", "rejected")]
    preference = QwenPreference(adapter.model, prompt, *responses)
    scores = preference.integrated_gradients(steps)
    base = preference.last_states(preference.embeddings)
    selections = [signed_indices(scores, top_k, positive) for positive in (True, False)]
    deltas = []
    for indices in selections:
        masked = preference.embeddings.clone()
        masked[:, indices] = 0
        ablated = preference.last_states(masked)
        deltas.append({layer: base[layer] - ablated[layer] for layer in base})
    return {"id": record["id"], "scores": scores.cpu(),
            "prompt_input_ids": prompt["input_ids"].cpu(),
            "image_indices": torch.where(prompt["input_ids"][0] == adapter.model.config.image_token_id)[0].cpu(),
            "positive_indices": selections[0].cpu(), "negative_indices": selections[1].cpu(),
            "positive_deltas": deltas[0], "negative_deltas": deltas[1]}


def extract(adapter: Any, records: list[dict], *, steps: int = 5, top_k: int = 20) -> dict:
    """Preserve raw polarity deltas and prompt IG, then construct all-layer directions."""
    layers = list(range(len(adapter.layers)))
    deltas = [{index: [] for index in layers} for _ in range(2)]
    attributions = []
    training = [(module, module.training) for module in adapter.model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in adapter.model.parameters()]
    adapter.model.eval()
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        for record in records:
            example = extract_example(adapter, record, steps=steps, top_k=top_k)
            for destination, key in zip(deltas, ("positive_deltas", "negative_deltas")):
                for layer in layers:
                    destination[layer].append(example[key][layer])
            attributions.append({key: value for key, value in example.items()
                                 if key not in ("positive_deltas", "negative_deltas")})
    finally:
        for module, state in training:
            module.training = state
        for parameter, state in parameters:
            parameter.requires_grad_(state)
    positive, negative = [{layer: torch.stack(values) for layer, values in group.items()}
                          for group in deltas]
    return {"vectors": polarity_vectors(positive, negative), "positive_deltas": positive,
            "negative_deltas": negative, "attributions": attributions,
            "profile": "paper-equations-qwen2.5", "ig_steps": steps, "top_k": top_k,
            "pca_sign_convention": "scikit-learn full SVD, independent polarities"}


@contextmanager
def steering(model: Any, vectors: dict[int, Tensor], strength: float):
    """Equation5 on every decoder layer/position, with direct lambda and token norms."""
    layers = model.model.language_model.layers
    if set(vectors) != set(range(len(layers))):
        raise ValueError("Paper steering requires a direction for every decoder layer")
    handles = []

    def hook(vector):
        def apply(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            candidate = hidden.float() + strength * vector.to(hidden.device).float()
            norm = candidate.norm(dim=-1, keepdim=True)
            if not bool(torch.isfinite(candidate).all()) or bool((norm == 0).any()):
                raise ValueError("Undefined or nonfinite normalized steering activation")
            updated = (candidate * hidden.float().norm(dim=-1, keepdim=True) / norm).to(hidden.dtype)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return apply

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(hook(vectors[index])))
        yield
    finally:
        for handle in handles:
            handle.remove()
